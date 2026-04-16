"""
token_extractor.py
------------------
Extracts token-level information from LLM outputs for XAI and metric computation.

Extracted signals:
1. Token log-probabilities   → used for HCG (Hallucination Confidence Gap)
2. Attention weight matrices → used for attention-based attribution (AAS)
3. Input/output gradients    → used for gradient × input attribution

Design notes:
- All extraction is done in a SINGLE forward pass to minimize memory overhead.
- Attention weights are averaged across heads (then optionally across layers)
  using attention rollout (Abnar & Zuidema, 2020) rather than naive averaging,
  which better captures information flow through transformer layers.
- Log-probs are computed over the ANSWER tokens only (not full sequence),
  following standard practice in hallucination detection literature.

References:
  - Abnar & Zuidema (2020) "Quantifying Attention Flow in Transformers"
  - Kadavath et al. (2022) "Language Models (Mostly) Know What They Know"
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class TokenExtractionResult:
    """
    Complete token-level information for a single LLM response.
    All fields are populated in one forward pass.
    """
    input_tokens: List[str]
    output_tokens: List[str]
    all_tokens: List[str]                       # input + output concatenated

    # Log-probabilities (output tokens only)
    output_log_probs: List[float]               # Per output token log-prob
    mean_log_prob: float                        # Mean over output tokens
    answer_confidence: float                    # exp(mean_log_prob) → [0, 1]

    # Attention (averaged across layers and heads via rollout)
    attention_rollout: Optional[np.ndarray]     # Shape: (n_input_tokens,)
    raw_attention: Optional[List[np.ndarray]]   # Shape: (n_layers, n_heads, seq, seq)

    # Gradients (populated only when compute_gradients=True)
    gradient_scores: Optional[np.ndarray]       # Shape: (n_input_tokens,) — grad × input norm


class TokenExtractor:
    """
    Extracts token-level signals from a HuggingFace causal LM.

    Usage:
        extractor = TokenExtractor(model, tokenizer)
        result = extractor.extract(prompt, generated_text)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        # Note: model placement is handled by LLMClient._load_huggingface()

    def extract(
        self,
        input_text: str,
        generated_text: str,
        compute_gradients: bool = False,
    ) -> TokenExtractionResult:
        """
        Main extraction method. Runs a forward pass over [input + generated_text]
        and extracts all signals.

        Args:
            input_text:      The prompt fed to the model.
            generated_text:  The model's generated response (answer + explanation).
            compute_gradients: Whether to compute gradient × input scores.
                               More expensive; use only when needed for XAI.

        Returns:
            TokenExtractionResult with all populated fields.
        """
        # ── 1. Tokenize full sequence ────────────────────────────────────────
        full_text = input_text + generated_text
        full_enc = self.tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.device)

        input_enc = self.tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.device)

        n_input_tokens = input_enc["input_ids"].shape[1]
        n_total_tokens = full_enc["input_ids"].shape[1]
        n_output_tokens = n_total_tokens - n_input_tokens

        input_tokens = self.tokenizer.convert_ids_to_tokens(
            input_enc["input_ids"][0].tolist()
        )
        full_token_ids = full_enc["input_ids"][0].tolist()
        output_token_ids = full_token_ids[n_input_tokens:]
        output_tokens = self.tokenizer.convert_ids_to_tokens(output_token_ids)
        all_tokens = self.tokenizer.convert_ids_to_tokens(full_token_ids)

        # ── 2. Forward pass with attention + gradient hooks ──────────────────
        if compute_gradients:
            # Enable gradients on input embeddings for gradient × input
            self.model.zero_grad()
            embeddings = self.model.get_input_embeddings()(full_enc["input_ids"])
            embeddings = embeddings.detach().requires_grad_(True)

            outputs = self.model(
                inputs_embeds=embeddings,
                attention_mask=full_enc["attention_mask"],
                output_attentions=True,
                return_dict=True,
            )
        else:
            with torch.no_grad():
                outputs = self.model(
                    **full_enc,
                    output_attentions=True,
                    return_dict=True,
                )

        logits = outputs.logits           # (1, seq_len, vocab_size)
        attentions = outputs.attentions   # Tuple of (1, n_heads, seq, seq) per layer

        # ── 3. Extract log-probabilities for output tokens ───────────────────
        # Shift: logits[i] predicts token[i+1]
        output_log_probs = self._extract_output_log_probs(
            logits, full_enc["input_ids"], n_input_tokens
        )
        mean_log_prob = float(np.mean(output_log_probs)) if output_log_probs else -np.inf
        answer_confidence = float(np.exp(mean_log_prob))

        # ── 4. Compute attention rollout ─────────────────────────────────────
        attention_rollout, raw_attention = self._compute_attention_rollout(
            attentions, n_input_tokens
        )

        # ── 5. Compute gradient × input scores (optional) ────────────────────
        gradient_scores = None
        if compute_gradients:
            gradient_scores = self._compute_gradient_input_scores(
                outputs.logits, embeddings, n_input_tokens, full_enc["input_ids"]
            )

        return TokenExtractionResult(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            all_tokens=all_tokens,
            output_log_probs=output_log_probs,
            mean_log_prob=mean_log_prob,
            answer_confidence=answer_confidence,
            attention_rollout=attention_rollout,
            raw_attention=raw_attention,
            gradient_scores=gradient_scores,
        )

    def _extract_output_log_probs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        n_input_tokens: int,
    ) -> List[float]:
        """
        Extract per-token log-probabilities for generated (output) tokens.

        Shift logic:
            logits[:, i, :] → distribution over token at position i+1
            So for output tokens starting at position n_input_tokens,
            we look at logits[:, n_input_tokens-1 : seq_len-1, :]
        """
        log_probs = F.log_softmax(logits[0], dim=-1)  # (seq_len, vocab_size)
        n_total = input_ids.shape[1]

        result = []
        for i in range(n_input_tokens, n_total):
            token_id = input_ids[0, i].item()
            lp = log_probs[i - 1, token_id].item()
            result.append(lp)

        return result

    def _compute_attention_rollout(
        self,
        attentions: Tuple[torch.Tensor, ...],
        n_input_tokens: int,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Implements attention rollout (Abnar & Zuidema, 2020).

        Standard averaging across heads is misleading because it doesn't
        account for the residual connection: each layer sees BOTH the
        attended information AND a direct copy of the input (identity matrix).

        Rollout:
            A_rollout[0] = I
            A_rollout[l] = A_rollout[l-1] · (0.5 * A[l] + 0.5 * I)
        where A[l] is the mean-head attention at layer l.

        Final: we take the last row of A_rollout corresponding to the
        last output token's attention over input tokens.
        """
        n_layers = len(attentions)
        seq_len = attentions[0].shape[-1]

        # Collect mean-head attention per layer: (n_layers, seq, seq)
        raw_attention = []
        mean_head_attentions = []
        for layer_attn in attentions:
            # layer_attn: (1, n_heads, seq, seq)
            layer_np = layer_attn[0].detach().cpu().float().numpy()
            raw_attention.append(layer_np)
            mean_heads = layer_np.mean(axis=0)  # (seq, seq)
            mean_head_attentions.append(mean_heads)

        # Rollout computation
        rollout = np.eye(seq_len)
        for mean_attn in mean_head_attentions:
            # Add residual connection: 0.5 * A + 0.5 * I
            augmented = 0.5 * mean_attn + 0.5 * np.eye(seq_len)
            # Normalize rows
            augmented /= augmented.sum(axis=-1, keepdims=True)
            rollout = rollout @ augmented

        # Take attention from last token → all input tokens
        last_token_rollout = rollout[-1, :n_input_tokens]

        # Normalize to [0, 1]
        if last_token_rollout.max() > 0:
            last_token_rollout = last_token_rollout / last_token_rollout.max()

        return last_token_rollout, raw_attention

    def _compute_gradient_input_scores(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        n_input_tokens: int,
        input_ids: torch.Tensor,
    ) -> np.ndarray:
        """
        Gradient × Input attribution for input tokens.

        Gradient × Input is more reliable than raw gradients for attribution
        because it accounts for the actual magnitude of the input values,
        reducing the "gradient saturation" problem in deep networks.

        Method:
            score[i] = ||grad(L, e_i) * e_i||_2
        where e_i is the embedding of input token i.

        L = log_prob of the most likely output token at the first output position.
        """
        # Use first output token's prediction as the loss target
        first_output_logit = logits[0, n_input_tokens - 1, :]
        target_id = first_output_logit.argmax()
        loss = -F.log_softmax(first_output_logit, dim=-1)[target_id]
        loss.backward()

        # Gradient × Input: (seq_len, hidden_dim) → norm over hidden dim
        grad = embeddings.grad[0].detach().cpu().float().numpy()  # (seq, hidden)
        emb = embeddings[0].detach().cpu().float().numpy()        # (seq, hidden)

        grad_input = grad * emb  # element-wise product
        scores = np.linalg.norm(grad_input, axis=-1)  # (seq,)

        # Return only input token scores, normalized
        input_scores = scores[:n_input_tokens]
        if input_scores.max() > 0:
            input_scores = input_scores / input_scores.max()

        return input_scores
