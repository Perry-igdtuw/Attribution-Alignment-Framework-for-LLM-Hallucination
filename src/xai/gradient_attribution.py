"""
gradient_attribution.py
-----------------------
Gradient-based attribution using Captum (Integrated Gradients).

Why Integrated Gradients (IG) over vanilla gradients?
- Vanilla gradient × input can fail the completeness axiom: the sum of
  attributions may not equal the output difference from baseline.
- IG (Sundararajan et al., 2017) satisfies both completeness and sensitivity,
  making it theoretically sound for attribution.
- IG is computable for any differentiable model, unlike attention rollout
  which requires access to attention weights.

Implementation:
- We use captum.attr.IntegratedGradients applied to input embeddings.
- Baseline: zero embedding vector (standard choice, interpretable as "absence").
- n_steps=50 is sufficient for convergence in most transformer models.
- We L2-norm the attribution across the embedding dimension to get a
  scalar importance score per token.

References:
  - Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks"
  - Bastings & Filippova (2020) "The elephant in the interpretability room"
"""

from __future__ import annotations

import logging
import numpy as np
import torch
from typing import Optional

from src.model.llm_client import LLMResponse
from src.xai.attribution_utils import AttributionResult, make_attribution_result

logger = logging.getLogger(__name__)


class GradientAttribution:
    """
    Computes Integrated Gradients attribution using captum.

    Requires:
        - HuggingFace backend (model must be a PyTorch nn.Module)
        - pip install captum

    Usage:
        attr = GradientAttribution(model, tokenizer, device)
        result = attr.compute(response, prompt_text)
    """

    def __init__(self, model, tokenizer, device: str = "cpu", n_steps: int = 50, top_k: int = 10):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_steps = n_steps
        self.top_k = top_k

    def compute(
        self,
        response: LLMResponse,
        prompt_text: str,
    ) -> Optional[AttributionResult]:
        """
        Compute Integrated Gradients for the input tokens of a prompt.

        Args:
            response:    LLMResponse containing the generated answer.
            prompt_text: The original prompt text (needed to re-tokenize for IG).

        Returns:
            AttributionResult with IG attribution scores, or None on error.
        """
        try:
            from captum.attr import IntegratedGradients
        except ImportError:
            logger.warning("captum not installed. Run: pip install captum")
            return None

        if response.backend != "huggingface":
            logger.warning("Gradient attribution requires HuggingFace backend.")
            return None

        # ── Tokenize prompt ───────────────────────────────────────────────────
        enc = self.tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        n_input = input_ids.shape[1]

        # ── Define forward function for captum ────────────────────────────────
        # captum expects: forward(inputs_embeds) → scalar
        # We use the log-prob of the most likely next token as the target.
        embedding_layer = self.model.get_input_embeddings()

        def forward_fn(input_embeds: torch.Tensor) -> torch.Tensor:
            """Forward wrapper: embeddings → scalar log-prob."""
            out = self.model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )
            # Log-prob of top predicted token at last input position
            logits = out.logits[0, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            return log_probs.max().unsqueeze(0)

        # ── Get input embeddings ──────────────────────────────────────────────
        with torch.no_grad():
            input_embeds = embedding_layer(input_ids).detach()
            baseline_embeds = torch.zeros_like(input_embeds)  # Zero baseline

        input_embeds.requires_grad_(True)

        # ── Run Integrated Gradients ──────────────────────────────────────────
        ig = IntegratedGradients(forward_fn)
        attributions, convergence_delta = ig.attribute(
            inputs=input_embeds,
            baselines=baseline_embeds,
            n_steps=self.n_steps,
            return_convergence_delta=True,
        )
        # attributions: (1, seq_len, hidden_dim)

        # L2 norm over embedding dimension → (seq_len,)
        attr_np = attributions[0].detach().cpu().float().numpy()
        scores = np.linalg.norm(attr_np, axis=-1)  # (n_input,)

        if abs(convergence_delta.item()) > 0.1:
            logger.warning(
                f"IG convergence delta = {convergence_delta.item():.4f}. "
                "Consider increasing n_steps for more accurate attribution."
            )

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

        return make_attribution_result(
            tokens=tokens,
            raw_scores=scores,
            method="integrated_gradients",
            top_k=self.top_k,
        )

    def compute_grad_x_input(self, response: LLMResponse) -> Optional[AttributionResult]:
        """
        Simpler gradient × input attribution (no integration).
        Faster than IG but less theoretically principled.
        Used as a speed/quality tradeoff option.

        Falls back to precomputed gradient_scores if available in the response.
        """
        if response.gradient_scores is not None:
            # Already computed during forward pass in token_extractor.py
            return make_attribution_result(
                tokens=response.input_tokens,
                raw_scores=response.gradient_scores,
                method="gradient_x_input",
                top_k=self.top_k,
            )
        return None
