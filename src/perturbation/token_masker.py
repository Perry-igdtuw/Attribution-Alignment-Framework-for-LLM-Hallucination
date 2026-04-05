"""
token_masker.py
---------------
Token masking strategies for causal perturbation analysis.

Three strategies are implemented (Feder et al., 2021):
1. MASK:    Replace tokens with [MASK] or model's pad token.
            → Tests if the model still "works around" the removed information.
2. DELETE:  Remove tokens entirely, shortening the sequence.
            → More aggressive; tests structural dependence.
3. REPLACE: Replace with random tokens from vocabulary.
            → Adds noise; useful for verifying robustness.

Why all three?
- Each strategy has different failure modes:
  * Mask can still be decoded by the model if it learns [MASK]→context
  * Delete changes sequence length, affecting positional encodings
  * Replace introduces OOD behavior
- Using all three and averaging their CIS gives a more robust causal estimate.
- This mirrors ablation methodology in (DeYoung et al., 2020) ERASER benchmark.

References:
  - Feder et al. (2021) "CausaLM: Causal Model Explanation"
  - DeYoung et al. (2020) "ERASER: A Benchmark to Evaluate Rationalized NLP Models"
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Set


class MaskStrategy(str, Enum):
    MASK = "mask"
    DELETE = "delete"
    REPLACE = "replace"


@dataclass
class MaskedText:
    """Result of applying a masking strategy to explanation tokens in a prompt."""
    original_text: str
    masked_text: str
    strategy: MaskStrategy
    masked_tokens: List[str]       # The tokens that were masked/removed
    n_tokens_affected: int


class TokenMasker:
    """
    Applies causal perturbation by masking explanation-relevant tokens in the prompt.

    The key insight: if explanation tokens are CAUSALLY responsible for the answer,
    masking them should significantly change the output. If the output barely changes,
    the explanation is not faithful (it's a post-hoc rationalization).

    Usage:
        masker = TokenMasker(mask_token="[MASK]")
        masked = masker.mask(
            original_prompt="What is the capital of France?",
            tokens_to_mask=["capital", "France"],
            strategy=MaskStrategy.MASK
        )
    """

    # Common English stopwords to NEVER mask (they carry no semantic info)
    STOPWORDS: Set[str] = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "of", "in", "on", "at", "to", "for", "with", "by", "from",
        "it", "its", "this", "that", "and", "or", "but", "not", "no",
        "what", "when", "where", "who", "how", "which", "why",
    }

    def __init__(
        self,
        mask_token: str = "[MASK]",
        vocab_size: int = 1000,
        random_seed: int = 42,
    ):
        self.mask_token = mask_token
        self.vocab_size = vocab_size
        self.rng = random.Random(random_seed)

        # Small replacement vocabulary (common English nouns — safe to sample from)
        self._replace_vocab = [
            "object", "thing", "entity", "concept", "element", "factor",
            "aspect", "feature", "property", "attribute", "item", "point",
            "unit", "part", "section", "type", "form", "kind", "sort", "class",
        ]

    def mask(
        self,
        original_text: str,
        tokens_to_mask: List[str],
        strategy: MaskStrategy = MaskStrategy.MASK,
    ) -> MaskedText:
        """
        Apply masking strategy to specified tokens in the text.

        Args:
            original_text:   The full prompt or text to perturb.
            tokens_to_mask:  List of token/word strings to remove/replace.
                             Should be the top-K attribution tokens (from AAS).
            strategy:        Which masking strategy to apply.

        Returns:
            MaskedText with both original and perturbed versions.
        """
        # Filter out stopwords from tokens to mask
        effective_tokens = [
            t for t in tokens_to_mask
            if t.lower() not in self.STOPWORDS and len(t) > 1
        ]

        if not effective_tokens:
            return MaskedText(
                original_text=original_text,
                masked_text=original_text,
                strategy=strategy,
                masked_tokens=[],
                n_tokens_affected=0,
            )

        if strategy == MaskStrategy.MASK:
            masked_text = self._apply_mask(original_text, effective_tokens)
        elif strategy == MaskStrategy.DELETE:
            masked_text = self._apply_delete(original_text, effective_tokens)
        elif strategy == MaskStrategy.REPLACE:
            masked_text = self._apply_replace(original_text, effective_tokens)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return MaskedText(
            original_text=original_text,
            masked_text=masked_text,
            strategy=strategy,
            masked_tokens=effective_tokens,
            n_tokens_affected=len(effective_tokens),
        )

    def mask_explanation_in_prompt(
        self,
        prompt: str,
        explanation: str,
        attribution_result,      # AttributionResult
        strategy: MaskStrategy = MaskStrategy.MASK,
        use_top_k: Optional[int] = None,
    ) -> MaskedText:
        """
        High-level method: mask the explanation's top-K attribution tokens
        within the larger prompt context.

        This is the primary method used by causal_engine.py.

        Args:
            prompt:              Full prompt text.
            explanation:         Model's generated explanation.
            attribution_result:  AttributionResult from any XAI method.
            strategy:            Masking strategy.
            use_top_k:           Override top_k (default: use all top_k_tokens).
        """
        top_tokens = attribution_result.top_k_tokens
        if use_top_k is not None:
            top_tokens = top_tokens[:use_top_k]

        return self.mask(prompt, top_tokens, strategy)

    # ── Private Strategy Implementations ─────────────────────────────────────

    def _apply_mask(self, text: str, tokens: List[str]) -> str:
        """Replace each token occurrence with mask_token (case-insensitive)."""
        result = text
        for token in tokens:
            pattern = r"\b" + re.escape(token) + r"\b"
            result = re.sub(pattern, self.mask_token, result, flags=re.IGNORECASE)
        return result

    def _apply_delete(self, text: str, tokens: List[str]) -> str:
        """Remove each token occurrence entirely (leaves a space gap)."""
        result = text
        for token in tokens:
            pattern = r"\s*\b" + re.escape(token) + r"\b\s*"
            result = re.sub(pattern, " ", result, flags=re.IGNORECASE)
        # Clean up multiple spaces
        result = re.sub(r"\s{2,}", " ", result).strip()
        return result

    def _apply_replace(self, text: str, tokens: List[str]) -> str:
        """Replace each token with a random word from replacement vocabulary."""
        result = text
        for token in tokens:
            replacement = self.rng.choice(self._replace_vocab)
            pattern = r"\b" + re.escape(token) + r"\b"
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result
