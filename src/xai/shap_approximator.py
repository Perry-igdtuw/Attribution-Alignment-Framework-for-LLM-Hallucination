"""
shap_approximator.py
--------------------
SHAP-based text attribution via random token masking.

Why SHAP for text (not TreeExplainer or KernelExplainer)?
- TreeExplainer doesn't apply to LLMs.
- KernelExplainer is model-agnostic but extremely slow (thousands of forward passes).
- We implement a lightweight SHAP approximation via random token masking, which
  approximates Shapley values by sampling coalitions and measuring marginal contribution.

This is equivalent to the approach used by:
- SHAP's `Explainer` with `masker=Text` (word-level, not subword)
- LIMEText (Ribeiro et al., 2016) with logistic output model

Computational note:
- Full Shapley requires 2^n forward passes (n = tokens). We sample n_samples
  coalitions instead. n_samples=100 is a good balance for 20-token inputs.

References:
  - Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"
  - Ribeiro et al. (2016) "Why Should I Trust You?" (LIME)
  - Kokalj et al. (2021) "BERT-based NLP Explainability" (text SHAP practices)
"""

from __future__ import annotations

import logging
import re
import numpy as np
from typing import Callable, List, Optional, Tuple

from src.xai.attribution_utils import AttributionResult, make_attribution_result

logger = logging.getLogger(__name__)

MASK_TOKEN = "[MASKED]"


class SHAPApproximator:
    """
    Approximates Shapley values for text input tokens via random coalition sampling.

    Works with ANY text generation function (model-agnostic).
    This makes it compatible with both HuggingFace and Ollama backends.

    Args:
        generate_fn:  Function mapping text → float (confidence score).
                      Typically: lambda prompt: llm_client.generate(prompt).answer_confidence
        n_samples:    Number of random coalition samples. Higher = more accurate but slower.
        top_k:        Number of top tokens to return in AttributionResult.

    Usage:
        shap = SHAPApproximator(generate_fn=my_fn, n_samples=100)
        result = shap.compute(prompt_text, question)
    """

    def __init__(
        self,
        generate_fn: Callable[[str], float],
        n_samples: int = 100,
        top_k: int = 10,
        random_seed: int = 42,
    ):
        self.generate_fn = generate_fn
        self.n_samples = n_samples
        self.top_k = top_k
        self.rng = np.random.default_rng(random_seed)

    def compute(
        self,
        prompt_text: str,
        reference_output: float,
    ) -> Optional[AttributionResult]:
        """
        Approximate Shapley values for each word token in the prompt.

        Args:
            prompt_text:       The full prompt text to analyze.
            reference_output:  The model's confidence score on the FULL prompt (baseline).
                               Used as the "grand coalition" value in Shapley computation.

        Returns:
            AttributionResult with approximate Shapley values as scores.
        """
        words = self._tokenize_words(prompt_text)
        n_words = len(words)

        if n_words == 0:
            return None

        if n_words > 50:
            logger.warning(
                f"Prompt has {n_words} words. SHAP will be slow. "
                "Consider truncating or reducing n_samples."
            )

        # Shapley value accumulator
        shapley_values = np.zeros(n_words)
        counts = np.zeros(n_words)

        for _ in range(self.n_samples):
            # Sample a random coalition (subset of word indices)
            coalition_mask = self.rng.random(n_words) > 0.5
            coalition_indices = np.where(coalition_mask)[0].tolist()

            # For each token NOT already in coalition, compute marginal contribution
            token_to_evaluate = self.rng.integers(0, n_words)

            # Value WITH token i
            coalition_with = coalition_indices.copy()
            if token_to_evaluate not in coalition_with:
                coalition_with.append(token_to_evaluate)
            text_with = self._mask_words_except(words, sorted(coalition_with))
            score_with = self._safe_predict(text_with)

            # Value WITHOUT token i
            coalition_without = [c for c in coalition_indices if c != token_to_evaluate]
            text_without = self._mask_words_except(words, coalition_without)
            score_without = self._safe_predict(text_without)

            # Marginal contribution = value delta
            shapley_values[token_to_evaluate] += score_with - score_without
            counts[token_to_evaluate] += 1

        # Average over samples (handle un-sampled tokens)
        counts = np.maximum(counts, 1)
        shapley_values /= counts

        # Shift values so minimum is 0 (Shapley can be negative)
        shapley_values -= shapley_values.min()

        return make_attribution_result(
            tokens=words,
            raw_scores=shapley_values,
            method="shap_approximation",
            top_k=self.top_k,
        )

    def _tokenize_words(self, text: str) -> List[str]:
        """Split prompt into word-level tokens (space-separated, not subword)."""
        return re.findall(r"\S+", text)

    def _mask_words_except(self, words: List[str], keep_indices: List[int]) -> str:
        """Return text with all words EXCEPT keep_indices replaced by MASK_TOKEN."""
        result = []
        for i, word in enumerate(words):
            if i in keep_indices:
                result.append(word)
            else:
                result.append(MASK_TOKEN)
        return " ".join(result)

    def _safe_predict(self, text: str) -> float:
        """Call generate_fn safely, returning 0.0 on error."""
        try:
            return float(self.generate_fn(text))
        except Exception as e:
            logger.debug(f"SHAP predict error: {e}")
            return 0.0
