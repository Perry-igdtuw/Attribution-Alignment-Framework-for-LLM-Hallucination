"""
output_comparator.py
--------------------
Measures the semantic and distributional shift between original and perturbed outputs.

Three complementary shift measures are used:
1. JS Divergence:         Distributional shift at token log-prob level (information-theoretic)
2. Semantic Similarity:   Embedding-space distance (captures meaning change)
3. ROUGE-L Delta:         Surface-level overlap change (lexical shift)

Why all three? Each captures a different aspect of output change:
- JS divergence is sensitive to ANY output change at probability level
- Semantic sim catches meaning changes even if wording differs
- ROUGE captures exact string changes (conservative measure)

The CIS (Causal Impact Score) uses a weighted combination of these.

References:
  - Lin & Och (2004) "ROUGE: A Package for Automatic Summary Evaluation"
  - Reimers & Gurevych (2019) "Sentence-BERT"
  - DeYoung et al. (2020) "ERASER Benchmark"
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class OutputShift:
    """
    Measures of output change after causal perturbation.
    """
    original_output: str
    perturbed_output: str
    strategy: str

    js_divergence: float            # Jensen-Shannon divergence ∈ [0, 1]
    semantic_distance: float        # 1 - cosine_sim ∈ [0, 1]
    rouge_l_delta: float            # Drop in ROUGE-L score ∈ [0, 1]

    # Combined causal impact (weighted average)
    combined_shift: float           # ∈ [0, 1], used as CIS


class OutputComparator:
    """
    Computes shift between original and perturbed LLM outputs.

    Args:
        semantic_model_name: SentenceTransformer model for semantic similarity.
        js_weight:           Weight for JS divergence in combined_shift.
        semantic_weight:     Weight for semantic distance.
        rouge_weight:        Weight for ROUGE-L delta.
    """

    def __init__(
        self,
        semantic_model_name: str = "all-MiniLM-L6-v2",
        js_weight: float = 0.4,
        semantic_weight: float = 0.4,
        rouge_weight: float = 0.2,
    ):
        self.js_weight = js_weight
        self.semantic_weight = semantic_weight
        self.rouge_weight = rouge_weight
        self._semantic_model = None
        self._semantic_model_name = semantic_model_name

    def _load_semantic_model(self):
        """Lazy load — only loads on first use to save startup time."""
        if self._semantic_model is None:
            from sentence_transformers import SentenceTransformer
            self._semantic_model = SentenceTransformer(self._semantic_model_name)
            logger.info(f"Loaded SentenceTransformer: {self._semantic_model_name}")

    def compare(
        self,
        original_output: str,
        perturbed_output: str,
        strategy: str,
        original_log_probs: Optional[List[float]] = None,
        perturbed_log_probs: Optional[List[float]] = None,
    ) -> OutputShift:
        """
        Compute all shift measures between original and perturbed outputs.

        Args:
            original_output:      Model's response to original prompt.
            perturbed_output:     Model's response to perturbed prompt.
            strategy:             Which masking strategy was used.
            original_log_probs:   Log-probs of original output tokens (optional, for JS).
            perturbed_log_probs:  Log-probs of perturbed output tokens (optional, for JS).

        Returns:
            OutputShift with all measures populated.
        """
        js_div = self._compute_js_divergence(original_log_probs, perturbed_log_probs)
        sem_dist = self._compute_semantic_distance(original_output, perturbed_output)
        rouge_delta = self._compute_rouge_delta(original_output, perturbed_output)

        combined = (
            self.js_weight * js_div
            + self.semantic_weight * sem_dist
            + self.rouge_weight * rouge_delta
        )
        combined = float(np.clip(combined, 0.0, 1.0))

        return OutputShift(
            original_output=original_output,
            perturbed_output=perturbed_output,
            strategy=strategy,
            js_divergence=js_div,
            semantic_distance=sem_dist,
            rouge_l_delta=rouge_delta,
            combined_shift=combined,
        )

    def _compute_js_divergence(
        self,
        log_probs_p: Optional[List[float]],
        log_probs_q: Optional[List[float]],
    ) -> float:
        """
        Jensen-Shannon divergence between two log-prob distributions.

        JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = (P+Q)/2

        If log-probs not available (Ollama mode), returns 0.0 as fallback.
        JS ∈ [0, 1] when using log base 2.
        """
        if log_probs_p is None or log_probs_q is None:
            return 0.0

        # Convert log-probs to probability distributions (softmax over token probs)
        p = np.exp(np.array(log_probs_p, dtype=np.float64))
        q = np.exp(np.array(log_probs_q, dtype=np.float64))

        # Align lengths (pad shorter with small epsilon)
        max_len = max(len(p), len(q))
        eps = 1e-10
        p = np.pad(p, (0, max_len - len(p)), constant_values=eps)
        q = np.pad(q, (0, max_len - len(q)), constant_values=eps)

        # Normalize to valid probability distributions
        p /= p.sum()
        q /= q.sum()

        m = 0.5 * (p + q)

        def kl(a, b):
            return np.sum(a * np.log2(a / b + eps))

        js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
        return float(np.clip(js, 0.0, 1.0))

    def _compute_semantic_distance(self, text_a: str, text_b: str) -> float:
        """
        Semantic distance = 1 - cosine_similarity(embed(A), embed(B)).
        Uses SentenceTransformers for sentence-level embeddings.

        Range: [0, 1] where 0 = identical meaning, 1 = completely different.
        """
        if not text_a.strip() or not text_b.strip():
            return 1.0  # Empty output = maximum shift

        self._load_semantic_model()

        embeddings = self._semantic_model.encode(
            [text_a, text_b], normalize_embeddings=True
        )
        cosine_sim = float(np.dot(embeddings[0], embeddings[1]))
        return float(np.clip(1.0 - cosine_sim, 0.0, 1.0))

    def _compute_rouge_delta(self, original: str, perturbed: str) -> float:
        """
        Computes the DROP in ROUGE-L score from original to perturbed.
        A high drop means the perturbed output shares less content with original.

        ROUGE-L uses longest common subsequence (LCS) — more robust than ROUGE-1/2.

        Returns: 1 - ROUGE-L(original, perturbed) ∈ [0, 1]
        """
        if not original.strip() or not perturbed.strip():
            return 1.0

        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            scores = scorer.score(original, perturbed)
            rouge_l = scores["rougeL"].fmeasure
            return float(np.clip(1.0 - rouge_l, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"ROUGE computation failed: {e}")
            return 0.0
