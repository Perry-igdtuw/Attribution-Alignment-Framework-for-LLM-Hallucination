"""
ess.py — Explanation Stability Score
--------------------------------------
Measures how consistent the model's explanation is across multiple runs
of the same question (with identical generation parameters).

Formula:
    ESS = 1 - (1/C(n,2)) * sum_{i<j} semantic_distance(E_i, E_j)

where:
    E_i, E_j are explanations from runs i and j
    C(n,2) = number of pairwise combinations
    semantic_distance = 1 - cosine_sim(embed(E_i), embed(E_j))

Interpretation:
    ESS → 1.0: Explanations are highly consistent across runs (stable)
    ESS → 0.0: Explanations vary widely (unstable = unreliable reasoning)

Why stability matters for hallucination detection:
    - A model that truly "knows" something generates consistent explanations.
    - A hallucinating model often generates different explanations each time
      because the "reasoning" is not grounded in stable factual knowledge.
    - This relates to the "self-consistency" literature (Wang et al., 2022)
      but applied to explanation quality rather than answer voting.

Implementation note:
    We use SentenceTransformer embeddings rather than ROUGE for pairwise
    comparison because paraphrase-level consistency matters (not exact strings).

References:
    - Wang et al. (2022) "Self-Consistency Improves CoT Reasoning in LLMs"
    - Atanasova et al. (2023) "Faithfulness Tests for NLG Explanations"
"""

from __future__ import annotations

import logging
import numpy as np
from itertools import combinations
from typing import List

logger = logging.getLogger(__name__)


def compute_ess(
    explanations: List[str],
    semantic_model=None,
    semantic_model_name: str = "all-MiniLM-L6-v2",
) -> float:
    """
    Compute Explanation Stability Score (ESS) over N explanation runs.

    Args:
        explanations:         List of explanation strings (N runs of same question).
                              Must have at least 2 explanations.
        semantic_model:       Pre-loaded SentenceTransformer (reuse for efficiency).
        semantic_model_name:  Model name if semantic_model not provided.

    Returns:
        ESS ∈ [0, 1]. Higher = more stable explanations.
    """
    n = len(explanations)
    if n < 2:
        logger.warning("ESS requires at least 2 explanations. Returning 1.0 (trivially stable).")
        return 1.0

    # Filter empty explanations
    valid = [e for e in explanations if e.strip()]
    if len(valid) < 2:
        return 0.0

    # Load semantic model if not provided
    if semantic_model is None:
        from sentence_transformers import SentenceTransformer
        semantic_model = SentenceTransformer(semantic_model_name)

    # Encode all explanations at once (batch for efficiency)
    embeddings = semantic_model.encode(valid, normalize_embeddings=True)

    # Pairwise cosine similarity → distance
    pairwise_distances = []
    for (i, j) in combinations(range(len(valid)), 2):
        cosine_sim = float(np.dot(embeddings[i], embeddings[j]))
        # Cosine sim ∈ [-1, 1], distance = 1 - sim, clipped to [0, 1]
        distance = float(np.clip(1.0 - cosine_sim, 0.0, 1.0))
        pairwise_distances.append(distance)

    mean_distance = float(np.mean(pairwise_distances))
    ess = 1.0 - mean_distance

    return float(np.clip(ess, 0.0, 1.0))


def compute_ess_from_responses(responses: List, semantic_model=None) -> float:
    """
    Convenience wrapper: extract explanations from LLMResponse list, then compute ESS.

    Args:
        responses: List of LLMResponse objects (from llm_client.generate_multiple).
    """
    explanations = [r.explanation for r in responses if r.explanation]
    return compute_ess(explanations, semantic_model=semantic_model)
