"""
aas.py — Attribution Alignment Score
--------------------------------------
Measures how well the model's explanation tokens align with the tokens
identified as important by XAI attribution methods.

Formula:
    AAS = |E ∩ A_top_k| / |E ∪ A_top_k|    (Jaccard overlap)

where:
    E         = set of content words in the explanation
    A_top_k   = top-K tokens by attribution score (from attention/gradient/SHAP)

Design rationale:
- Jaccard is symmetric and bounded [0,1], unlike precision/recall which require
  choosing a "reference" set. Here, neither E nor A is the "ground truth".
- We use content words (excluding stopwords) to avoid trivial high overlap
  on function words that appear in both by coincidence.
- If multiple attribution methods are available, AAS is computed for each
  and the final AAS is the mean (reduces method-specific bias).

Interpretation:
- AAS → 1.0: explanation tokens closely match what the model attended to
- AAS → 0.0: explanation is disconnected from model's actual attention pattern
  → Potential faithfulness failure / hallucination signal

Paper note: This is conceptually related to the "sufficiency" metric in
ERASER (DeYoung et al., 2020) but applied at token-attribution level.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Set

from src.xai.attribution_utils import AttributionResult, tokenize_text_words


def compute_aas(
    explanation: str,
    attribution_results: List[AttributionResult],
    top_k: int = 10,
) -> float:
    """
    Compute Attribution Alignment Score (AAS).

    Args:
        explanation:          The model's generated explanation text.
        attribution_results:  List of AttributionResult from different XAI methods.
                              If multiple, AAS is averaged across methods.
        top_k:                Number of top attribution tokens to consider.

    Returns:
        AAS ∈ [0, 1]. Higher = more aligned = more faithful explanation.
    """
    if not explanation.strip() or not attribution_results:
        return 0.0

    # Parse explanation into content word set
    explanation_words: Set[str] = tokenize_text_words(explanation)

    if not explanation_words:
        return 0.0

    aas_per_method = []
    for attr_result in attribution_results:
        if attr_result is None:
            continue

        # Get top-K attribution tokens
        top_attr_tokens = set(attr_result.top_k_tokens[:top_k])

        # Jaccard similarity: |intersection| / |union|
        intersection = explanation_words & top_attr_tokens
        union = explanation_words | top_attr_tokens

        if len(union) == 0:
            continue

        jaccard = len(intersection) / len(union)
        aas_per_method.append(jaccard)

    if not aas_per_method:
        return 0.0

    return float(np.mean(aas_per_method))


def compute_aas_precision_recall(
    explanation: str,
    attribution_result: AttributionResult,
    top_k: int = 10,
) -> dict:
    """
    Extended AAS: also compute precision and recall components separately.
    Useful for detailed ablation analysis and error analysis in the paper.

    Precision = what fraction of attr tokens appear in the explanation?
    Recall    = what fraction of explanation words appear in attr tokens?
    F1        = harmonic mean (standard AAS proxy)
    """
    explanation_words = tokenize_text_words(explanation)
    top_attr_tokens = set(attribution_result.top_k_tokens[:top_k])

    if not explanation_words or not top_attr_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "jaccard": 0.0}

    tp = len(explanation_words & top_attr_tokens)
    precision = tp / len(top_attr_tokens) if top_attr_tokens else 0.0
    recall = tp / len(explanation_words) if explanation_words else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    jaccard = tp / len(explanation_words | top_attr_tokens)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
    }
