"""
cis.py — Causal Impact Score
-----------------------------
Measures the causal influence of explanation tokens on the model's output.

Formula:
    CIS = mean_over_strategies(combined_shift_score)

where combined_shift_score is the average of:
    - JS divergence between original and perturbed log-prob distributions
    - Semantic distance between original and perturbed output embeddings
    - ROUGE-L delta (lexical change)

Causal interpretation:
    CIS → 1.0: Removing explanation tokens causes large output change
               → Explanation is causally relevant (faithful)
    CIS → 0.0: Removing explanation tokens barely changes output
               → Explanation is a post-hoc decoration (hallucination risk)

This mirrors the "comprehensiveness" metric from ERASER (DeYoung et al., 2020),
but applied to free-form explanation tokens rather than rationale annotations.

Key design decision: CIS is computed as an AVERAGE across masking strategies
(mask, delete, replace) to reduce strategy-specific artifacts in the estimate.
"""

from __future__ import annotations

import numpy as np
from typing import List

from src.perturbation.output_comparator import OutputShift


def compute_cis(output_shifts: List[OutputShift]) -> float:
    """
    Compute Causal Impact Score (CIS) from a list of OutputShift results.

    Args:
        output_shifts: List of OutputShift objects, one per masking strategy.
                       Pass results from all three strategies for best estimate.

    Returns:
        CIS ∈ [0, 1]. Higher = explanation tokens are causally important.
    """
    if not output_shifts:
        return 0.0

    combined_shifts = [s.combined_shift for s in output_shifts]
    return float(np.mean(combined_shifts))


def compute_cis_by_strategy(output_shifts: List[OutputShift]) -> dict:
    """
    Breakdown of CIS per masking strategy.
    Useful for ablation analysis: which strategy is most informative?

    Returns:
        Dict mapping strategy name → combined_shift score.
    """
    return {s.strategy: s.combined_shift for s in output_shifts}


def compute_cis_components(output_shifts: List[OutputShift]) -> dict:
    """
    Detailed breakdown of all three shift components across strategies.
    Used for creating the component breakdown table in the paper.
    """
    if not output_shifts:
        return {
            "js_divergence": 0.0,
            "semantic_distance": 0.0,
            "rouge_l_delta": 0.0,
            "combined": 0.0,
        }

    return {
        "js_divergence": float(np.mean([s.js_divergence for s in output_shifts])),
        "semantic_distance": float(np.mean([s.semantic_distance for s in output_shifts])),
        "rouge_l_delta": float(np.mean([s.rouge_l_delta for s in output_shifts])),
        "combined": compute_cis(output_shifts),
    }
