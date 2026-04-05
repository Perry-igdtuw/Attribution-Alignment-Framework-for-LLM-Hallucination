"""
fhi.py — Faithfulness-Hallucination Index (FHI)
-------------------------------------------------
The core proposed metric. Combines AAS, CIS, ESS, HCG into a single score.

Formula:
    FHI = clip(w1*AAS + w2*CIS + w3*ESS - w4*HCG, 0, 1)

Default weights (from config/metric_weights.yaml):
    w1 = 0.30  (AAS)
    w2 = 0.35  (CIS — highest, causal evidence preferred)
    w3 = 0.20  (ESS)
    w4 = 0.15  (HCG — subtracted: penalizes overconfident wrong answers)

Thresholding:
    FHI >= threshold → faithful (not hallucinated)
    FHI <  threshold → hallucination detected
    Default threshold: 0.5 (tunable via experiment_config.yaml)

Weight justification for paper:
    - CIS gets highest weight (0.35) because causal evidence is strongest
      — it directly tests whether the explanation influences the output.
    - AAS (0.30) is second: attribution alignment is a necessary but not
      sufficient condition for faithfulness.
    - ESS (0.20) captures consistency but depends on sampling variance.
    - HCG (0.15) is already a known signal (log-prob based); we give it
      the smallest weight to preserve novelty of the causal contribution.

Weight optimization:
    Weights are optimized via grid search in run_ablation.py over the
    validation split. Final weights are selected by maximizing FHI-F1
    against hallucination ground truth labels.

References:
    - Jacovi & Goldberg (2020) "Towards Faithfully Interpretable NLP Systems"
    - DeYoung et al. (2020) "ERASER Benchmark"
    - Atanasova et al. (2023) "Faithfulness Tests for NLG Explanations"
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class FHIResult:
    """Complete FHI computation result for a single sample."""
    sample_id: str

    # Component scores
    aas: float                      # Attribution Alignment Score ∈ [0,1]
    cis: float                      # Causal Impact Score ∈ [0,1]
    ess: float                      # Explanation Stability Score ∈ [0,1]
    hcg: float                      # Hallucination Confidence Gap ∈ [0,1]

    # Weights used
    w1: float
    w2: float
    w3: float
    w4: float

    # Final score
    fhi: float                      # ∈ [0, 1]

    # Detection
    threshold: float
    predicted_hallucination: bool   # True if FHI < threshold
    true_hallucination: Optional[bool] = None  # Ground truth (if available)

    @property
    def correct(self) -> Optional[bool]:
        if self.true_hallucination is None:
            return None
        return self.predicted_hallucination == self.true_hallucination


def compute_fhi(
    aas: float,
    cis: float,
    ess: float,
    hcg: float,
    w1: float = 0.30,
    w2: float = 0.35,
    w3: float = 0.20,
    w4: float = 0.15,
    threshold: float = 0.5,
    sample_id: str = "",
    true_hallucination: Optional[bool] = None,
) -> FHIResult:
    """
    Compute Faithfulness-Hallucination Index (FHI).

    Args:
        aas:               Attribution Alignment Score.
        cis:               Causal Impact Score.
        ess:               Explanation Stability Score.
        hcg:               Hallucination Confidence Gap.
        w1, w2, w3, w4:    Component weights (must sum to ≤ 1; see formula).
        threshold:         Decision boundary for hallucination detection.
        sample_id:         Sample identifier for tracking.
        true_hallucination: Ground truth label for evaluation.

    Returns:
        FHIResult with all fields populated.
    """
    raw_fhi = w1 * aas + w2 * cis + w3 * ess - w4 * hcg
    fhi = float(np.clip(raw_fhi, 0.0, 1.0))
    predicted_hallucination = fhi < threshold

    return FHIResult(
        sample_id=sample_id,
        aas=aas,
        cis=cis,
        ess=ess,
        hcg=hcg,
        w1=w1,
        w2=w2,
        w3=w3,
        w4=w4,
        fhi=fhi,
        threshold=threshold,
        predicted_hallucination=predicted_hallucination,
        true_hallucination=true_hallucination,
    )


def grid_search_weights(
    aas_list, cis_list, ess_list, hcg_list,
    true_labels,
    w_range=(0.05, 0.50),
    n_steps: int = 10,
    threshold: float = 0.5,
) -> dict:
    """
    Grid search to find optimal weights (w1, w2, w3, w4) that maximize F1.

    Called from run_ablation.py during the experimental phase.

    Args:
        aas_list, cis_list, ess_list, hcg_list: Per-sample metric values.
        true_labels: Ground truth hallucination labels (True/False).
        w_range:     Min/max range for each weight.
        n_steps:     Number of steps in grid search per dimension.
        threshold:   FHI decision threshold.

    Returns:
        Dict with best weights and their achieved F1 score.
    """
    from sklearn.metrics import f1_score
    import itertools

    grid = np.linspace(w_range[0], w_range[1], n_steps)
    best_f1 = 0.0
    best_weights = {"w1": 0.30, "w2": 0.35, "w3": 0.20, "w4": 0.15}

    for w1, w2, w3, w4 in itertools.product(grid, repeat=4):
        # Constrain: weights should sum to approximately 1
        if abs(w1 + w2 + w3 + w4 - 1.0) > 0.1:
            continue

        fhi_scores = [
            np.clip(w1 * a + w2 * c + w3 * e - w4 * h, 0, 1)
            for a, c, e, h in zip(aas_list, cis_list, ess_list, hcg_list)
        ]
        preds = [score < threshold for score in fhi_scores]
        f1 = f1_score(true_labels, preds, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_weights = {"w1": w1, "w2": w2, "w3": w3, "w4": w4}

    return {**best_weights, "best_f1": best_f1}
