"""
evaluator.py
------------
Statistical evaluator for comparing FHI performance against baselines.
Computes Precision, Recall, F1, and AUC-ROC for hallucination detection.
"""

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from typing import List, Dict, Any

class SystemEvaluator:
    """
    Takes predictions from FHI and Baselines alongside ground truth labels
    and computes standard classification metrics.
    """
    
    @staticmethod
    def evaluate(
        y_true: List[bool], 
        fhi_scores: List[float], 
        logprob_preds: List[bool], 
        self_consistency_scores: List[float],
        fhi_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        y_true: True if hallucination, False if factual.
        fhi_scores: Continuous FHI scores (higher = more faithful / less hallucinated).
        logprob_preds: Boolean. True if logprob baseline flagged hallucination.
        self_consistency_scores: Continuous risk scores.
        """
        # Convert continuous FHI scores into boolean hallucination predictions
        # Low FHI means hallucination
        fhi_preds = [score < fhi_threshold for score in fhi_scores]
        
        # Handle AUC-ROC where FHI is inverted (we want hallucination probability)
        # 1.0 - FHI = pseudo-probability of hallucination
        fhi_probs = [1.0 - score for score in fhi_scores]
        
        # Calculate reports
        report = {
            "FHI": classification_report(y_true, fhi_preds, output_dict=True, zero_division=0),
            "LogProb": classification_report(y_true, logprob_preds, output_dict=True, zero_division=0),
        }
        
        # Calculate AUC-ROC if there's variance in the labels 
        # (needs both True and False in y_true)
        if len(set(y_true)) > 1:
            report["FHI_AUC"] = float(roc_auc_score(y_true, fhi_probs))
            report["SelfConsistency_AUC"] = float(roc_auc_score(y_true, self_consistency_scores))
        else:
            report["FHI_AUC"] = None
            report["SelfConsistency_AUC"] = None
            
        return report
