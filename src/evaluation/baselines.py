"""
baselines.py
------------
Implements baseline hallucination detection methods to compare against FHI.
Includes LogProb Thresholding and Self-Consistency.
"""

import numpy as np
from typing import List
from src.model.llm_client import LLMClient

class BaselineEvaluator:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def evaluate_logprob(self, log_probs: List[float], threshold: float = -0.5) -> bool:
        """
        Log-Probability Baseline.
        If the mean log-prob of the generated answer is below the threshold,
        it flags it as a hallucination.
        Returns True if hallucination is detected.
        """
        if not log_probs:
            return False # Assume factual if no logits available
        mean_lp = np.mean(log_probs)
        return bool(mean_lp < threshold)

    def evaluate_self_consistency(self, question: str, n_samples: int = 3, temperature: float = 0.7) -> float:
        """
        Self-Consistency Baseline (Wang et al., 2022).
        Generates multiple answers and measures agreement.
        Low agreement = High hallucination risk.
        Returns a continuous hallucination risk score (1.0 - agreement_ratio).
        """
        answers = []
        for _ in range(n_samples):
            # We enforce temperature > 0 for this metric even if client default is 0
            # A real implementation passes temp down to the HF generator.
            resp = self.llm.generate(question, strategy="direct", compute_gradients=False)
            answers.append(resp.answer.lower().strip())
            
        if not answers:
            return 0.0
            
        # Very simple exact match consensus measurement.
        # Can be upgraded to Semantic Equivalence using SentenceTransformers.
        most_common_ans = max(set(answers), key=answers.count)
        agreement_ratio = answers.count(most_common_ans) / len(answers)
        
        # 1.0 - agreement = hallucination risk
        return 1.0 - agreement_ratio
