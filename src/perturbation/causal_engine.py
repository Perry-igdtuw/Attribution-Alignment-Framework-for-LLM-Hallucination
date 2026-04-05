"""
causal_engine.py
----------------
Orchestrates the entire causal perturbation process for a single sample.

Given a prompt, an explanation, and XAI attribution results, the causal
engine seamlessly handles masking across all strategies, generating
new responses, comparing outputs, and producing the final OutputShift bundle.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from src.model.llm_client import LLMClient
from src.perturbation.token_masker import TokenMasker, MaskStrategy
from src.perturbation.output_comparator import OutputComparator, OutputShift
from src.xai.attribution_utils import AttributionResult

logger = logging.getLogger(__name__)


class CausalEngine:
    def __init__(
        self,
        llm_client: LLMClient,
        masker: TokenMasker,
        comparator: OutputComparator,
        strategies: List[str] = ["mask", "delete", "replace"],
    ):
        self.llm = llm_client
        self.masker = masker
        self.comparator = comparator
        self.strategies = [MaskStrategy(s) for s in strategies]

    def measure_causal_impact(
        self,
        original_question: str,
        original_response_text: str,
        original_explanation: str,
        original_log_probs: Optional[List[float]],
        attribution_result: AttributionResult,
        top_k: Optional[int] = None,
    ) -> List[OutputShift]:
        """
        Runs the full causal perturbation suite.
        Returns a list of OutputShift objects (one per strategy).
        """
        shifts = []

        for strategy in self.strategies:
            # 1. Mask the prompt
            masked = self.masker.mask_explanation_in_prompt(
                prompt=original_question,
                explanation=original_explanation,
                attribution_result=attribution_result,
                strategy=strategy,
                use_top_k=top_k,
            )

            # If no tokens were masked (e.g., all were stopwords), 0 shift
            if masked.n_tokens_affected == 0:
                shifts.append(
                    OutputShift(
                        original_output=original_response_text,
                        perturbed_output=original_response_text,
                        strategy=strategy.value,
                        js_divergence=0.0,
                        semantic_distance=0.0,
                        rouge_l_delta=0.0,
                        combined_shift=0.0,
                    )
                )
                continue

            # 2. Re-run inference with the perturbed prompt
            perturbed_resp = self.llm.generate(
                question=masked.masked_text,
                strategy="direct",  # Use direct mode to evaluate answer shift
                compute_gradients=False,
            )

            # 3. Compare original output vs perturbed output
            shift = self.comparator.compare(
                original_output=original_response_text,
                perturbed_output=perturbed_resp.answer,
                strategy=strategy.value,
                original_log_probs=original_log_probs,
                perturbed_log_probs=perturbed_resp.token_log_probs or None,
            )
            shifts.append(shift)

        return shifts
