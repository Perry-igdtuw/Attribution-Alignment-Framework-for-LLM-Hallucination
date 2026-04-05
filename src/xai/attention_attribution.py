"""
attention_attribution.py
-------------------------
Attention-based attribution using attention rollout.

Why attention rollout (not raw attention)?
- Raw attention averaging across heads is unreliable as an explanation method:
  it ignores that transformers have residual connections, meaning each layer
  also passes information directly without going through attention.
- Attention rollout (Abnar & Zuidema, 2020) correctly models this by composing
  attention matrices across layers with identity residuals added.

Limitation flagged for paper:
- Attention-based attribution has been criticized (Jain & Wallace, 2019;
  Wiegreffe & Pinter, 2019) as not faithfully reflecting model decisions.
- We use it as ONE of three attribution methods and combine them — this
  addresses the limitation and strengthens our multi-method claim.

References:
  - Abnar & Zuidema (2020) "Quantifying Attention Flow in Transformers"
  - Jain & Wallace (2019) "Attention is not Explanation"
  - Wiegreffe & Pinter (2019) "Attention is not not Explanation"
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from src.model.llm_client import LLMResponse
from src.xai.attribution_utils import AttributionResult, make_attribution_result


class AttentionAttribution:
    """
    Computes attention-based token attribution from a precomputed LLMResponse.

    Input:  LLMResponse with attention_rollout populated (HF backend required).
    Output: AttributionResult with normalized scores over input tokens.
    """

    def __init__(self, top_k: int = 10):
        self.top_k = top_k

    def compute(self, response: LLMResponse) -> Optional[AttributionResult]:
        """
        Compute attention-based attribution from rollout scores.

        Args:
            response: LLMResponse with attention_rollout field populated.
                      (Requires HuggingFace backend with output_attentions=True)

        Returns:
            AttributionResult, or None if attention is not available.
        """
        if response.attention_rollout is None:
            return None

        if len(response.input_tokens) == 0:
            return None

        scores = response.attention_rollout  # Already normalized from TokenExtractor

        return make_attribution_result(
            tokens=response.input_tokens,
            raw_scores=scores,
            method="attention",
            top_k=self.top_k,
        )

    def compute_layer_wise(
        self,
        response: LLMResponse,
        layer_idx: int = -1,
    ) -> Optional[AttributionResult]:
        """
        Compute attribution from a specific layer's attention (not rolled out).
        Useful for layer-wise analysis in ablation studies.

        Args:
            layer_idx: Index of transformer layer (-1 = last layer).
        """
        if response.raw_attention is None:
            return None

        n_input = len(response.input_tokens)
        layer_attn = response.raw_attention[layer_idx]  # (n_heads, seq, seq)

        # Average over heads, take last token's attention over input tokens
        mean_attn = layer_attn.mean(axis=0)             # (seq, seq)
        scores = mean_attn[-1, :n_input]                # Last token → input tokens

        return make_attribution_result(
            tokens=response.input_tokens,
            raw_scores=scores,
            method=f"attention_layer_{layer_idx}",
            top_k=self.top_k,
        )
