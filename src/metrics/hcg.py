"""
hcg.py — Hallucination Confidence Gap
---------------------------------------
Measures the gap between the model's confidence and actual factual correctness.

Formula:
    HCG = max(0, confidence - correctness_score)

where:
    confidence       = model's answer token confidence (exp(mean log-prob)) ∈ [0,1]
    correctness_score = factual accuracy metric ∈ [0,1] (e.g. token-level F1 vs ground truth)

Interpretation:
    HCG → 1.0: Model is highly confident but factually wrong (classic hallucination)
    HCG → 0.0: Model's confidence matches its accuracy (calibrated)
    HCG < 0:   Clamped to 0 (model more accurate than confident — not a problem)

Why this matters:
    - Calibration research (Guo et al., 2017) shows LLMs are overconfident.
    - When a model hallucinates, it often generates plausible-sounding text
      at high token probability — high confidence despite being wrong.
    - HCG penalizes exactly this failure mode.
    - Note: HCG is SUBTRACTED in the FHI formula (high HCG → lower FHI → hallucination)

Correctness measures implemented:
    1. Token F1 (default): Overlap between predicted answer and gold answer tokens
       → Standard SQuAD evaluation metric, widely used in QA papers
    2. Exact match: Binary (1 if exactly correct, 0 otherwise)
    3. Semantic similarity: SentenceTransformer-based (for non-extractive answers)

References:
    - Guo et al. (2017) "On Calibration of Modern Neural Networks"
    - Rajpurkar et al. (2016) "SQuAD: 100,000+ Questions" (token-F1 definition)
    - Manakul et al. (2023) "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection"
"""

from __future__ import annotations

import re
import numpy as np
from typing import Optional


def compute_hcg(
    model_confidence: float,
    predicted_answer: str,
    gold_answer: str,
    correctness_method: str = "token_f1",
    semantic_model=None,
) -> float:
    """
    Compute Hallucination Confidence Gap (HCG).

    Args:
        model_confidence:   exp(mean log-prob) of answer tokens ∈ [0, 1].
                            From LLMResponse.answer_confidence.
        predicted_answer:   The model's generated answer string.
        gold_answer:        The ground-truth reference answer.
        correctness_method: "token_f1" | "exact_match" | "semantic"
        semantic_model:     Pre-loaded SentenceTransformer (for "semantic" method).

    Returns:
        HCG ∈ [0, 1]. Higher = more hallucinated (high confidence, wrong answer).
    """
    correctness = _compute_correctness(
        predicted_answer, gold_answer, correctness_method, semantic_model
    )
    hcg = max(0.0, float(model_confidence) - correctness)
    return float(np.clip(hcg, 0.0, 1.0))


def compute_hcg_detailed(
    model_confidence: float,
    predicted_answer: str,
    gold_answer: str,
    semantic_model=None,
) -> dict:
    """
    Compute HCG with all correctness methods for detailed reporting.
    Used in the error analysis section of the paper.
    """
    token_f1 = _compute_correctness(predicted_answer, gold_answer, "token_f1")
    exact = _compute_correctness(predicted_answer, gold_answer, "exact_match")
    semantic = _compute_correctness(predicted_answer, gold_answer, "semantic", semantic_model)

    return {
        "model_confidence": model_confidence,
        "token_f1_correctness": token_f1,
        "exact_match_correctness": exact,
        "semantic_correctness": semantic,
        "hcg_token_f1": max(0.0, model_confidence - token_f1),
        "hcg_exact": max(0.0, model_confidence - exact),
        "hcg_semantic": max(0.0, model_confidence - semantic),
    }


# ── Correctness Measures ──────────────────────────────────────────────────────

def _compute_correctness(
    predicted: str,
    gold: str,
    method: str = "token_f1",
    semantic_model=None,
) -> float:
    """Internal dispatcher for correctness computation."""
    if method == "token_f1":
        return _token_f1(predicted, gold)
    elif method == "exact_match":
        return _exact_match(predicted, gold)
    elif method == "semantic":
        return _semantic_similarity(predicted, gold, semantic_model)
    else:
        raise ValueError(f"Unknown correctness method: {method}")


def _normalize_text(text: str) -> str:
    """Standard SQuAD normalization: lowercase, strip punctuation and articles."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _token_f1(predicted: str, gold: str) -> float:
    """
    Token-level F1 score (SQuAD evaluation metric).
    Standard for extractive QA evaluation.
    """
    pred_tokens = set(_normalize_text(predicted).split())
    gold_tokens = set(_normalize_text(gold).split())

    if not pred_tokens or not gold_tokens:
        return 0.0

    tp = len(pred_tokens & gold_tokens)
    if tp == 0:
        return 0.0

    precision = tp / len(pred_tokens)
    recall = tp / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return float(f1)


def _exact_match(predicted: str, gold: str) -> float:
    """Binary exact match after normalization. Returns 1.0 or 0.0."""
    return 1.0 if _normalize_text(predicted) == _normalize_text(gold) else 0.0


def _semantic_similarity(
    predicted: str, gold: str, semantic_model=None
) -> float:
    """
    Semantic similarity via SentenceTransformer cosine similarity.
    More lenient than token-F1; handles paraphrased correct answers.
    """
    if semantic_model is None:
        # Lazy import
        from sentence_transformers import SentenceTransformer
        semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = semantic_model.encode(
        [predicted, gold], normalize_embeddings=True
    )
    cosine_sim = float(np.dot(embeddings[0], embeddings[1]))
    return float(np.clip(cosine_sim, 0.0, 1.0))
