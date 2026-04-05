"""
attribution_utils.py
--------------------
Shared utilities for all attribution methods.

Responsibilities:
1. Normalize attribution scores to [0, 1]
2. Align model tokens with explanation tokens (handles subword tokenization)
3. Extract top-K important tokens for AAS computation
4. Clean token strings (remove special chars like u2581, ##, G)
"""

from __future__ import annotations

import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Set, Tuple


SPECIAL_TOKENS: Set[str] = {
    "<s>", "</s>", "<pad>", "<unk>", "[CLS]", "[SEP]", "[PAD]",
    "<bos>", "<eos>", "<mask>", "bos_token", "eos_token",
}


@dataclass
class AttributionResult:
    """
    Standardized output of any attribution method.
    Used as input to AAS (Attribution Alignment Score) computation.
    """
    tokens: List[str]                   # All input token strings (cleaned)
    scores: np.ndarray                  # Shape: (n_tokens,), normalized [0, 1]
    method: str                         # "attention" | "gradient" | "shap"
    top_k_tokens: List[str] = field(default_factory=list)
    top_k_indices: List[int] = field(default_factory=list)


def normalize_scores(scores: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Min-max normalize scores to [0, 1]. Handles constant arrays."""
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < eps:
        return np.ones_like(scores) * 0.5
    return (scores - s_min) / (s_max - s_min + eps)


def clean_token(token: str) -> str:
    """
    Remove tokenizer-specific prefix artifacts and lowercase.
    Handles SentencePiece, BPE (GPT), and WordPiece (BERT) conventions.
    """
    token = re.sub(r"^[#\u2581G\u0120]+", "", token)
    return token.strip().lower()


def clean_tokens(tokens: List[str]) -> List[str]:
    return [clean_token(t) for t in tokens]


def get_top_k_tokens(
    tokens: List[str],
    scores: np.ndarray,
    k: int = 10,
    exclude_special: bool = True,
) -> Tuple[List[str], List[int]]:
    """
    Return the top-K tokens by attribution score.

    Args:
        tokens:          All token strings.
        scores:          Attribution scores, same length as tokens.
        k:               Number of top tokens to return.
        exclude_special: Skip special/padding tokens.

    Returns:
        (top_k_token_strings, top_k_indices)
    """
    cleaned = clean_tokens(tokens)
    sorted_indices = np.argsort(scores)[::-1]

    top_k_tokens = []
    top_k_indices = []
    for idx in sorted_indices:
        if exclude_special and cleaned[idx] in SPECIAL_TOKENS:
            continue
        if cleaned[idx] == "":
            continue
        top_k_tokens.append(cleaned[idx])
        top_k_indices.append(int(idx))
        if len(top_k_tokens) >= k:
            break

    return top_k_tokens, top_k_indices


def make_attribution_result(
    tokens: List[str],
    raw_scores: np.ndarray,
    method: str,
    top_k: int = 10,
) -> AttributionResult:
    """
    Factory: normalize scores, extract top-K, and return AttributionResult.
    All attribution modules should call this at the end.
    """
    norm_scores = normalize_scores(raw_scores)
    top_k_tokens, top_k_indices = get_top_k_tokens(tokens, norm_scores, k=top_k)
    return AttributionResult(
        tokens=clean_tokens(tokens),
        scores=norm_scores,
        method=method,
        top_k_tokens=top_k_tokens,
        top_k_indices=top_k_indices,
    )


def tokenize_text_words(text: str) -> Set[str]:
    """
    Tokenize free-form text into a set of lowercase words.
    Used to align explanation sentences with token vocabulary.
    Removes stopwords for more meaningful overlap.
    """
    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "of", "in", "on", "at", "to", "for", "with", "by", "from",
        "it", "its", "this", "that", "and", "or", "but", "not", "no",
    }
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    return {w for w in words if w not in STOPWORDS}
