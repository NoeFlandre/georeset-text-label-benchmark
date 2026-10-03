"""Model-card-compatible E5 encoding and deterministic candidate ranking."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _validate_text_values(texts: Sequence[str]) -> None:
    if not texts or any(not isinstance(text, str) or not text for text in texts):
        raise ValueError("texts must be non-empty strings")


def _validate_prefix(prefix: str) -> None:
    if prefix not in {"query", "passage"}:
        raise ValueError("prefix must be query or passage")


def _validate_dimensions(batch_size: int, max_length: int) -> None:
    if batch_size < 1 or max_length < 1:
        raise ValueError("batch size and max length must be positive")


def _validate_text_input(
    texts: Sequence[str], prefix: str, batch_size: int, max_length: int
) -> None:
    _validate_text_values(texts)
    _validate_prefix(prefix)
    _validate_dimensions(batch_size, max_length)


def average_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
    """Return an attention-mask-aware mean over each token sequence."""
    if last_hidden_states.ndim != 3 or tuple(last_hidden_states.shape[:2]) != tuple(
        attention_mask.shape
    ):
        raise ValueError("hidden states and attention mask shapes must agree")
    mask = attention_mask.unsqueeze(-1).bool()
    masked_states = last_hidden_states.masked_fill(~mask, 0.0)
    denominator = attention_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
    return masked_states.sum(dim=1) / denominator


def encode_texts(
    texts: Sequence[str],
    tokenizer: Any,
    model: Any,
    *,
    prefix: str,
    batch_size: int = 16,
    max_length: int = 512,
) -> Any:
    """Encode text batches with E5 prefixes, masked mean pooling, and L2 normalization."""
    import torch
    import torch.nn.functional as functional

    _validate_text_input(texts, prefix, batch_size, max_length)
    model.eval()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = [f"{prefix}: {text}" for text in texts[start : start + batch_size]]
            tokens = tokenizer(
                batch_texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            outputs = model(**tokens)
            pooled = average_pool(outputs.last_hidden_state, tokens["attention_mask"])
            vectors.append(functional.normalize(pooled, p=2, dim=1))
    return torch.cat(vectors, dim=0)


def _validate_embedding_shapes(
    query_embeddings: Any,
    candidate_embeddings: Any,
    candidate_codes: Sequence[str],
) -> None:
    if query_embeddings.ndim != 2 or candidate_embeddings.ndim != 2:
        raise ValueError("query and candidate embeddings must be two-dimensional")
    if query_embeddings.shape[1] != candidate_embeddings.shape[1]:
        raise ValueError("query and candidate embedding dimensions must agree")
    if candidate_embeddings.shape[0] != len(candidate_codes):
        raise ValueError("candidate codes and candidate embeddings must align")


def _validate_codes(candidate_codes: Sequence[str], top_k: int) -> None:
    if not candidate_codes or len(set(candidate_codes)) != len(candidate_codes):
        raise ValueError("candidate codes must be non-empty and unique")
    if top_k < 1 or top_k > len(candidate_codes):
        raise ValueError("top_k must fit the candidate set")


def rank_candidates(
    query_embeddings: Any,
    candidate_embeddings: Any,
    candidate_codes: Sequence[str],
    *,
    top_k: int = 5,
) -> list[list[tuple[str, float]]]:
    """Rank normalized candidate vectors by cosine similarity with stable ties."""
    _validate_embedding_shapes(query_embeddings, candidate_embeddings, candidate_codes)
    _validate_codes(candidate_codes, top_k)
    scores = (query_embeddings @ candidate_embeddings.T).tolist()
    rankings = []
    for row in scores:
        ordered = sorted(
            zip(candidate_codes, row, strict=True), key=lambda item: (-item[1], item[0])
        )
        rankings.append([(code, float(score)) for code, score in ordered[:top_k]])
    return rankings
