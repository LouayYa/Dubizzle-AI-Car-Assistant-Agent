"""Retrieval: structured filtering + semantic (embedding) search + hybrid ranking.

Embeddings are precomputed offline by scripts/enrich.py and cached to
data/embeddings.npy (+ data/embeddings_ids.json), so app startup makes no API
call. Only the *query* is embedded at request time (one call per semantic
search).
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Optional

import numpy as np

from . import config

_MATRIX: Optional[np.ndarray] = None  # (N, D) L2-normalized rows
_IDS: Optional[list[int]] = None
_ID_TO_ROW: dict[int, int] = {}


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def load_embeddings(force: bool = False) -> tuple[np.ndarray, list[int]]:
    """Load and cache the embedding matrix + aligned listing ids (normalized)."""
    global _MATRIX, _IDS, _ID_TO_ROW
    if _MATRIX is not None and not force:
        return _MATRIX, _IDS

    if not config.EMBEDDINGS_NPY.exists() or not config.EMBEDDINGS_IDS_JSON.exists():
        raise FileNotFoundError(
            "Embeddings cache missing. Run `uv run python scripts/enrich.py` to "
            "generate data/embeddings.npy and data/embeddings_ids.json."
        )
    mat = np.load(config.EMBEDDINGS_NPY).astype(np.float32)
    with open(config.EMBEDDINGS_IDS_JSON, "r", encoding="utf-8") as fh:
        ids = [int(i) for i in json.load(fh)]

    if mat.shape[0] != len(ids):
        raise ValueError(
            f"Embedding matrix rows ({mat.shape[0]}) != id count ({len(ids)})."
        )

    _MATRIX = _l2_normalize(mat)
    _IDS = ids
    _ID_TO_ROW = {lid: i for i, lid in enumerate(ids)}
    return _MATRIX, _IDS


@lru_cache(maxsize=256)
def embed_query(text: str) -> tuple[float, ...]:
    """Embed a query string via LiteLLM. Cached to avoid repeat API calls.

    Lazy import: non-semantic code paths (and tests) never need litellm.
    """
    import litellm

    config.require_api_key()
    resp = litellm.embedding(model=config.EMBED_MODEL, input=[text])
    vec = resp["data"][0]["embedding"]
    return tuple(float(x) for x in vec)


def semantic_rank(
    query: str,
    *,
    candidate_ids: Optional[list[int]] = None,
    limit: int = 10,
    query_vector: Optional[list[float]] = None,
) -> list[tuple[int, float]]:
    """Rank listing ids by cosine similarity to the query.

    If candidate_ids is given, only those are ranked (hybrid mode: structural
    filter first, semantic ranking second). query_vector lets tests inject a
    precomputed vector, skipping the API call.
    """
    mat, ids = load_embeddings()

    if query_vector is not None:
        q = np.asarray(query_vector, dtype=np.float32)
    else:
        q = np.asarray(embed_query(query), dtype=np.float32)
    qn = np.linalg.norm(q)
    if qn == 0:
        return []
    q = q / qn

    if candidate_ids is not None:
        rows = [_ID_TO_ROW[i] for i in candidate_ids if i in _ID_TO_ROW]
        if not rows:
            return []
        sub = mat[rows]
        sims = sub @ q
        order = np.argsort(-sims)[:limit]
        return [(ids[rows[j]], float(sims[j])) for j in order]

    sims = mat @ q  # rows already normalized
    order = np.argsort(-sims)[:limit]
    return [(ids[j], float(sims[j])) for j in order]
