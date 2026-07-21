"""Semantic ranking + hybrid filtering, using an injected in-memory matrix.

No embeddings file and no API call: we set the module's cached matrix directly
and pass query_vector, so cosine ranking is tested deterministically.
"""
from __future__ import annotations

import numpy as np
import pytest

from app import retrieval


@pytest.fixture()
def fake_embeddings(monkeypatch):
    ids = [1, 2, 3]
    # id1 = "sporty" axis, id2 = "family" axis, id3 = mixed.
    mat = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float32)
    monkeypatch.setattr(retrieval, "_MATRIX", retrieval._l2_normalize(mat))
    monkeypatch.setattr(retrieval, "_IDS", ids)
    monkeypatch.setattr(retrieval, "_ID_TO_ROW", {lid: i for i, lid in enumerate(ids)})
    return ids


def test_semantic_rank_orders_by_similarity(fake_embeddings):
    ranked = retrieval.semantic_rank("sporty", query_vector=[1.0, 0.0], limit=3)
    order = [lid for lid, _ in ranked]
    assert order[0] == 1          # closest to the "sporty" axis
    assert order[-1] == 2         # "family" axis is least similar
    # scores are descending
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_filters_then_ranks(fake_embeddings):
    # Restrict to candidates {2,3} (structural filter) then rank by "sporty".
    ranked = retrieval.semantic_rank("sporty", candidate_ids=[2, 3], query_vector=[1.0, 0.0])
    order = [lid for lid, _ in ranked]
    assert order == [3, 2]        # id1 excluded; id3 more "sporty" than id2


def test_hybrid_empty_candidates(fake_embeddings):
    assert retrieval.semantic_rank("x", candidate_ids=[], query_vector=[1.0, 0.0]) == []
