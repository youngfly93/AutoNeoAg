from __future__ import annotations

import numpy as np

from autoneoag.metrics.ranking import metric_bundle, ndcg_at_k


def test_ndcg_at_k_handles_single_document() -> None:
    labels = np.asarray([1.0], dtype=float)
    scores = np.asarray([0.8], dtype=float)
    assert ndcg_at_k(labels, scores, 20) == 0.0


def test_metric_bundle_handles_single_document_split() -> None:
    labels = np.asarray([0.0], dtype=float)
    scores = np.asarray([0.2], dtype=float)
    metrics = metric_bundle(labels, scores)
    assert metrics["val_score"] >= 0.0
    assert metrics["ndcg20"] == 0.0
