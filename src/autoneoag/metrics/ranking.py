from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, ndcg_score


def ppv_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    if len(labels) == 0:
        return 0.0
    order = np.argsort(scores)[::-1][: min(k, len(labels))]
    return float(np.mean(labels[order]))


def ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    if len(labels) == 0 or np.all(labels == 0):
        return 0.0
    return float(ndcg_score(labels.reshape(1, -1), scores.reshape(1, -1), k=min(k, len(labels))))


def metric_bundle(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = labels.astype(float)
    scores = scores.astype(float)
    auprc = float(average_precision_score(labels, scores)) if len(np.unique(labels)) > 1 else 0.0
    ppv10 = ppv_at_k(labels, scores, 10)
    ppv20 = ppv_at_k(labels, scores, 20)
    ndcg20 = ndcg_at_k(labels, scores, 20)
    val_score = 0.45 * auprc + 0.35 * ppv20 + 0.10 * ppv10 + 0.10 * ndcg20
    return {
        "val_score": val_score,
        "auprc": auprc,
        "ppv10": ppv10,
        "ppv20": ppv20,
        "ndcg20": ndcg20,
    }


def summarize_metrics(metrics: dict[str, float]) -> str:
    return "\n".join(f"{key}: {value:.6f}" for key, value in metrics.items())

