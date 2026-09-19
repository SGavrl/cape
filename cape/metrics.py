from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def _arrays(probabilities, labels):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if probabilities.shape != labels.shape or probabilities.ndim != 1:
        raise ValueError("probabilities and labels must be equal one-dimensional arrays")
    if len(labels) == 0:
        raise ValueError("metrics require at least one example")
    if not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("evaluation metrics require binary ground truth")
    return probabilities, labels


def calibration_bins(probabilities, labels, bins=15, *, adaptive=False):
    probabilities, labels = _arrays(probabilities, labels)
    if adaptive:
        order = np.argsort(probabilities)
        groups = np.array_split(order, min(bins, len(order)))
    else:
        edges = np.linspace(0.0, 1.0, bins + 1)
        groups = []
        for index in range(bins):
            upper_inclusive = index == bins - 1
            mask = probabilities >= edges[index]
            mask &= (
                probabilities <= edges[index + 1]
                if upper_inclusive
                else probabilities < edges[index + 1]
            )
            groups.append(np.flatnonzero(mask))

    error = 0.0
    stats = []
    for indexes in groups:
        if len(indexes) == 0:
            stats.append(
                {"count": 0, "mean_probability": None, "positive_rate": None}
            )
            continue
        mean_probability = float(probabilities[indexes].mean())
        positive_rate = float(labels[indexes].mean())
        error += len(indexes) / len(labels) * abs(mean_probability - positive_rate)
        stats.append(
            {
                "count": int(len(indexes)),
                "mean_probability": mean_probability,
                "positive_rate": positive_rate,
                "minimum_probability": float(probabilities[indexes].min()),
                "maximum_probability": float(probabilities[indexes].max()),
            }
        )
    return float(error), stats


def high_confidence_errors(probabilities, labels, thresholds=(0.90, 0.95, 0.99)):
    probabilities, labels = _arrays(probabilities, labels)
    predictions = (probabilities >= 0.5).astype(np.float64)
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    errors = predictions != labels
    result = {}
    for threshold in thresholds:
        selected = confidence >= threshold
        count = int(selected.sum())
        mistakes = int((selected & errors).sum())
        result[f"{threshold:.2f}"] = {
            "predictions": count,
            "errors": mistakes,
            "error_rate": float(mistakes / count) if count else None,
        }
    return result


def selective_accuracy(probabilities, labels, coverages=(0.25, 0.5, 0.75, 0.9, 1.0)):
    probabilities, labels = _arrays(probabilities, labels)
    predictions = (probabilities >= 0.5).astype(np.float64)
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    order = np.argsort(-confidence)
    result = []
    for requested in coverages:
        count = max(1, min(len(labels), int(np.ceil(len(labels) * requested))))
        indexes = order[:count]
        result.append(
            {
                "coverage": float(count / len(labels)),
                "examples": int(count),
                "accuracy": float((predictions[indexes] == labels[indexes]).mean()),
                "minimum_confidence": float(confidence[indexes].min()),
            }
        )
    return result


def metrics_from_arrays(probabilities, labels, bins=15):
    probabilities, labels = _arrays(probabilities, labels)
    predictions = (probabilities >= 0.5).astype(np.float64)
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    ece, probability_bins = calibration_bins(
        probabilities, labels, bins, adaptive=False
    )
    adaptive_ece, adaptive_bins = calibration_bins(
        probabilities, labels, bins, adaptive=True
    )
    return {
        "examples": int(len(labels)),
        "accuracy": float((predictions == labels).mean()),
        "auroc": (
            float(roc_auc_score(labels, probabilities))
            if len(np.unique(labels)) >= 2
            else None
        ),
        "nll": float(
            -np.mean(
                labels * np.log(clipped)
                + (1.0 - labels) * np.log(1.0 - clipped)
            )
        ),
        "brier_score": float(np.mean((probabilities - labels) ** 2)),
        "ece": ece,
        "adaptive_ece": adaptive_ece,
        "probability_bins": probability_bins,
        "adaptive_probability_bins": adaptive_bins,
        "high_confidence_errors": high_confidence_errors(
            probabilities, labels
        ),
        "selective_accuracy": selective_accuracy(probabilities, labels),
    }
