import numpy as np

from cape.metrics import metrics_from_arrays


def test_extended_metrics_report_high_confidence_errors_and_selectivity():
    probabilities = np.array([0.99, 0.96, 0.2, 0.1])
    labels = np.array([0, 1, 0, 0])
    metrics = metrics_from_arrays(probabilities, labels, bins=2)
    assert metrics["high_confidence_errors"]["0.99"]["errors"] == 1
    assert metrics["adaptive_ece"] >= 0.0
    assert metrics["selective_accuracy"][-1]["coverage"] == 1.0
