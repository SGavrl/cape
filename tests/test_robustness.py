from cape.robustness import robustness_metrics
from scripts.prepare_robustness import robustness_rows


def test_robustness_metrics_measure_invariance_and_complements():
    rows = robustness_rows()
    probabilities = [0.9 if row["target"] else 0.1 for row in rows]
    metrics = robustness_metrics(rows, probabilities)
    assert metrics["equivalent_groups"] == 3
    assert metrics["max_paraphrase_delta"] == 0.0
    assert metrics["complement_pairs"] == 2
    assert metrics["mean_complement_sum_deviation"] == 0.0
