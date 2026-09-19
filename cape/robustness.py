from __future__ import annotations

from collections import defaultdict
import statistics


def robustness_metrics(rows, probabilities):
    if len(rows) != len(probabilities):
        raise ValueError("row and probability counts differ")
    groups = defaultdict(list)
    for row, probability in zip(rows, probabilities, strict=True):
        group = row.get("robustness_group")
        relation = row.get("robustness_relation")
        if group and relation:
            groups[(relation, group)].append(float(probability))

    invariance_deltas = []
    complement_deviations = []
    for (relation, _), values in groups.items():
        if relation == "equivalent" and len(values) >= 2:
            invariance_deltas.append(max(values) - min(values))
        elif relation == "complement" and len(values) == 2:
            complement_deviations.append(abs(sum(values) - 1.0))

    return {
        "equivalent_groups": len(invariance_deltas),
        "mean_paraphrase_delta": (
            statistics.mean(invariance_deltas) if invariance_deltas else None
        ),
        "max_paraphrase_delta": max(invariance_deltas) if invariance_deltas else None,
        "complement_pairs": len(complement_deviations),
        "mean_complement_sum_deviation": (
            statistics.mean(complement_deviations)
            if complement_deviations
            else None
        ),
    }
