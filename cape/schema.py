from __future__ import annotations

import math


NLI_LABELS = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
}
RELEVANCE_LABELS = {
    "irrelevant": 0,
    "relevant": 1,
}
STRENGTH_LABELS = {
    "possible": 0,
    "likely": 1,
    "certain": 2,
}
AUXILIARY_LABELS = {
    "nli_label": NLI_LABELS,
    "relevance_label": RELEVANCE_LABELS,
    "strength_label": STRENGTH_LABELS,
}
MISSING_LABEL = -100


def support_target(row: dict) -> float:
    value = row.get("target", row.get("label"))
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError("expected finite target in [0, 1]")
    return float(value)


def auxiliary_label(row: dict, field: str) -> int:
    mapping = AUXILIARY_LABELS[field]
    value = row.get(field)
    if value is None:
        return MISSING_LABEL
    if isinstance(value, str):
        if value not in mapping:
            raise ValueError(
                f"invalid {field} {value!r}; expected one of {sorted(mapping)}"
            )
        return mapping[value]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"invalid {field} {value!r}")
    if value not in mapping.values():
        raise ValueError(f"invalid {field} {value!r}")
    return value


def normalize_example(row: dict) -> dict:
    context = row.get("context")
    assertion = row.get("assertion")
    if not isinstance(context, str) or not context.strip():
        raise ValueError("context must be a non-empty string")
    if not isinstance(assertion, str) or not assertion.strip():
        raise ValueError("assertion must be a non-empty string")

    normalized = dict(row)
    normalized["context"] = context.strip()
    normalized["assertion"] = assertion.strip()
    normalized["target"] = support_target(row)
    normalized["label"] = normalized["target"]
    normalized["source"] = row.get("source") or "unknown"
    normalized["capability"] = row.get("capability") or normalized["source"]
    for field in AUXILIARY_LABELS:
        auxiliary_label(normalized, field)
    return normalized
