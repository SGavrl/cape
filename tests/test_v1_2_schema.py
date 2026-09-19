import json

import pytest

from cape.dataset import CAPEDataset, _labels
from cape.schema import MISSING_LABEL, auxiliary_label, normalize_example
from scripts.prepare_v1_2 import upgrade_v1_1_row


def test_schema_accepts_target_and_legacy_label(tmp_path):
    path = tmp_path / "examples.jsonl"
    path.write_text(
        json.dumps(
            {
                "context": "Evidence was recorded.",
                "assertion": "Evidence exists.",
                "target": 1.0,
                "nli_label": "entailment",
            }
        )
        + "\n"
        + json.dumps(
            {
                "context": "No result was recorded.",
                "assertion": "A result was recorded.",
                "label": 0,
            }
        )
        + "\n"
    )
    dataset = CAPEDataset(path)
    assert [row["target"] for row in dataset.examples] == [1.0, 0.0]
    assert dataset.examples[0]["nli_label"] == "entailment"


def test_collator_labels_accept_legacy_and_v1_2_rows():
    labels = _labels(
        [{"label": 1}, {"target": 0.0}],
        include_auxiliary=False,
    )

    assert labels["labels"].tolist() == [1.0, 0.0]


def test_missing_auxiliary_label_is_masked():
    row = normalize_example(
        {"context": "c", "assertion": "a", "target": 0.0}
    )
    assert auxiliary_label(row, "nli_label") == MISSING_LABEL


def test_unknown_auxiliary_label_is_rejected():
    with pytest.raises(ValueError, match="invalid nli_label"):
        normalize_example(
            {
                "context": "c",
                "assertion": "a",
                "target": 0.0,
                "nli_label": "maybe",
            }
        )


def test_neutral_v1_1_examples_become_unsupported_with_nli_label():
    upgraded = upgrade_v1_1_row(
        {
            "context": "A premise.",
            "assertion": "An unsupported hypothesis.",
            "label": 0.5,
            "source": "snli",
        },
        4,
    )
    assert upgraded["target"] == 0.0
    assert upgraded["nli_label"] == "neutral"
