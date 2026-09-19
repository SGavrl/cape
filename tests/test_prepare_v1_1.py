import json
import os

from scripts.prepare_v1_1 import (
    convert_snli_row,
    extra_train_paths,
    generate_synthetic_examples,
    load_supplemental_examples,
    pair_squad_rows,
)


def test_snli_label_mapping():
    expected = {0: 1.0, 1: 0.5, 2: 0.0}
    for label, target in expected.items():
        converted = convert_snli_row(
            {"premise": "p", "hypothesis": "h", "label": label}
        )
        assert converted["label"] == target


def test_squad_negative_pairing_never_self_pairs():
    rows = [
        {
            "id": str(index),
            "title": f"topic-{index}",
            "question": f"question {index}",
            "context": f"context {index}",
        }
        for index in range(8)
    ]
    pairs = pair_squad_rows(rows, seed=3)
    assert all(positive["id"] != negative["id"] for positive, negative in pairs)
    assert all(
        positive["context"] != negative["context"]
        for positive, negative in pairs
    )


def test_synthetic_reasoning_is_deterministic_and_bounded():
    first = generate_synthetic_examples("train", 20, seed=17)
    second = generate_synthetic_examples("train", 20, seed=17)
    assert first == second
    assert all(0.0 <= row["label"] <= 1.0 for row in first)


def test_synthetic_splits_do_not_overlap():
    train = generate_synthetic_examples("train", 20)
    validation = generate_synthetic_examples("validation", 20)
    train_pairs = {(row["context"], row["assertion"]) for row in train}
    validation_pairs = {
        (row["context"], row["assertion"]) for row in validation
    }
    assert train_pairs.isdisjoint(validation_pairs)


def test_supplemental_jsonl_uses_public_example_schema(tmp_path):
    path = tmp_path / "extra.jsonl"
    rows = [
        {
            "context": "A service returned status 503.",
            "assertion": "The service was unavailable.",
            "label": 0.9,
            "source": "synthetic_operations",
        },
        {
            "context": "The queue is empty.",
            "assertion": "An item can be removed immediately.",
            "label": 0.0,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    loaded = load_supplemental_examples([path])

    assert loaded[0] == rows[0]
    assert loaded[1]["source"] == "supplemental"
    assert all(isinstance(row["label"], float) for row in loaded)


def test_extra_train_paths_combine_environment_and_cli():
    separator = os.pathsep
    paths = extra_train_paths(
        ["cli.jsonl", "shared.jsonl"],
        {"CAPE_EXTRA_TRAIN_DATA": f"env.jsonl{separator}shared.jsonl"},
    )

    assert paths == ["env.jsonl", "shared.jsonl", "cli.jsonl"]
