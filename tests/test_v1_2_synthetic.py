import hashlib
import json
from pathlib import Path

from cape.synthetic import generate_v1_2_synthetic
from scripts.prepare_v1_2 import (
    assert_no_challenge_overlap,
    build_hard_relevance,
    split_validation_for_calibration,
)


def test_v1_2_synthetic_is_deterministic_and_unique():
    first = generate_v1_2_synthetic("train", 24, seed=9)
    second = generate_v1_2_synthetic("train", 24, seed=9)
    assert first == second
    assert len({row["id"] for row in first}) == len(first)
    pairs = {(row["context"], row["assertion"]) for row in first}
    assert len(pairs) == len(first)


def test_template_ood_splits_do_not_overlap_or_touch_challenge():
    train = generate_v1_2_synthetic("train", 30)
    test = generate_v1_2_synthetic("test", 30)
    train_pairs = {(row["context"], row["assertion"]) for row in train}
    test_pairs = {(row["context"], row["assertion"]) for row in test}
    assert train_pairs.isdisjoint(test_pairs)
    assert_no_challenge_overlap(train + test)


def test_frozen_holdout_matches_manifest_and_is_disjoint():
    root = Path(__file__).parents[1]
    holdout_path = root / "benchmarks/v1_2_holdout.jsonl"
    manifest = json.loads(
        (root / "benchmarks/v1_2_holdout_manifest.json").read_text()
    )
    payload = holdout_path.read_bytes()
    rows = [json.loads(line) for line in payload.splitlines()]
    assert hashlib.sha256(payload).hexdigest() == manifest["sha256"]
    assert len(rows) == manifest["examples"]
    assert len({row["id"] for row in rows}) == len(rows)
    assert_no_challenge_overlap(rows)


def test_hard_relevance_uses_distinct_same_topic_passage():
    rows = [
        {
            "id": "a",
            "title": "ocean",
            "question": "Which layer receives sunlight?",
            "context": "The upper ocean receives sunlight.",
            "answers": {"text": ["upper ocean"]},
        },
        {
            "id": "b",
            "title": "ocean",
            "question": "Where are trenches found?",
            "context": "Deep ocean trenches form near plate boundaries.",
            "answers": {"text": ["plate boundaries"]},
        },
        {
            "id": "c",
            "title": "music",
            "question": "What creates rhythm?",
            "context": "Repeated beats create rhythm.",
            "answers": {"text": ["repeated beats"]},
        },
    ]
    examples = build_hard_relevance(rows, "train", limit=1, seed=2)
    assert len(examples) == 2
    assert {row["target"] for row in examples} == {0.0, 1.0}
    assert examples[0]["context"] != examples[1]["context"]
    assert examples[1]["negative_kind"] in {"same_topic", "lexical"}


def test_calibration_partitions_are_disjoint():
    rows = [
        {
            "id": f"row-{index}",
            "capability": "test",
            "target": float(index % 2),
        }
        for index in range(40)
    ]
    parts = split_validation_for_calibration(rows, seed=7)
    id_sets = [{row["id"] for row in part} for part in parts]
    assert sum(map(len, id_sets)) == len(rows)
    assert id_sets[0].isdisjoint(id_sets[1])
    assert id_sets[0].isdisjoint(id_sets[2])
    assert id_sets[1].isdisjoint(id_sets[2])
