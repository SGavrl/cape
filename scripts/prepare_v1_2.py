import argparse
import json
import os
import random
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from cape.schema import normalize_example
from cape.synthetic import SPLIT_SEEDS, generate_v1_2_synthetic
try:
    from scripts.challenge import CASES as FROZEN_CHALLENGE_CASES
except ModuleNotFoundError:
    from challenge import CASES as FROZEN_CHALLENGE_CASES


HANS_URLS = {
    "train": (
        "https://raw.githubusercontent.com/tommccoy1/hans/master/"
        "heuristics_train_set.jsonl"
    ),
    "test": (
        "https://raw.githubusercontent.com/tommccoy1/hans/master/"
        "heuristics_evaluation_set.jsonl"
    ),
}
RELEVANCE_ASSERTIONS = {
    "train": (
        "The passage provides evidence useful for answering the question.",
        "The information in the passage supports an answer to the question.",
    ),
    "validation": (
        "This material contributes evidence toward answering the question.",
    ),
    "test": (
        "The question can be answered in part using this passage.",
    ),
}
DEFAULT_SYNTHETIC_COUNTS = {
    "train": 1500,
    "validation": 200,
    "test": 400,
}
DEFAULT_RELEVANCE_LIMITS = {
    "train": 12000,
    "validation": 1500,
    "test": 3000,
}


def read_jsonl(path):
    rows = []
    with Path(path).open() as file:
        for line_number, line in enumerate(file, 1):
            try:
                rows.append(normalize_example(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(f"invalid row at {path}:{line_number}: {error}") from error
    return rows


def write_jsonl(path, rows, seed):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    with Path(path).open("w") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def upgrade_v1_1_row(row, index, split="unknown"):
    row = normalize_example(row)
    source = row["source"]
    target = row["target"]
    upgraded = {
        **row,
        "id": row.get("id", f"v1.1-{split}-{source}-{index:08d}"),
        "target": target,
        "capability": row.get("capability", source),
    }
    if source in {"snli", "mnli"}:
        if target == 1.0:
            upgraded["nli_label"] = "entailment"
        elif source == "snli" and target == 0.5:
            upgraded["target"] = 0.0
            upgraded["label"] = 0.0
            upgraded["nli_label"] = "neutral"
        else:
            upgraded["nli_label"] = "contradiction"
    if source == "squad_relevance":
        upgraded["relevance_label"] = (
            "relevant" if target == 1.0 else "irrelevant"
        )
    return upgraded


def upgrade_v1_1(rows, split):
    return [
        upgrade_v1_1_row(row, index, split)
        for index, row in enumerate(rows)
    ]


def _tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _overlap(left, right):
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def _answer_strings(row):
    answers = row.get("answers", {})
    return [str(text).lower() for text in answers.get("text", []) if text]


def _valid_negative(positive, candidate):
    if candidate["context"] == positive["context"]:
        return False
    lowered = candidate["context"].lower()
    return not any(answer in lowered for answer in _answer_strings(positive))


def build_hard_relevance(rows, split, limit, seed):
    rows = list(rows)
    rng = random.Random(seed)
    sample = rng.sample(rows, min(len(rows), max(limit * 4, limit)))
    sample.sort(key=lambda row: _overlap(row["question"], row["context"]))
    selected = sample[:limit]

    by_title = defaultdict(list)
    for row in rows:
        by_title[row.get("title", "")].append(row)

    output = []
    assertions = RELEVANCE_ASSERTIONS[split]
    for index, row in enumerate(selected):
        same_topic = [
            candidate
            for candidate in by_title[row.get("title", "")]
            if _valid_negative(row, candidate)
        ]
        if same_topic:
            negative = max(
                same_topic[:64],
                key=lambda candidate: _overlap(
                    row["question"], candidate["context"]
                ),
            )
        else:
            pool = [
                candidate
                for candidate in rng.sample(rows, min(128, len(rows)))
                if _valid_negative(row, candidate)
            ]
            if not pool:
                continue
            negative = max(
                pool,
                key=lambda candidate: _overlap(
                    row["question"], candidate["context"]
                ),
            )

        assertion = assertions[index % len(assertions)]
        base_id = str(row.get("id", index))
        common = {
            "source": "squad_hard_relevance",
            "capability": "indirect_relevance",
            "relevance_label": "relevant",
            "split_protocol": "hard_relevance",
        }
        output.append(
            {
                **common,
                "id": f"squad-{split}-{base_id}-positive",
                "context": (
                    f"Question: {row['question'].strip()}\n"
                    f"Passage: {row['context'].strip()}"
                ),
                "assertion": assertion,
                "target": 1.0,
            }
        )
        output.append(
            {
                **common,
                "id": f"squad-{split}-{base_id}-hard-negative",
                "context": (
                    f"Question: {row['question'].strip()}\n"
                    f"Passage: {negative['context'].strip()}"
                ),
                "assertion": assertion,
                "target": 0.0,
                "relevance_label": "irrelevant",
                "negative_kind": (
                    "same_topic"
                    if negative.get("title") == row.get("title")
                    else "lexical"
                ),
            }
        )
    return output


def download_jsonl(url, cache_path):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as response:
            cache_path.write_bytes(response.read())
    with cache_path.open() as file:
        return [json.loads(line) for line in file]


def convert_hans(rows, split):
    converted = []
    for index, row in enumerate(rows):
        entailment = row["gold_label"] == "entailment"
        converted.append(
            {
                "id": f"hans-{split}-{row.get('pairID', index)}",
                "context": row["sentence1"],
                "assertion": row["sentence2"],
                "target": float(entailment),
                "source": "hans",
                "capability": f"hans_{row.get('heuristic', 'unknown')}",
                "nli_label": "entailment" if entailment else "neutral",
                "split_protocol": "official",
            }
        )
    return converted


def split_hans_training(rows, seed, validation_fraction=0.1):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    validation_size = int(len(rows) * validation_fraction)
    return rows[validation_size:], rows[:validation_size]


def split_validation_for_calibration(rows, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("capability"), row.get("target", row.get("label")))].append(row)
    model_validation = []
    calibration = []
    calibration_selection = []
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
        calibration_size = max(1, int(len(group) * 0.2)) if len(group) >= 5 else 0
        selection_size = calibration_size
        calibration.extend(group[:calibration_size])
        calibration_selection.extend(
            group[calibration_size : calibration_size + selection_size]
        )
        model_validation.extend(group[calibration_size + selection_size :])
    return model_validation, calibration, calibration_selection


def frozen_challenge_pairs():
    return {
        (context.strip(), assertion.strip())
        for cases in FROZEN_CHALLENGE_CASES.values()
        for context, assertion in cases
    }


def assert_no_challenge_overlap(rows):
    forbidden = frozen_challenge_pairs()
    overlap = {
        (row["context"].strip(), row["assertion"].strip())
        for row in rows
    } & forbidden
    if overlap:
        raise RuntimeError("generated data overlaps the frozen challenge")


def print_stats(name, rows):
    print(f"\n{name}\n{'-' * 60}")
    print("total:", f"{len(rows):,}")
    for field in (
        "target",
        "source",
        "capability",
        "nli_label",
        "relevance_label",
        "strength_label",
    ):
        counts = Counter(row.get(field) for row in rows if row.get(field) is not None)
        if counts:
            print(f"{field}: {dict(sorted(counts.items()))}")


def prepare(data_dir=Path("data"), extra_train_data=()):
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("install project dependencies before preparing data") from error

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    base = {
        split: upgrade_v1_1(
            read_jsonl(data_dir / f"v1_1_{split}.jsonl"), split
        )
        for split in ("train", "validation", "test")
    }

    squad = load_dataset("rajpurkar/squad")
    squad_parts = squad["train"].train_test_split(test_size=0.05, seed=240119)
    squad_rows = {
        "train": squad_parts["train"],
        "validation": squad_parts["test"],
        "test": squad["validation"],
    }
    relevance = {
        split: build_hard_relevance(
            squad_rows[split],
            split,
            DEFAULT_RELEVANCE_LIMITS[split],
            SPLIT_SEEDS[split],
        )
        for split in ("train", "validation", "test")
    }

    hans_train_raw = download_jsonl(
        HANS_URLS["train"], data_dir / "cache/hans_train.jsonl"
    )
    hans_test_raw = download_jsonl(
        HANS_URLS["test"], data_dir / "cache/hans_test.jsonl"
    )
    hans_train, hans_validation = split_hans_training(
        convert_hans(hans_train_raw, "train"), SPLIT_SEEDS["train"]
    )
    hans_test = convert_hans(hans_test_raw, "test")

    synthetic = {
        split: generate_v1_2_synthetic(
            split, DEFAULT_SYNTHETIC_COUNTS[split]
        )
        for split in ("train", "validation", "test")
    }
    supplemental = []
    for path in extra_train_data:
        supplemental.extend(read_jsonl(path))

    validation_all = (
        base["validation"]
        + relevance["validation"]
        + hans_validation
        + synthetic["validation"]
    )
    model_validation, calibration, calibration_selection = (
        split_validation_for_calibration(
            validation_all, SPLIT_SEEDS["validation"]
        )
    )
    combined = {
        "train": base["train"] + relevance["train"] + hans_train + synthetic["train"] + supplemental,
        "validation": model_validation,
        "calibration": calibration,
        "calibration_selection": calibration_selection,
        "test": base["test"] + relevance["test"] + synthetic["test"],
    }
    for split, rows in combined.items():
        assert_no_challenge_overlap(rows)
        print_stats(split, rows)
        write_jsonl(
            data_dir / f"v1_2_{split}.jsonl",
            rows,
            SPLIT_SEEDS.get(split, SPLIT_SEEDS["validation"]),
        )

    assert_no_challenge_overlap(hans_test)
    write_jsonl(data_dir / "v1_2_hans_test.jsonl", hans_test, SPLIT_SEEDS["test"])
    print_stats("hans official test", hans_test)
    return {**combined, "hans_test": hans_test}


def main():
    parser = argparse.ArgumentParser(description="Prepare CAPE v1.2 experiment data")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--extra-train-data",
        action="append",
        default=[],
        metavar="JSONL",
    )
    args = parser.parse_args()
    configured = [
        path
        for path in os.environ.get("CAPE_EXTRA_TRAIN_DATA", "").split(os.pathsep)
        if path
    ]
    extra_paths = list(dict.fromkeys(configured + args.extra_train_data))
    prepare(Path(args.data_dir), extra_paths)


if __name__ == "__main__":
    main()
