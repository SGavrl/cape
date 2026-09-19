import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path


SEED = 240119
SPLIT_SEEDS = {
    "train": SEED,
    "validation": SEED + 1,
    "test": SEED + 2,
}
SNLI_TRAIN_PER_LABEL = 40_000
SQUAD_TRAIN_ROWS = 35_000
SYNTHETIC_COUNTS = {
    "train": 5_000,
    "validation": 500,
    "test": 500,
}

SNLI_LABELS = {
    0: 1.0,
    1: 0.5,
    2: 0.0,
}

RELEVANCE_ASSERTIONS = (
    "This document is relevant to the question.",
    "This passage helps answer the question.",
    "This document contains information useful for answering the question.",
)


def cape_example(context, assertion, label, source):
    if not isinstance(context, str) or not isinstance(assertion, str):
        raise ValueError("context and assertion must be strings")
    if not isinstance(source, str) or not source:
        raise ValueError("source must be a non-empty string")
    label = float(label)
    if not 0.0 <= label <= 1.0:
        raise ValueError(f"label must be between 0 and 1, got {label}")
    return {
        "context": context.strip(),
        "assertion": assertion.strip(),
        "label": label,
        "source": source,
    }


def convert_snli_row(row):
    label = row["label"]
    if label not in SNLI_LABELS:
        return None
    return cape_example(
        row["premise"],
        row["hypothesis"],
        SNLI_LABELS[label],
        "snli",
    )


def convert_snli_split(split, *, include_neutral, per_label_limit=None):
    counts = Counter()
    examples = []
    included_labels = (0, 1, 2) if include_neutral else (0, 2)
    for row in split:
        if per_label_limit is not None and all(
            counts[label] >= per_label_limit for label in included_labels
        ):
            break
        label = row["label"]
        if label not in SNLI_LABELS or (label == 1 and not include_neutral):
            continue
        if per_label_limit is not None and counts[label] >= per_label_limit:
            continue
        examples.append(convert_snli_row(row))
        counts[label] += 1
    return examples


def pair_squad_rows(rows, seed=SEED):
    rows = list(rows)
    if len(rows) < 2:
        raise ValueError("SQuAD pairing requires at least two rows")

    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    pairs = []

    for position, row_index in enumerate(order):
        row = rows[row_index]
        negative_index = None
        fallback_index = None

        for offset in range(1, len(order)):
            candidate_index = order[(position + offset) % len(order)]
            candidate = rows[candidate_index]
            if candidate_index == row_index or candidate["context"] == row["context"]:
                continue
            if fallback_index is None:
                fallback_index = candidate_index
            if candidate.get("title") != row.get("title"):
                negative_index = candidate_index
                break

        negative_index = negative_index if negative_index is not None else fallback_index
        if negative_index is None:
            raise ValueError("Could not find a distinct negative SQuAD context")
        pairs.append((row, rows[negative_index]))

    return pairs


def convert_squad_rows(rows, seed=SEED):
    examples = []
    for index, (row, negative_row) in enumerate(pair_squad_rows(rows, seed)):
        assertion = RELEVANCE_ASSERTIONS[index % len(RELEVANCE_ASSERTIONS)]
        question = row["question"].strip()
        examples.append(
            cape_example(
                f"Question: {question}\nDocument: {row['context'].strip()}",
                assertion,
                1.0,
                "squad_relevance",
            )
        )
        examples.append(
            cape_example(
                f"Question: {question}\nDocument: {negative_row['context'].strip()}",
                assertion,
                0.0,
                "squad_relevance",
            )
        )
    return examples


def _temporal_example(rng, index):
    names = rng.sample(
        ("Iris", "Jon", "Kemi", "Luis", "Mei", "Nora", "Omar", "Pia"),
        3,
    )
    first, second, third = names
    schedule = f"{index + 1}-{rng.randrange(1_000_000):06d}"
    context = (
        f"For schedule {schedule}, {first}'s task finished before {second}'s task. "
        f"{second}'s task finished before {third}'s task."
    )
    if index % 2:
        return context, f"{first}'s task finished before {third}'s task.", 1.0
    return context, f"{third}'s task finished before {first}'s task.", 0.0


def _quantifier_example(rng, index):
    group = f"batch-{index + 1}-{rng.randrange(1_000_000):06d}"
    cases = (
        (
            f"Every parcel in {group} has a seal. At least one parcel is in {group}.",
            f"At least one parcel in {group} has a seal.",
            1.0,
        ),
        (
            f"Some parcel in {group} has a seal. Nothing is known about the remaining parcels.",
            f"Every parcel in {group} has a seal.",
            0.0,
        ),
        (
            f"Some parcel in {group} has a seal.",
            f"At least one parcel in {group} has a seal.",
            1.0,
        ),
        (
            f"No parcel in {group} has a seal.",
            f"At least one parcel in {group} has a seal.",
            0.0,
        ),
    )
    return cases[index % len(cases)]


def _numeric_example(rng, index):
    before = rng.randint(25, 900)
    delta = rng.randint(1, min(before, 120))
    if index % 4 in (0, 1):
        after = before + delta
        assertion = rng.choice(
            (
                f"The value increased by {delta} units.",
                "The final value is greater than the initial value.",
            )
        )
        label = 1.0
    else:
        after = before - delta
        assertion = rng.choice(
            (
                f"The value increased by {delta} units.",
                "The final value is greater than the initial value.",
            )
        )
        label = 0.0
    measurement = f"{index + 1}-{rng.randrange(1_000_000):06d}"
    return (
        f"Measurement {measurement} changed from {before} units to {after} units.",
        assertion,
        label,
    )


def _systems_example(rng, index):
    component = f"component-{index + 1}-{rng.randrange(1_000_000):06d}"
    cases = (
        (
            f"{component}'s storage volume reports zero free blocks before an append operation.",
            "The append operation may return an insufficient-space error.",
            1.0,
        ),
        (
            f"{component}'s bounded event buffer is at capacity, and new events are rejected until space is freed.",
            "The next event may be rejected while the buffer remains at capacity.",
            1.0,
        ),
        (
            f"Two threads modify {component}'s shared map without synchronization.",
            "The updates have a race-condition risk.",
            1.0,
        ),
        (
            f"{component} releases a buffer and later dereferences the stale address.",
            "The stale dereference is a memory-safety problem.",
            1.0,
        ),
        (
            f"A patch adds a blocking pause to {component}, which executes once for every request.",
            "The patch can reduce request throughput.",
            1.0,
        ),
        (
            f"A patch edits only prose documentation beside {component}; all executable statements are identical.",
            "The prose edit alone changes runtime results.",
            0.0,
        ),
    )
    return cases[index % len(cases)]


SYNTHETIC_GENERATORS = {
    "temporal": _temporal_example,
    "quantifiers": _quantifier_example,
    "numeric": _numeric_example,
    "systems": _systems_example,
}


def generate_synthetic_examples(split, count_per_category, seed=None):
    if split not in SPLIT_SEEDS:
        raise ValueError(f"Unknown split: {split}")
    if count_per_category < 0:
        raise ValueError("count_per_category must be at least 0")
    rng = random.Random(SPLIT_SEEDS[split] if seed is None else seed)
    examples = []
    for category, generator in SYNTHETIC_GENERATORS.items():
        for index in range(count_per_category):
            context, assertion, label = generator(rng, index)
            examples.append(
                cape_example(
                    context,
                    assertion,
                    label,
                    f"synthetic_{category}",
                )
            )
    rng.shuffle(examples)
    return examples


def read_jsonl(path, *, default_source=None):
    examples = []
    with Path(path).open() as file:
        for line_number, line in enumerate(file, 1):
            try:
                row = json.loads(line)
                examples.append(
                    cape_example(
                        row["context"],
                        row["assertion"],
                        row["label"],
                        row.get("source", default_source or "unknown"),
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid row at {path}:{line_number}: {error}") from error
    return examples


def load_supplemental_examples(paths):
    examples = []
    for path in paths:
        examples.extend(
            read_jsonl(
                Path(path),
                default_source="supplemental",
            )
        )
    return examples


def extra_train_paths(cli_paths=(), environment=None):
    environment = os.environ if environment is None else environment
    configured = environment.get("CAPE_EXTRA_TRAIN_DATA", "")
    paths = [path for path in configured.split(os.pathsep) if path]
    paths.extend(cli_paths)
    return list(dict.fromkeys(paths))


def write_jsonl(path, examples, seed):
    examples = list(examples)
    random.Random(seed).shuffle(examples)
    with Path(path).open("w") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")


def print_stats(name, examples):
    sources = Counter(example["source"] for example in examples)
    targets = Counter(example["label"] for example in examples)
    soft_count = sum(example["label"] not in (0.0, 1.0) for example in examples)
    print(f"\n{name}\n{'-' * 60}")
    print(f"total:       {len(examples):,}")
    print(f"targets:     {dict(sorted(targets.items()))}")
    print(f"soft labels: {soft_count:,}")
    for source, count in sorted(sources.items()):
        print(f"  {source:<24} {count:>10,}")


def prepare(data_dir=Path("data"), extra_train_data=()):
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Dataset preparation requires the 'datasets' package. "
            "Install the project dependencies before running this command."
        ) from error

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    mix = {
        split: read_jsonl(data_dir / f"mix_{split}.jsonl")
        for split in ("train", "validation", "test")
    }

    print("Loading SNLI...")
    snli = load_dataset("stanfordnlp/snli")
    snli_examples = {
        "train": convert_snli_split(
            snli["train"].shuffle(seed=SPLIT_SEEDS["train"]),
            include_neutral=True,
            per_label_limit=SNLI_TRAIN_PER_LABEL,
        ),
        "validation": convert_snli_split(
            snli["validation"], include_neutral=False
        ),
        "test": convert_snli_split(snli["test"], include_neutral=False),
    }

    print("Loading SQuAD...")
    squad = load_dataset("rajpurkar/squad")
    squad_parts = squad["train"].train_test_split(
        test_size=0.05, seed=SEED
    )
    squad_train = squad_parts["train"].shuffle(seed=SEED)
    squad_train = squad_train.select(
        range(min(SQUAD_TRAIN_ROWS, len(squad_train)))
    )
    squad_examples = {
        "train": convert_squad_rows(squad_train, SPLIT_SEEDS["train"]),
        "validation": convert_squad_rows(
            squad_parts["test"], SPLIT_SEEDS["validation"]
        ),
        "test": convert_squad_rows(squad["validation"], SPLIT_SEEDS["test"]),
    }

    synthetic = {
        split: generate_synthetic_examples(
            split, SYNTHETIC_COUNTS[split], SPLIT_SEEDS[split]
        )
        for split in ("train", "validation", "test")
    }
    supplemental = load_supplemental_examples(extra_train_data)

    combined = {}
    for split in ("train", "validation", "test"):
        combined[split] = (
            mix[split]
            + snli_examples[split]
            + squad_examples[split]
            + synthetic[split]
        )
        if split == "train":
            combined[split].extend(supplemental)
        print_stats(split, combined[split])
        write_jsonl(
            data_dir / f"v1_1_{split}.jsonl",
            combined[split],
            SPLIT_SEEDS[split],
        )

    return combined


def main():
    parser = argparse.ArgumentParser(description="Prepare CAPE v1.1 datasets.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--extra-train-data",
        action="append",
        default=[],
        metavar="JSONL",
        help=(
            "Additional JSONL training examples. May be supplied more than once; "
            "CAPE_EXTRA_TRAIN_DATA provides the same setting."
        ),
    )
    args = parser.parse_args()
    outputs = prepare(
        Path(args.data_dir),
        extra_train_data=extra_train_paths(args.extra_train_data),
    )
    print("\nWrote:")
    for split in outputs:
        print(f"  {Path(args.data_dir) / f'v1_1_{split}.jsonl'}")


if __name__ == "__main__":
    main()
