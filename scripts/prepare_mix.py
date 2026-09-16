import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset


SEED = 1337
RNG = random.Random(SEED)

DATA_DIR = Path("data")

OUTPUTS = {
    "train": DATA_DIR / "mix_train.jsonl",
    "validation": DATA_DIR / "mix_validation.jsonl",
    "test": DATA_DIR / "mix_test.jsonl",
}

# AG News is large enough that it would otherwise dominate the mix.
AG_NEWS_TRAIN_LIMIT = 30_000


def cape_example(
    context: str,
    assertion: str,
    label: int,
    source: str,
):
    return {
        "context": context.strip(),
        "assertion": assertion.strip(),
        "label": int(label),
        "source": source,
    }


def read_jsonl(path: Path, source: str):
    examples = []

    with path.open() as f:
        for line in f:
            row = json.loads(line)

            examples.append(
                cape_example(
                    context=row["context"],
                    assertion=row["assertion"],
                    label=row["label"],
                    source=source,
                )
            )

    return examples


def random_wrong_label(correct: int, num_labels: int):
    choices = [
        i
        for i in range(num_labels)
        if i != correct
    ]

    return RNG.choice(choices)


def convert_banking77(
    split,
    label_names,
):
    examples = []

    for row in split:
        text = row["text"]
        correct_name = row["category"]

        wrong_choices = [
            name
            for name in label_names
            if name != correct_name
        ]

        wrong_name = RNG.choice(
            wrong_choices
        )

        correct_name_text = (
            correct_name.replace("_", " ")
        )

        wrong_name_text = (
            wrong_name.replace("_", " ")
        )

        examples.append(
            cape_example(
                context=text,
                assertion=(
                    "This customer request is about "
                    f"{correct_name_text}."
                ),
                label=1,
                source="banking77",
            )
        )

        examples.append(
            cape_example(
                context=text,
                assertion=(
                    "This customer request is about "
                    f"{wrong_name_text}."
                ),
                label=0,
                source="banking77",
            )
        )

    return examples


AG_ASSERTIONS = {
    0: (
        "This news article is about world "
        "or international affairs."
    ),
    1: "This news article is about sports.",
    2: "This news article is about business.",
    3: (
        "This news article is about science "
        "or technology."
    ),
}


def convert_ag_news(split):
    examples = []

    for row in split:
        text = row["text"]
        correct = row["label"]

        wrong = random_wrong_label(
            correct,
            len(AG_ASSERTIONS),
        )

        examples.append(
            cape_example(
                context=text,
                assertion=AG_ASSERTIONS[correct],
                label=1,
                source="ag_news",
            )
        )

        examples.append(
            cape_example(
                context=text,
                assertion=AG_ASSERTIONS[wrong],
                label=0,
                source="ag_news",
            )
        )

    return examples


def convert_sst2(split):
    examples = []

    for row in split:
        text = row["sentence"]
        label = row["label"]

        # Ignore unlabeled GLUE test rows if any
        # are ever accidentally passed here.
        if label not in (0, 1):
            continue

        examples.append(
            cape_example(
                context=text,
                assertion=(
                    "The author expresses "
                    "a positive opinion."
                ),
                label=label,
                source="sst2",
            )
        )

        examples.append(
            cape_example(
                context=text,
                assertion=(
                    "The author expresses "
                    "a negative opinion."
                ),
                label=1 - label,
                source="sst2",
            )
        )

    return examples


def split_train_validation(
    dataset,
    validation_fraction=0.05,
):
    result = dataset.train_test_split(
        test_size=validation_fraction,
        seed=SEED,
    )

    return (
        result["train"],
        result["test"],
    )


def write_jsonl(path, examples):
    RNG.shuffle(examples)

    with path.open("w") as f:
        for example in examples:
            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )


def print_stats(name, examples):
    source_counts = Counter(
        example["source"]
        for example in examples
    )

    label_counts = Counter(
        example["label"]
        for example in examples
    )

    print()
    print(name)
    print("-" * 50)
    print(f"total: {len(examples):,}")
    print(
        "labels:",
        dict(sorted(label_counts.items())),
    )

    for source, count in sorted(
        source_counts.items()
    ):
        print(
            f"{source:<15} {count:>10,}"
        )


def main():
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------
    # Existing MNLI CAPE data
    # -----------------------------------

    print("Loading existing MNLI CAPE data...")

    mnli = {
        "train": read_jsonl(
            DATA_DIR / "train.jsonl",
            "mnli",
        ),
        "validation": read_jsonl(
            DATA_DIR / "validation.jsonl",
            "mnli",
        ),
        "test": read_jsonl(
            DATA_DIR / "test.jsonl",
            "mnli",
        ),
    }

    # -----------------------------------
    # BANKING77
    # -----------------------------------

    print("Downloading BANKING77...")

    # Current Hugging Face `datasets` versions no longer
    # execute BANKING77's legacy dataset script. Load the
    # original CSV files directly instead.
    BANKING77_TRAIN_URL = (
        "https://raw.githubusercontent.com/"
        "PolyAI-LDN/task-specific-datasets/"
        "master/banking_data/train.csv"
    )

    BANKING77_TEST_URL = (
        "https://raw.githubusercontent.com/"
        "PolyAI-LDN/task-specific-datasets/"
        "master/banking_data/test.csv"
    )

    banking = load_dataset(
        "csv",
        data_files={
            "train": BANKING77_TRAIN_URL,
            "test": BANKING77_TEST_URL,
        },
    )

    banking_labels = sorted(
        set(
            banking["train"]["category"]
        )
    )

    if len(banking_labels) != 77:
        raise RuntimeError(
            "Expected 77 BANKING77 labels, "
            f"found {len(banking_labels)}."
        )

    print(
        f"BANKING77 labels: "
        f"{len(banking_labels)}"
    )

    banking_train, banking_val = (
        split_train_validation(
            banking["train"]
        )
    )

    banking_examples = {
        "train": convert_banking77(
            banking_train,
            banking_labels,
        ),
        "validation": convert_banking77(
            banking_val,
            banking_labels,
        ),
        "test": convert_banking77(
            banking["test"],
            banking_labels,
        ),
    }

    # -----------------------------------
    # AG News
    # -----------------------------------

    print("Downloading AG News...")

    ag = load_dataset(
        "fancyzhx/ag_news"
    )

    ag_train_source = (
        ag["train"]
        .shuffle(seed=SEED)
        .select(
            range(
                min(
                    AG_NEWS_TRAIN_LIMIT,
                    len(ag["train"]),
                )
            )
        )
    )

    ag_train, ag_val = (
        split_train_validation(
            ag_train_source
        )
    )

    ag_examples = {
        "train": convert_ag_news(
            ag_train
        ),
        "validation": convert_ag_news(
            ag_val
        ),
        "test": convert_ag_news(
            ag["test"]
        ),
    }

    # -----------------------------------
    # SST-2
    # -----------------------------------

    print("Downloading SST-2...")

    sst = load_dataset(
        "nyu-mll/glue",
        "sst2",
    )

    # Official SST-2 test labels are not
    # available, so:
    #
    # train -> train + our validation
    # official validation -> our test

    sst_train, sst_val = (
        split_train_validation(
            sst["train"]
        )
    )

    sst_examples = {
        "train": convert_sst2(
            sst_train
        ),
        "validation": convert_sst2(
            sst_val
        ),
        "test": convert_sst2(
            sst["validation"]
        ),
    }

    # -----------------------------------
    # Merge
    # -----------------------------------

    combined = {}

    for split in (
        "train",
        "validation",
        "test",
    ):
        combined[split] = (
            mnli[split]
            + banking_examples[split]
            + ag_examples[split]
            + sst_examples[split]
        )

        print_stats(
            split,
            combined[split],
        )

        write_jsonl(
            OUTPUTS[split],
            combined[split],
        )

    print()
    print("Wrote:")
    print(
        f"  {OUTPUTS['train']}"
    )
    print(
        f"  {OUTPUTS['validation']}"
    )
    print(
        f"  {OUTPUTS['test']}"
    )


if __name__ == "__main__":
    main()

