import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from cape.cape_model import CAPEModel
from cape.dataset import make_collate_fn


DEFAULT_CHECKPOINTS = [
    "checkpoints/mix_v1/last.pt",
    "checkpoints/cape_mix_v1.pt",
]

DEFAULT_TEST_PATH = "data/mix_test.jsonl"
DEFAULT_OUTPUT_DIR = "evaluation/cape_mix_v1"

BATCH_SIZE = 64
MAX_LENGTH = 256
ECE_BINS = 15


class EvalDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def load_jsonl(path):
    examples = []

    with Path(path).open() as f:
        for line in f:
            row = json.loads(line)

            examples.append(
                {
                    "context": row["context"],
                    "assertion": row["assertion"],
                    "label": int(row["label"]),
                    "source": row.get(
                        "source",
                        "unknown",
                    ),
                }
            )

    return examples


def checkpoint_id(checkpoint_path):
    path = Path(checkpoint_path)

    parts = list(path.parts)

    if "checkpoints" in parts:
        index = parts.index("checkpoints")
        parts = parts[index + 1 :]

    return "_".join(
        Path(part).stem
        for part in parts
    )


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int = 15,
):
    bin_edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    ece = 0.0
    bin_stats = []

    for i in range(bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == bins - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        count = int(mask.sum())

        if count == 0:
            bin_stats.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": 0,
                    "confidence": None,
                    "accuracy": None,
                }
            )
            continue

        bin_probs = probabilities[mask]
        bin_labels = labels[mask]

        predictions = (
            bin_probs >= 0.5
        ).astype(np.float32)

        confidence = np.where(
            predictions == 1,
            bin_probs,
            1.0 - bin_probs,
        ).mean()

        accuracy = (
            predictions == bin_labels
        ).mean()

        weight = (
            count / len(probabilities)
        )

        ece += weight * abs(
            accuracy - confidence
        )

        bin_stats.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "confidence": float(
                    confidence
                ),
                "accuracy": float(
                    accuracy
                ),
            }
        )

    return float(ece), bin_stats


def binary_ece(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int = 15,
):
    bin_edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    ece = 0.0
    stats = []

    for i in range(bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == bins - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        count = int(mask.sum())

        if count == 0:
            stats.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": 0,
                    "mean_probability": None,
                    "positive_rate": None,
                }
            )
            continue

        mean_probability = (
            probabilities[mask].mean()
        )

        positive_rate = (
            labels[mask].mean()
        )

        weight = (
            count / len(probabilities)
        )

        ece += weight * abs(
            mean_probability
            - positive_rate
        )

        stats.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_probability": float(
                    mean_probability
                ),
                "positive_rate": float(
                    positive_rate
                ),
            }
        )

    return float(ece), stats


def metrics_from_arrays(
    probabilities,
    labels,
):
    probabilities = np.asarray(
        probabilities,
        dtype=np.float32,
    )

    labels = np.asarray(
        labels,
        dtype=np.float32,
    )

    predictions = (
        probabilities >= 0.5
    ).astype(np.float32)

    accuracy = float(
        (predictions == labels).mean()
    )

    eps = 1e-7
    clipped = np.clip(
        probabilities,
        eps,
        1.0 - eps,
    )

    nll = float(
        -np.mean(
            labels * np.log(clipped)
            + (1.0 - labels)
            * np.log(1.0 - clipped)
        )
    )

    brier = float(
        np.mean(
            (probabilities - labels)
            ** 2
        )
    )

    if len(np.unique(labels)) >= 2:
        auroc = float(
            roc_auc_score(
                labels,
                probabilities,
            )
        )
    else:
        auroc = None

    class_ece, class_ece_bins = (
        expected_calibration_error(
            probabilities,
            labels,
            ECE_BINS,
        )
    )

    probability_ece, probability_bins = (
        binary_ece(
            probabilities,
            labels,
            ECE_BINS,
        )
    )

    return {
        "examples": int(len(labels)),
        "accuracy": accuracy,
        "auroc": auroc,
        "nll": nll,
        "brier_score": brier,
        "ece": probability_ece,
        "classification_ece": (
            class_ece
        ),
        "probability_bins": (
            probability_bins
        ),
        "classification_bins": (
            class_ece_bins
        ),
    }


def run_inference(
    model,
    tokenizer,
    examples,
    device,
    temperature,
):
    dataset = EvalDataset(
        examples
    )

    collate_fn = make_collate_fn(
        tokenizer,
        MAX_LENGTH,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    all_probabilities = []
    all_labels = []

    with torch.inference_mode():
        for batch_index, batch in enumerate(
            loader
        ):
            labels = batch.pop(
                "labels"
            ).to(
                device,
                non_blocking=True,
            )

            batch = {
                key: value.to(
                    device,
                    non_blocking=True,
                )
                for key, value
                in batch.items()
            }

            logits = model(**batch)

            if not torch.isfinite(
                logits
            ).all():
                raise RuntimeError(
                    "Non-finite logits "
                    f"in batch {batch_index}"
                )

            probabilities = torch.sigmoid(
                logits.float()
                / temperature
            )

            all_probabilities.append(
                probabilities.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

    probabilities = torch.cat(
        all_probabilities
    ).numpy()

    labels = torch.cat(
        all_labels
    ).numpy()

    return probabilities, labels


def evaluate_checkpoint(
    checkpoint_path,
    examples,
    device,
):
    print()
    print("=" * 72)
    print(
        f"Evaluating: {checkpoint_path}"
    )
    print("=" * 72)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    temperature = float(
        checkpoint.get(
            "temperature",
            1.0,
        )
    )

    model_name = checkpoint[
        "model_name"
    ]

    tokenizer = (
        AutoTokenizer.from_pretrained(
            model_name
        )
    )

    model = CAPEModel(
        model_name
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model = model.to(device)
    model.eval()

    probabilities, labels = (
        run_inference(
            model,
            tokenizer,
            examples,
            device,
            temperature,
        )
    )

    overall = metrics_from_arrays(
        probabilities,
        labels,
    )

    source_indexes = defaultdict(
        list
    )

    for index, example in enumerate(
        examples
    ):
        source_indexes[
            example["source"]
        ].append(index)

    by_source = {}

    for source in sorted(
        source_indexes
    ):
        indexes = np.asarray(
            source_indexes[source],
            dtype=np.int64,
        )

        by_source[source] = (
            metrics_from_arrays(
                probabilities[indexes],
                labels[indexes],
            )
        )

    result = {
        "checkpoint": checkpoint_path,
        "epoch": checkpoint.get(
            "epoch"
        ),
        "temperature": temperature,
        "overall": overall,
        "by_source": by_source,
    }

    print(
        f"Temperature:    {temperature:.6f}"
    )

    print_metrics(
        "overall",
        overall,
    )

    print()
    print("Per-source metrics")
    print("-" * 72)

    for source, metrics in (
        by_source.items()
    ):
        print_metrics(
            source,
            metrics,
            compact=True,
        )

    # Make sure the previous model is actually
    # released before evaluating another checkpoint.
    del model
    del tokenizer

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def print_metrics(
    name,
    metrics,
    compact=False,
):
    auroc = metrics["auroc"]

    auroc_text = (
        f"{auroc:.4f}"
        if auroc is not None
        else "n/a"
    )

    if compact:
        print(
            f"{name:<12} "
            f"n={metrics['examples']:>6,} "
            f"acc={metrics['accuracy']:.4f} "
            f"auc={auroc_text:>6} "
            f"nll={metrics['nll']:.4f} "
            f"brier={metrics['brier_score']:.4f} "
            f"ece={metrics['ece']:.4f}"
        )
        return

    print(
        f"Epoch:          "
        f"{name}"
    )
    print(
        f"Examples:       "
        f"{metrics['examples']:,}"
    )
    print(
        f"Accuracy:       "
        f"{metrics['accuracy']:.4f} "
        f"({metrics['accuracy']:.2%})"
    )
    print(
        f"AUROC:          "
        f"{auroc_text}"
    )
    print(
        f"NLL:            "
        f"{metrics['nll']:.4f}"
    )
    print(
        f"Brier score:    "
        f"{metrics['brier_score']:.4f}"
    )
    print(
        f"ECE:            "
        f"{metrics['ece']:.4f}"
    )


def save_reliability_diagram(
    metrics,
    output_path,
    title,
):
    bins = metrics[
        "probability_bins"
    ]

    x = []
    y = []

    for bin_data in bins:
        if (
            bin_data["count"] == 0
            or bin_data[
                "mean_probability"
            ]
            is None
        ):
            continue

        x.append(
            bin_data[
                "mean_probability"
            ]
        )

        y.append(
            bin_data[
                "positive_rate"
            ]
        )

    plt.figure(
        figsize=(6, 6)
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    plt.plot(
        x,
        y,
        marker="o",
        label="CAPE",
    )

    plt.xlabel(
        "Predicted probability"
    )

    plt.ylabel(
        "Observed positive rate"
    )

    plt.title(title)

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.grid(
        alpha=0.25
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=160,
    )

    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=DEFAULT_CHECKPOINTS,
    )

    parser.add_argument(
        "--data",
        default=DEFAULT_TEST_PATH,
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        torch.set_float32_matmul_precision(
            "high"
        )

    print(
        "Device:",
        device,
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    examples = load_jsonl(
        args.data
    )

    print(
        f"Test examples: "
        f"{len(examples):,}"
    )

    source_counts = defaultdict(
        int
    )

    for example in examples:
        source_counts[
            example["source"]
        ] += 1

    print("Test sources:")

    for source in sorted(
        source_counts
    ):
        print(
            f"  {source:<12} "
            f"{source_counts[source]:>7,}"
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for checkpoint_path in (
        args.checkpoints
    ):
        result = evaluate_checkpoint(
            checkpoint_path,
            examples,
            device,
        )

        results.append(result)

        name = checkpoint_id(
            checkpoint_path
        )

        save_reliability_diagram(
            result["overall"],
            output_dir
            / f"{name}_reliability.png",
            f"CAPE Reliability — {name}",
        )

        with (
            output_dir
            / f"{name}_metrics.json"
        ).open("w") as f:
            json.dump(
                result,
                f,
                indent=2,
            )

    print()
    print("=" * 92)
    print("OVERALL SUMMARY")
    print("=" * 92)

    print(
        f"{'checkpoint':<20} "
        f"{'temp':>8} "
        f"{'acc':>8} "
        f"{'AUROC':>8} "
        f"{'NLL':>8} "
        f"{'Brier':>8} "
        f"{'ECE':>8}"
    )

    for result in results:
        metrics = result[
            "overall"
        ]

        name = checkpoint_id(
            result["checkpoint"]
        )

        auroc = metrics["auroc"]

        auroc_text = (
            f"{auroc:.4f}"
            if auroc is not None
            else "n/a"
        )

        print(
            f"{name:<20} "
            f"{result['temperature']:>8.4f} "
            f"{metrics['accuracy']:>8.4f} "
            f"{auroc_text:>8} "
            f"{metrics['nll']:>8.4f} "
            f"{metrics['brier_score']:>8.4f} "
            f"{metrics['ece']:>8.4f}"
        )

    print()
    print("=" * 92)
    print("SOURCE ACCURACY")
    print("=" * 92)

    sources = sorted(
        {
            source
            for result in results
            for source in result[
                "by_source"
            ]
        }
    )

    header = (
        f"{'source':<15}"
        + "".join(
            f"{checkpoint_id(result['checkpoint']):>20}"
            for result in results
        )
    )

    print(header)

    for source in sources:
        row = f"{source:<15}"

        for result in results:
            metrics = result[
                "by_source"
            ].get(source)

            if metrics is None:
                row += (
                    f"{'n/a':>20}"
                )
            else:
                row += (
                    f"{metrics['accuracy']:>20.2%}"
                )

        print(row)

    summary_path = (
        output_dir / "summary.json"
    )

    with summary_path.open(
        "w"
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print(
        f"Saved results to "
        f"{output_dir}/"
    )


if __name__ == "__main__":
    main()
