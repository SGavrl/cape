import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from cape.cape_model import CAPEModel
from cape.dataset import CAPEDataset, make_collate_fn


DEFAULT_CHECKPOINTS = [
    "checkpoints/best.pt",
    "checkpoints/last.pt",
]

DEFAULT_TEST_PATH = "data/test.jsonl"
DEFAULT_OUTPUT_DIR = "evaluation"

BATCH_SIZE = 64
MAX_LENGTH = 256
ECE_BINS = 15


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int = 15,
):
    bin_edges = np.linspace(0.0, 1.0, bins + 1)

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

        weight = count / len(probabilities)

        ece += weight * abs(
            accuracy - confidence
        )

        bin_stats.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "confidence": float(confidence),
                "accuracy": float(accuracy),
            }
        )

    return float(ece), bin_stats


def binary_ece(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int = 15,
):
    """
    Calibration of P(y=1).

    For each probability bin, compare:
        mean predicted P(y=1)
    against:
        fraction of actual positives
    """

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

        mean_probability = probabilities[
            mask
        ].mean()

        positive_rate = labels[
            mask
        ].mean()

        weight = count / len(
            probabilities
        )

        ece += weight * abs(
            mean_probability - positive_rate
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


def evaluate_checkpoint(
    checkpoint_path: str,
    dataset: CAPEDataset,
    device: torch.device,
):
    print()
    print("=" * 72)
    print(f"Evaluating: {checkpoint_path}")
    print("=" * 72)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model_name = checkpoint[
        "model_name"
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
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
        pin_memory=True,
    )

    model = CAPEModel(
        model_name
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model = model.to(device)
    model.eval()

    all_probabilities = []
    all_labels = []

    total_loss = 0.0
    total_examples = 0

    criterion = torch.nn.BCEWithLogitsLoss(
        reduction="sum"
    )

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
                for key, value in batch.items()
            }

            logits = model(**batch)

            if not torch.isfinite(
                logits
            ).all():
                raise RuntimeError(
                    "Non-finite logits "
                    f"in batch {batch_index}"
                )

            loss = criterion(
                logits,
                labels,
            )

            probabilities = torch.sigmoid(
                logits
            )

            total_loss += loss.item()
            total_examples += labels.size(0)

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

    predictions = (
        probabilities >= 0.5
    ).astype(np.float32)

    accuracy = float(
        (predictions == labels).mean()
    )

    nll = float(
        total_loss / total_examples
    )

    brier = float(
        np.mean(
            (probabilities - labels) ** 2
        )
    )

    auroc = float(
        roc_auc_score(
            labels,
            probabilities,
        )
    )

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

    metrics = {
        "checkpoint": checkpoint_path,
        "epoch": checkpoint.get(
            "epoch"
        ),
        "examples": int(
            total_examples
        ),
        "accuracy": accuracy,
        "auroc": auroc,
        "nll": nll,
        "brier_score": brier,
        "ece": probability_ece,
        "classification_ece": class_ece,
        "probability_bins": probability_bins,
        "classification_bins": class_ece_bins,
    }

    print(
        f"Epoch:          "
        f"{metrics['epoch']}"
    )
    print(
        f"Examples:       "
        f"{metrics['examples']:,}"
    )
    print(
        f"Accuracy:       "
        f"{accuracy:.4f} "
        f"({accuracy:.2%})"
    )
    print(
        f"AUROC:          "
        f"{auroc:.4f}"
    )
    print(
        f"NLL:            "
        f"{nll:.4f}"
    )
    print(
        f"Brier score:    "
        f"{brier:.4f}"
    )
    print(
        f"ECE:            "
        f"{probability_ece:.4f}"
    )

    return metrics


def save_reliability_diagram(
    metrics,
    output_path: Path,
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

    plt.title(
        "CAPE Reliability Diagram"
    )

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

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required."
        )

    device = torch.device(
        "cuda"
    )

    torch.set_float32_matmul_precision(
        "high"
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    dataset = CAPEDataset(
        args.data
    )

    print(
        f"Test examples: "
        f"{len(dataset):,}"
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for checkpoint_path in args.checkpoints:
        metrics = evaluate_checkpoint(
            checkpoint_path,
            dataset,
            device,
        )

        results.append(metrics)

        checkpoint_name = (
            Path(checkpoint_path).stem
        )

        save_reliability_diagram(
            metrics,
            output_dir
            / f"{checkpoint_name}_reliability.png",
        )

        with (
            output_dir
            / f"{checkpoint_name}_metrics.json"
        ).open("w") as f:
            json.dump(
                metrics,
                f,
                indent=2,
            )

        # Free GPU memory before loading
        # the next checkpoint.
        torch.cuda.empty_cache()

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print(
        f"{'checkpoint':<12} "
        f"{'acc':>8} "
        f"{'AUROC':>8} "
        f"{'NLL':>8} "
        f"{'Brier':>8} "
        f"{'ECE':>8}"
    )

    for result in results:
        name = Path(
            result["checkpoint"]
        ).name

        print(
            f"{name:<12} "
            f"{result['accuracy']:>8.4f} "
            f"{result['auroc']:>8.4f} "
            f"{result['nll']:>8.4f} "
            f"{result['brier_score']:>8.4f} "
            f"{result['ece']:>8.4f}"
        )

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
