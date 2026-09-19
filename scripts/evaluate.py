import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from cape.calibration import calibrated_probabilities, calibration_from_checkpoint
from cape.cape_model import SHARED_CONTEXT
from cape.checkpoints import (
    build_model_from_checkpoint,
    checkpoint_architecture,
    checkpoint_backbone,
)
from cape.dataset import make_collate_fn, make_shared_context_collate_fn
from cape.metrics import metrics_from_arrays


DEFAULT_CHECKPOINTS = [
    "checkpoints/mix_v1/last.pt",
    "checkpoints/cape_mix_v1.pt",
]
DEFAULT_TEST_PATH = "data/mix_test.jsonl"
DEFAULT_OUTPUT_DIR = "evaluation/cape_mix_v1"
BATCH_SIZE = 64


class EvalDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def load_jsonl(path):
    examples = []
    with Path(path).open() as file:
        for line_number, line in enumerate(file, 1):
            row = json.loads(line)
            target = float(row.get("target", row.get("label")))
            if target not in (0.0, 1.0):
                raise ValueError(
                    f"evaluation requires binary labels at {path}:{line_number}"
                )
            examples.append(
                {
                    "context": row["context"],
                    "assertion": row["assertion"],
                    "label": target,
                    "source": row.get("source", "unknown"),
                    "capability": row.get(
                        "capability", row.get("source", "unknown")
                    ),
                    "id": row.get("id"),
                }
            )
    return examples


def checkpoint_id(checkpoint_path):
    path = Path(checkpoint_path)
    parts = list(path.parts)
    if "checkpoints" in parts:
        parts = parts[parts.index("checkpoints") + 1 :]
    return "_".join(Path(part).stem for part in parts)


def run_inference(
    model,
    tokenizer,
    examples,
    device,
    calibration,
    architecture,
    max_length,
):
    dataset = EvalDataset(examples)
    collate = (
        make_shared_context_collate_fn(
            tokenizer, max_length, include_auxiliary=False
        )
        if architecture == SHARED_CONTEXT
        else make_collate_fn(tokenizer, max_length)
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    probabilities = []
    labels = []
    model.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            batch_labels = batch.pop("labels").to(device, non_blocking=True)
            inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            logits = model(**inputs)
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"non-finite logits in batch {batch_index}")
            probabilities.append(
                calibrated_probabilities(logits, calibration).cpu()
            )
            labels.append(batch_labels.cpu())
    return torch.cat(probabilities).numpy(), torch.cat(labels).numpy()


def grouped_metrics(probabilities, labels, examples, field):
    indexes = defaultdict(list)
    for index, example in enumerate(examples):
        indexes[example[field]].append(index)
    return {
        name: metrics_from_arrays(
            probabilities[np.asarray(group, dtype=np.int64)],
            labels[np.asarray(group, dtype=np.int64)],
        )
        for name, group in sorted(indexes.items())
    }


def high_confidence_mistakes(probabilities, labels, examples, threshold=0.90):
    predictions = (probabilities >= 0.5).astype(np.float32)
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    indexes = np.flatnonzero(
        (predictions != labels) & (confidence >= threshold)
    )
    return [
        {
            "index": int(index),
            "id": examples[index]["id"],
            "source": examples[index]["source"],
            "capability": examples[index]["capability"],
            "context": examples[index]["context"],
            "assertion": examples[index]["assertion"],
            "label": int(labels[index]),
            "probability": float(probabilities[index]),
            "confidence": float(confidence[index]),
        }
        for index in indexes
    ]


def print_metrics(name, metrics, compact=False):
    auroc = "n/a" if metrics["auroc"] is None else f"{metrics['auroc']:.4f}"
    if compact:
        print(
            f"{name:<28} n={metrics['examples']:>7,} "
            f"acc={metrics['accuracy']:.4f} auc={auroc:>6} "
            f"nll={metrics['nll']:.4f} brier={metrics['brier_score']:.4f} "
            f"ece={metrics['ece']:.4f}"
        )
        return
    print(f"Examples:       {metrics['examples']:,}")
    print(f"Accuracy:       {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})")
    print(f"AUROC:          {auroc}")
    print(f"NLL:            {metrics['nll']:.4f}")
    print(f"Brier score:    {metrics['brier_score']:.4f}")
    print(f"ECE:            {metrics['ece']:.4f}")
    print(f"Adaptive ECE:   {metrics['adaptive_ece']:.4f}")
    for threshold, values in metrics["high_confidence_errors"].items():
        print(
            f"Errors >= {threshold}: "
            f"{values['errors']}/{values['predictions']}"
        )


def save_reliability_diagram(metrics, output_path, title):
    populated = [
        item
        for item in metrics["probability_bins"]
        if item["count"] and item["mean_probability"] is not None
    ]
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(
        [item["mean_probability"] for item in populated],
        [item["positive_rate"] for item in populated],
        marker="o",
        label="CAPE",
    )
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed positive rate")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def evaluate_checkpoint(checkpoint_path, examples, device, output_dir):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    architecture = checkpoint_architecture(checkpoint)
    backbone = checkpoint_backbone(checkpoint)
    calibration = calibration_from_checkpoint(checkpoint)
    max_length = int(checkpoint.get("max_length", 256))
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = build_model_from_checkpoint(checkpoint).to(device)
    probabilities, labels = run_inference(
        model,
        tokenizer,
        examples,
        device,
        calibration,
        architecture,
        max_length,
    )
    result = {
        "checkpoint": checkpoint_path,
        "epoch": checkpoint.get("epoch"),
        "architecture": architecture,
        "backbone": backbone,
        "calibration": calibration,
        "overall": metrics_from_arrays(probabilities, labels),
        "by_source": grouped_metrics(
            probabilities, labels, examples, "source"
        ),
        "by_capability": grouped_metrics(
            probabilities, labels, examples, "capability"
        ),
        "high_confidence_mistakes": high_confidence_mistakes(
            probabilities, labels, examples
        ),
    }
    name = checkpoint_id(checkpoint_path)
    save_reliability_diagram(
        result["overall"],
        output_dir / f"{name}_reliability.png",
        f"CAPE reliability — {name}",
    )
    (output_dir / f"{name}_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(f"\n{'=' * 80}\n{name} ({architecture})\n{'=' * 80}")
    print_metrics(name, result["overall"])
    print("\nPer source")
    for source, metrics in result["by_source"].items():
        print_metrics(source, metrics, compact=True)
    print("\nPer capability")
    for capability, metrics in result["by_capability"].items():
        print_metrics(capability, metrics, compact=True)
    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate CAPE checkpoints")
    parser.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--data", default=DEFAULT_TEST_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    examples = load_jsonl(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Device:", device)
    print("Test examples:", f"{len(examples):,}")
    results = [
        evaluate_checkpoint(path, examples, device, output_dir)
        for path in args.checkpoints
    ]
    (output_dir / "summary.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print("\nOVERALL SUMMARY")
    print(
        f"{'checkpoint':<28} {'acc':>8} {'AUROC':>8} "
        f"{'NLL':>8} {'Brier':>8} {'ECE':>8} {'ACE':>8}"
    )
    for result in results:
        metrics = result["overall"]
        auroc = "n/a" if metrics["auroc"] is None else f"{metrics['auroc']:.4f}"
        print(
            f"{checkpoint_id(result['checkpoint']):<28} "
            f"{metrics['accuracy']:>8.4f} {auroc:>8} "
            f"{metrics['nll']:>8.4f} {metrics['brier_score']:>8.4f} "
            f"{metrics['ece']:>8.4f} {metrics['adaptive_ece']:>8.4f}"
        )
    print("Saved results to", output_dir)


if __name__ == "__main__":
    main()
