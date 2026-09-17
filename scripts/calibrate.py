import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from cape.cape_model import CAPEModel
from cape.dataset import CAPEDataset, make_collate_fn


BATCH_SIZE = 64
MAX_LENGTH = 256
ECE_BINS = 15


def collect_logits(
    model,
    tokenizer,
    dataset,
    device,
):
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

    logits_all = []
    labels_all = []

    model.eval()

    with torch.inference_mode():
        for batch in loader:
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
                    "Non-finite logits detected."
                )

            logits_all.append(
                logits.float().cpu()
            )

            labels_all.append(
                labels.float().cpu()
            )

    return (
        torch.cat(logits_all),
        torch.cat(labels_all),
    )


def binary_ece(
    probabilities,
    labels,
    bins=15,
):
    probabilities = np.asarray(
        probabilities
    )

    labels = np.asarray(
        labels
    )

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    ece = 0.0

    for i in range(bins):
        lower = edges[i]
        upper = edges[i + 1]

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
            continue

        predicted = (
            probabilities[mask].mean()
        )

        observed = (
            labels[mask].mean()
        )

        weight = (
            count
            / len(probabilities)
        )

        ece += (
            weight
            * abs(
                predicted - observed
            )
        )

    return float(ece)


def metrics(
    logits,
    labels,
    temperature,
):
    scaled_logits = (
        logits / temperature
    )

    probabilities = torch.sigmoid(
        scaled_logits
    )

    predictions = (
        probabilities >= 0.5
    ).float()

    accuracy = (
        predictions == labels
    ).float().mean().item()

    nll = (
        torch.nn.functional
        .binary_cross_entropy_with_logits(
            scaled_logits,
            labels,
        )
        .item()
    )

    brier = torch.mean(
        (
            probabilities - labels
        ) ** 2
    ).item()

    ece = binary_ece(
        probabilities.numpy(),
        labels.numpy(),
        ECE_BINS,
    )

    return {
        "accuracy": accuracy,
        "nll": nll,
        "brier_score": brier,
        "ece": ece,
    }


def fit_temperature(
    logits,
    labels,
):
    # Optimize log(T), which guarantees
    # that temperature always stays > 0.
    log_temperature = torch.zeros(
        (),
        dtype=torch.float32,
        requires_grad=True,
    )

    criterion = (
        torch.nn.BCEWithLogitsLoss()
    )

    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()

        temperature = (
            log_temperature.exp()
        )

        loss = criterion(
            logits / temperature,
            labels,
        )

        loss.backward()

        return loss

    optimizer.step(closure)

    temperature = (
        log_temperature
        .detach()
        .exp()
        .item()
    )

    return temperature


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/"
            "mix_v1/last.pt"
        ),
    )

    parser.add_argument(
        "--validation",
        default=(
            "data/"
            "mix_validation.jsonl"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "checkpoints/"
            "cape_mix_v1.pt"
        ),
    )

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device(
            "cuda"
        )
    else:
        device = torch.device(
            "cpu"
        )

    if device.type == "cuda":
        torch.set_float32_matmul_precision(
            "high"
        )

    print(
        "Device:",
        device,
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
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

    model = model.to(
        device
    )

    dataset = CAPEDataset(
        args.validation
    )

    print(
        "Validation examples:",
        f"{len(dataset):,}",
    )

    logits, labels = collect_logits(
        model,
        tokenizer,
        dataset,
        device,
    )

    before = metrics(
        logits,
        labels,
        temperature=1.0,
    )

    temperature = fit_temperature(
        logits,
        labels,
    )

    after = metrics(
        logits,
        labels,
        temperature=temperature,
    )

    print()
    print(
        f"Temperature: "
        f"{temperature:.6f}"
    )

    print()
    print("Before calibration")
    print("-" * 40)

    for key, value in before.items():
        print(
            f"{key:<15} "
            f"{value:.6f}"
        )

    print()
    print("After calibration")
    print("-" * 40)

    for key, value in after.items():
        print(
            f"{key:<15} "
            f"{value:.6f}"
        )

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint[
        "temperature"
    ] = temperature

    checkpoint[
        "model_version"
    ] = "cape-mix-v1"

    checkpoint[
        "base_checkpoint"
    ] = args.checkpoint

    checkpoint[
        "calibration"
    ] = {
        "validation_data": (
            args.validation
        ),
        "temperature": (
            temperature
        ),
        "before": before,
        "after": after,
    }

    torch.save(
        checkpoint,
        output,
    )

    print()
    print(
        f"Saved calibrated model to "
        f"{output}"
    )


if __name__ == "__main__":
    main()
