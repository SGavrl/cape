import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from cape.calibration import calibrated_probabilities, fit_calibration
from cape.cape_model import SHARED_CONTEXT
from cape.checkpoints import (
    build_model_from_checkpoint,
    checkpoint_architecture,
    checkpoint_backbone,
)
from cape.dataset import (
    CAPEDataset,
    make_collate_fn,
    make_shared_context_collate_fn,
)
from cape.metrics import metrics_from_arrays


BATCH_SIZE = 64
MAX_LENGTH = 256


def infer_model_version(output_path, checkpoint):
    stem = Path(output_path).stem
    if stem == "cape_v1_1":
        return "cape-v1.1"
    if stem == "cape_mix_v1":
        return "cape-mix-v1"
    return checkpoint.get("model_version", stem.replace("_", "-"))


def collect_logits(
    model, tokenizer, dataset, device, architecture, max_length=MAX_LENGTH
):
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
    logits_all = []
    labels_all = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels").to(device, non_blocking=True)
            inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            logits = model(**inputs)
            if not torch.isfinite(logits).all():
                raise RuntimeError("non-finite logits during calibration")
            logits_all.append(logits.float().cpu())
            labels_all.append(labels.float().cpu())
    return torch.cat(logits_all), torch.cat(labels_all)


def calibration_metrics(logits, labels, parameters):
    probabilities = calibrated_probabilities(logits, parameters)
    metrics = metrics_from_arrays(probabilities.numpy(), labels.numpy())
    return {
        key: metrics[key]
        for key in (
            "accuracy",
            "nll",
            "brier_score",
            "ece",
            "adaptive_ece",
            "high_confidence_errors",
        )
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fit CAPE calibration parameters on validation data"
    )
    parser.add_argument("--checkpoint", default="checkpoints/mix_v1/last.pt")
    parser.add_argument("--validation", default="data/mix_validation.jsonl")
    parser.add_argument("--output", default="checkpoints/cape_mix_v1.pt")
    parser.add_argument("--model-version", default=None)
    parser.add_argument(
        "--method",
        choices=("temperature", "platt", "beta"),
        default="temperature",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    architecture = checkpoint_architecture(checkpoint)
    backbone = checkpoint_backbone(checkpoint)
    max_length = int(checkpoint.get("max_length", MAX_LENGTH))
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = build_model_from_checkpoint(checkpoint).to(device)
    dataset = CAPEDataset(args.validation)
    logits, labels = collect_logits(
        model, tokenizer, dataset, device, architecture, max_length
    )

    identity = {"method": "temperature", "temperature": 1.0}
    before = calibration_metrics(logits, labels, identity)
    parameters = fit_calibration(logits, labels, method=args.method)
    after = calibration_metrics(logits, labels, parameters)

    print("Device:", device)
    print("Validation examples:", f"{len(dataset):,}")
    print("Method:", args.method)
    print("Parameters:", parameters)
    print("Before:", before)
    print("After:", after)

    checkpoint["calibration_parameters"] = parameters
    if parameters["method"] == "temperature":
        checkpoint["temperature"] = parameters["temperature"]
    else:
        checkpoint.pop("temperature", None)
    checkpoint["model_version"] = args.model_version or infer_model_version(
        args.output, checkpoint
    )
    checkpoint["base_checkpoint"] = args.checkpoint
    checkpoint["calibration"] = {
        "validation_data": args.validation,
        "method": args.method,
        "parameters": parameters,
        "before": before,
        "after": after,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print("Saved calibrated model to", output)


if __name__ == "__main__":
    main()
