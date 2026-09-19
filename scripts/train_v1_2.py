import json
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from cape.cape_model import (
    CROSS_ENCODER,
    MULTITASK_CROSS_ENCODER,
    SHARED_CONTEXT,
    build_model,
)
from cape.checkpoints import checkpoint_backbone, initialize_from_checkpoint
from cape.dataset import (
    CAPEDataset,
    make_collate_fn,
    make_shared_context_collate_fn,
)
from cape.losses import DEFAULT_AUXILIARY_WEIGHTS, multitask_loss


BACKBONE = os.environ.get("CAPE_MODEL_NAME", "microsoft/deberta-v3-small")
ARCHITECTURE = os.environ.get(
    "CAPE_ARCHITECTURE", MULTITASK_CROSS_ENCODER
)
POOLING = os.environ.get("CAPE_POOLING", "first_token")
TRAIN_PATH = os.environ.get("CAPE_TRAIN_PATH", "data/v1_2_train.jsonl")
VAL_PATH = os.environ.get("CAPE_VAL_PATH", "data/v1_2_validation.jsonl")
CHECKPOINT_DIR = Path(
    os.environ.get("CAPE_CHECKPOINT_DIR", "checkpoints/cape_v1_2")
)
INIT_CHECKPOINT = os.environ.get("CAPE_INIT_CHECKPOINT")
BATCH_SIZE = int(os.environ.get("CAPE_BATCH_SIZE", "32"))
MAX_LENGTH = int(os.environ.get("CAPE_MAX_LENGTH", "256"))
EPOCHS = int(os.environ.get("CAPE_EPOCHS", "2"))
LEARNING_RATE = float(os.environ.get("CAPE_LEARNING_RATE", "2e-5"))
WEIGHT_DECAY = float(os.environ.get("CAPE_WEIGHT_DECAY", "0.01"))
WARMUP_RATIO = float(os.environ.get("CAPE_WARMUP_RATIO", "0.06"))
SEED = int(os.environ.get("CAPE_SEED", "120240"))
LOG_INTERVAL = int(os.environ.get("CAPE_LOG_INTERVAL", "100"))
MAX_TRAIN_STEPS = os.environ.get("CAPE_MAX_TRAIN_STEPS")
MAX_TRAIN_STEPS = int(MAX_TRAIN_STEPS) if MAX_TRAIN_STEPS else None
AUXILIARY_WEIGHTS = {
    name: float(os.environ.get(f"CAPE_{name.upper()}_LOSS_WEIGHT", default))
    for name, default in DEFAULT_AUXILIARY_WEIGHTS.items()
}


def validate_paths(init_checkpoint, checkpoint_dir):
    if not init_checkpoint:
        return
    init_path = Path(init_checkpoint).resolve()
    outputs = {
        (Path(checkpoint_dir) / "last.pt").resolve(),
        (Path(checkpoint_dir) / "best.pt").resolve(),
    }
    if init_path in outputs:
        raise ValueError("initial checkpoint must not be an output checkpoint")


def load_initialization(model, path, expected_backbone):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    actual_backbone = checkpoint_backbone(checkpoint)
    if actual_backbone != expected_backbone:
        raise ValueError(
            f"checkpoint backbone {actual_backbone!r} does not match "
            f"{expected_backbone!r}"
        )
    return initialize_from_checkpoint(model, checkpoint)


def split_batch(batch, device, multitask):
    labels = batch.pop("labels").to(device, non_blocking=True)
    auxiliary = {}
    for name in ("nli_labels", "relevance_labels", "strength_labels"):
        if name in batch:
            auxiliary[name] = batch.pop(name).to(device, non_blocking=True)
    inputs = {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }
    if not multitask:
        return inputs, {"labels": labels}
    auxiliary["labels"] = labels
    return inputs, auxiliary


def loss_for_batch(model, inputs, labels, multitask):
    if not multitask:
        logits = model(**inputs)
        loss = F.binary_cross_entropy_with_logits(logits, labels["labels"])
        return loss, logits, {"support": loss}, {"support": len(logits)}
    outputs = model(**inputs, return_auxiliary=True)
    loss, losses, counts = multitask_loss(
        outputs,
        labels,
        AUXILIARY_WEIGHTS,
    )
    return loss, outputs["support"], losses, counts


@torch.inference_mode()
def evaluate(model, loader, device, multitask):
    model.eval()
    loss_sum = 0.0
    support_loss_sum = 0.0
    correct = 0
    examples = 0
    for batch in loader:
        inputs, labels = split_batch(batch, device, multitask)
        loss, logits, losses, _ = loss_for_batch(
            model, inputs, labels, multitask
        )
        count = labels["labels"].numel()
        loss_sum += loss.item() * count
        support_loss_sum += losses["support"].item() * count
        predictions = (torch.sigmoid(logits.float()) >= 0.5).float()
        correct += (predictions == labels["labels"]).sum().item()
        examples += count
    return {
        "loss": loss_sum / examples,
        "support_loss": support_loss_sum / examples,
        "accuracy": correct / examples,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for v1.2 training")
    if ARCHITECTURE not in {
        CROSS_ENCODER,
        MULTITASK_CROSS_ENCODER,
        SHARED_CONTEXT,
    }:
        raise ValueError(f"unsupported architecture: {ARCHITECTURE}")

    validate_paths(INIT_CHECKPOINT, CHECKPOINT_DIR)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    multitask = ARCHITECTURE != CROSS_ENCODER

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    train_dataset = CAPEDataset(TRAIN_PATH)
    val_dataset = CAPEDataset(VAL_PATH)
    collate = (
        make_shared_context_collate_fn(
            tokenizer, MAX_LENGTH, include_auxiliary=multitask
        )
        if ARCHITECTURE == SHARED_CONTEXT
        else make_collate_fn(
            tokenizer, MAX_LENGTH, include_auxiliary=multitask
        )
    )
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        collate_fn=collate,
        num_workers=4,
        pin_memory=True,
    )

    model = build_model(BACKBONE, ARCHITECTURE, pooling=POOLING)
    initialization = None
    if INIT_CHECKPOINT:
        initialization = load_initialization(model, INIT_CHECKPOINT, BACKBONE)
    model.to(device)
    # Auxiliary-head initialization consumes RNG values. Reset before the
    # first batch so zero-weight and enabled-head ablations share dropout and
    # data-order randomness.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    planned_steps = len(train_loader) * EPOCHS
    if MAX_TRAIN_STEPS is not None:
        planned_steps = min(planned_steps, MAX_TRAIN_STEPS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(planned_steps * WARMUP_RATIO),
        num_training_steps=planned_steps,
    )

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("architecture:", ARCHITECTURE, flush=True)
    print("backbone:", BACKBONE, flush=True)
    print("pooling:", POOLING, flush=True)
    print("initial checkpoint:", INIT_CHECKPOINT, flush=True)
    initialization_summary = (
        {
            "loaded": len(initialization["loaded"]),
            "missing": initialization["missing"],
            "unexpected": initialization["unexpected"],
        }
        if initialization
        else None
    )
    print("initialization:", initialization_summary, flush=True)
    print("train summary:", json.dumps(train_dataset.summary()), flush=True)
    print("validation summary:", json.dumps(val_dataset.summary()), flush=True)
    print("planned steps:", f"{planned_steps:,}", flush=True)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_support_loss = math.inf
    global_step = 0
    stop = False

    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        running_steps = 0
        for batch in train_loader:
            inputs, labels = split_batch(batch, device, multitask)
            optimizer.zero_grad(set_to_none=True)
            loss, logits, _, _ = loss_for_batch(
                model, inputs, labels, multitask
            )
            if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                raise RuntimeError("non-finite value during training")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            running += loss.item()
            running_steps += 1
            if running_steps == LOG_INTERVAL:
                print(
                    f"step={global_step:06d} "
                    f"loss={running / running_steps:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )
                running = 0.0
                running_steps = 0
            if MAX_TRAIN_STEPS is not None and global_step >= MAX_TRAIN_STEPS:
                stop = True
                break

        metrics = evaluate(model, val_loader, device, multitask)
        print(f"epoch={epoch + 1} validation={metrics}", flush=True)
        checkpoint = {
            "model": model.state_dict(),
            "cape_version": "1.2-candidate",
            "model_version": "cape-v1.2-candidate",
            "architecture": ARCHITECTURE,
            "backbone": BACKBONE,
            "model_name": BACKBONE,
            "pooling": POOLING,
            "epoch": epoch + 1,
            "global_step": global_step,
            "temperature": 1.0,
            "calibration_parameters": {
                "method": "temperature",
                "temperature": 1.0,
            },
            "max_length": MAX_LENGTH,
            "training_config": {
                "train_path": TRAIN_PATH,
                "validation_path": VAL_PATH,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "warmup_ratio": WARMUP_RATIO,
                "seed": SEED,
                "pooling": POOLING,
                "auxiliary_weights": AUXILIARY_WEIGHTS if multitask else {},
                "initial_checkpoint": INIT_CHECKPOINT,
            },
            "dataset_summary": {
                "train": train_dataset.summary(),
                "validation": val_dataset.summary(),
            },
            "validation": metrics,
        }
        torch.save(checkpoint, CHECKPOINT_DIR / "last.pt")
        if metrics["support_loss"] < best_support_loss:
            best_support_loss = metrics["support_loss"]
            torch.save(checkpoint, CHECKPOINT_DIR / "best.pt")
            print("saved new best checkpoint", flush=True)
        if stop:
            break


if __name__ == "__main__":
    main()
