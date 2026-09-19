import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from cape.cape_model import CAPEModel
from cape.dataset import CAPEDataset, make_collate_fn


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = os.environ.get(
    "CAPE_MODEL_NAME",
    "microsoft/deberta-v3-small",
)

TRAIN_PATH = os.environ.get(
    "CAPE_TRAIN_PATH",
    "data/train.jsonl",
)

VAL_PATH = os.environ.get(
    "CAPE_VAL_PATH",
    "data/validation.jsonl",
)

CHECKPOINT_DIR = Path(
    os.environ.get(
        "CAPE_CHECKPOINT_DIR",
        "checkpoints",
    )
)

INIT_CHECKPOINT = os.environ.get(
    "CAPE_INIT_CHECKPOINT"
)

BATCH_SIZE = int(
    os.environ.get(
        "CAPE_BATCH_SIZE",
        "32",
    )
)

MAX_LENGTH = int(
    os.environ.get(
        "CAPE_MAX_LENGTH",
        "256",
    )
)

EPOCHS = int(
    os.environ.get(
        "CAPE_EPOCHS",
        "2",
    )
)

LEARNING_RATE = float(
    os.environ.get(
        "CAPE_LEARNING_RATE",
        "2e-5",
    )
)

WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06

LOG_INTERVAL = 100

# Set this to an integer like 200 for a short sanity run.
# Leave as None for full training.
MAX_TRAIN_STEPS = None


def load_initial_weights(
    model,
    checkpoint_path,
    *,
    expected_model_name=None,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if "model" not in checkpoint:
        raise ValueError(
            f"Init checkpoint is missing required field 'model': {checkpoint_path}"
        )
    checkpoint_model_name = checkpoint.get("model_name")
    if (
        expected_model_name
        and checkpoint_model_name
        and checkpoint_model_name != expected_model_name
    ):
        raise ValueError(
            "Init checkpoint model_name does not match CAPE_MODEL_NAME: "
            f"{checkpoint_model_name!r} != {expected_model_name!r}"
        )
    model.load_state_dict(checkpoint["model"])


def create_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


def validate_checkpoint_paths(init_checkpoint, checkpoint_dir):
    if not init_checkpoint:
        return
    init_path = Path(init_checkpoint).resolve()
    output_paths = {
        (Path(checkpoint_dir) / "last.pt").resolve(),
        (Path(checkpoint_dir) / "best.pt").resolve(),
    }
    if init_path in output_paths:
        raise ValueError(
            "CAPE_INIT_CHECKPOINT must not be one of the training output paths."
        )


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels").to(
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
                    "Non-finite logits detected "
                    "during evaluation."
                )

            loss = criterion(
                logits,
                labels,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite evaluation loss "
                    f"detected: {loss.item()}"
                )

            probabilities = torch.sigmoid(
                logits.float()
            )

            predictions = (
                probabilities >= 0.5
            ).float()

            total_correct += (
                predictions == labels
            ).sum().item()

            total_examples += (
                labels.size(0)
            )

            total_loss += (
                loss.item()
                * labels.size(0)
            )

    return {
        "loss": (
            total_loss
            / total_examples
        ),
        "accuracy": (
            total_correct
            / total_examples
        ),
    }


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required for this training script."
        )

    device = torch.device("cuda")

    validate_checkpoint_paths(
        INIT_CHECKPOINT,
        CHECKPOINT_DIR,
    )

    # Use fast TF32-style matrix operations on the 4090
    # while keeping the model itself in FP32.
    torch.set_float32_matmul_precision(
        "high"
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print()
    print("CAPE training configuration")
    print("-" * 50)
    print(f"model:       {MODEL_NAME}")
    print(f"train:       {TRAIN_PATH}")
    print(f"validation:  {VAL_PATH}")
    print(f"checkpoints: {CHECKPOINT_DIR}")
    print(f"init:        {INIT_CHECKPOINT or 'pretrained DeBERTa'}")
    print(f"batch size:  {BATCH_SIZE}")
    print(f"max length:  {MAX_LENGTH}")
    print(f"epochs:      {EPOCHS}")
    print(f"lr:          {LEARNING_RATE}")
    print()

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Tokenizer + datasets
    # -----------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    train_dataset = CAPEDataset(
        TRAIN_PATH
    )

    val_dataset = CAPEDataset(
        VAL_PATH
    )

    print(
        f"Training examples:   "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation examples: "
        f"{len(val_dataset):,}"
    )

    collate_fn = make_collate_fn(
        tokenizer,
        MAX_LENGTH,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = CAPEModel(
        MODEL_NAME
    )

    if INIT_CHECKPOINT:
        print(
            f"Loading model weights from init checkpoint: {INIT_CHECKPOINT}"
        )
        load_initial_weights(
            model,
            INIT_CHECKPOINT,
            expected_model_name=MODEL_NAME,
        )

    model = model.to(device)
    optimizer = create_optimizer(model)

    full_training_steps = (
        len(train_loader)
        * EPOCHS
    )

    if MAX_TRAIN_STEPS is None:
        scheduler_steps = (
            full_training_steps
        )
    else:
        scheduler_steps = min(
            full_training_steps,
            MAX_TRAIN_STEPS,
        )

    warmup_steps = int(
        scheduler_steps
        * WARMUP_RATIO
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=scheduler_steps,
        )
    )

    criterion = (
        torch.nn.BCEWithLogitsLoss()
    )

    print()
    print(
        f"Steps per epoch: "
        f"{len(train_loader):,}"
    )

    print(
        f"Planned training steps: "
        f"{scheduler_steps:,}"
    )

    print(
        f"Warmup steps: "
        f"{warmup_steps:,}"
    )

    # -----------------------------------------------------
    # Main training loop
    # -----------------------------------------------------

    best_val_loss = math.inf

    global_step = 0
    stop_training = False

    for epoch in range(EPOCHS):
        model.train()

        print()
        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        running_loss = 0.0
        running_steps = 0

        for batch in train_loader:
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

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(**batch)

            if not torch.isfinite(
                logits
            ).all():
                raise RuntimeError(
                    "Non-finite logits detected "
                    "during training."
                )

            loss = criterion(
                logits,
                labels,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite training loss "
                    f"detected: {loss.item()}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()
            scheduler.step()

            running_loss += (
                loss.item()
            )

            running_steps += 1
            global_step += 1

            # ---------------------------------------------
            # Logging
            # ---------------------------------------------

            if (
                running_steps
                >= LOG_INTERVAL
            ):
                avg_loss = (
                    running_loss
                    / running_steps
                )

                lr = (
                    scheduler
                    .get_last_lr()[0]
                )

                print(
                    f"step={global_step:05d} "
                    f"loss={avg_loss:.4f} "
                    f"lr={lr:.2e}"
                )

                running_loss = 0.0
                running_steps = 0

            # ---------------------------------------------
            # Optional sanity-run cutoff
            # ---------------------------------------------

            if (
                MAX_TRAIN_STEPS
                is not None
                and global_step
                >= MAX_TRAIN_STEPS
            ):
                stop_training = True
                break

        # Print any partial logging window.
        if running_steps > 0:
            avg_loss = (
                running_loss
                / running_steps
            )

            lr = (
                scheduler
                .get_last_lr()[0]
            )

            print(
                f"step={global_step:05d} "
                f"loss={avg_loss:.4f} "
                f"lr={lr:.2e}"
            )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        metrics = evaluate(
            model,
            val_loader,
            device,
        )

        print()
        print(
            "validation loss: "
            f"{metrics['loss']:.4f}"
        )

        print(
            "validation accuracy: "
            f"{metrics['accuracy']:.2%}"
        )

        # -------------------------------------------------
        # Checkpoint
        # -------------------------------------------------

        checkpoint = {
            "model": model.state_dict(),
            "model_name": MODEL_NAME,
            "init_checkpoint": INIT_CHECKPOINT,
            "epoch": epoch + 1,
            "global_step": global_step,
            "validation": metrics,
            "config": {
                "train_path": TRAIN_PATH,
                "val_path": VAL_PATH,
                "batch_size": BATCH_SIZE,
                "max_length": MAX_LENGTH,
                "epochs": EPOCHS,
                "learning_rate": (
                    LEARNING_RATE
                ),
                "weight_decay": (
                    WEIGHT_DECAY
                ),
                "warmup_ratio": (
                    WARMUP_RATIO
                ),
                "init_checkpoint": (
                    INIT_CHECKPOINT
                ),
            },
        }

        torch.save(
            checkpoint,
            CHECKPOINT_DIR
            / "last.pt",
        )

        if (
            metrics["loss"]
            < best_val_loss
        ):
            best_val_loss = (
                metrics["loss"]
            )

            torch.save(
                checkpoint,
                CHECKPOINT_DIR
                / "best.pt",
            )

            print(
                "saved new best checkpoint"
            )

        if stop_training:
            print(
                f"Sanity run complete "
                f"at step {global_step}."
            )
            break


if __name__ == "__main__":
    main()
