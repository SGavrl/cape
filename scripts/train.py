import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from cape.cape_model import CAPEModel
from cape.dataset import CAPEDataset, make_collate_fn


MODEL_NAME = "microsoft/deberta-v3-small"

TRAIN_PATH = "data/train.jsonl"
VAL_PATH = "data/validation.jsonl"

CHECKPOINT_DIR = Path("checkpoints")

BATCH_SIZE = 32
MAX_LENGTH = 256

EPOCHS = 2

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06

# Temporary sanity-run limit. Set to None for a full training run.
MAX_TRAIN_STEPS = None


def evaluate(model, loader, device):
    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            logits = model(**batch)

            loss = criterion(
                logits,
                labels,
            )

            if not torch.isfinite(logits).all():
                raise RuntimeError(
                    "Non-finite logits detected during evaluation."
                )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite evaluation loss detected: {loss.item()}"
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

            total_examples += labels.size(0)

            total_loss += (
                loss.item() * labels.size(0)
            )

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required for this training script."
        )

    device = torch.device("cuda")

    torch.set_float32_matmul_precision("high")

    print("GPU:", torch.cuda.get_device_name(0))

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    train_dataset = CAPEDataset(
        TRAIN_PATH
    )

    val_dataset = CAPEDataset(
        VAL_PATH
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

    model = CAPEModel(
        MODEL_NAME
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = (
        len(train_loader) * EPOCHS
    )

    warmup_steps = int(
        total_steps * WARMUP_RATIO
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_loss = math.inf

    global_step = 0
    stop_training = False

    for epoch in range(EPOCHS):
        model.train()

        print()
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        running_loss = 0.0

        for step, batch in enumerate(train_loader):
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

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(**batch)

            loss = criterion(
                logits,
                labels,
            )

            if not torch.isfinite(logits).all():
                raise RuntimeError(
                    "Non-finite logits detected during evaluation."
                )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite evaluation loss detected: {loss.item()}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            global_step += 1

            if (
                MAX_TRAIN_STEPS is not None
                and global_step >= MAX_TRAIN_STEPS
            ):
                stop_training = True

            if global_step % 100 == 0:
                avg_loss = (
                    running_loss / 100
                )

                lr = scheduler.get_last_lr()[0]

                print(
                    f"step={global_step:05d} "
                    f"loss={avg_loss:.4f} "
                    f"lr={lr:.2e}"
                )

                running_loss = 0.0

            if stop_training:
                break

        metrics = evaluate(
            model,
            val_loader,
            device,
        )

        print()
        print(
            f"validation loss: "
            f"{metrics['loss']:.4f}"
        )

        print(
            f"validation accuracy: "
            f"{metrics['accuracy']:.2%}"
        )

        checkpoint = {
            "model": model.state_dict(),
            "model_name": MODEL_NAME,
            "epoch": epoch + 1,
            "validation": metrics,
        }

        torch.save(
            checkpoint,
            CHECKPOINT_DIR / "last.pt",
        )

        if metrics["loss"] < best_val_loss:
            best_val_loss = metrics["loss"]

            torch.save(
                checkpoint,
                CHECKPOINT_DIR / "best.pt",
            )

            print("saved new best checkpoint")

        if stop_training:
            print(
                f"Sanity run complete at step {global_step}. "
                "Set MAX_TRAIN_STEPS = None for a full run."
            )
            break


if __name__ == "__main__":
    main()
