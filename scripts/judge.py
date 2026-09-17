import argparse

import torch
from transformers import AutoTokenizer

from cape.cape_model import CAPEModel


DEFAULT_CHECKPOINT = "checkpoints/cape_mix_v1.pt"


def load_cape(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model_name = checkpoint["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = CAPEModel(
        model_name
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model = model.to(device)
    model.eval()

    temperature = float(
        checkpoint.get(
            "temperature",
            1.0,
        )
    )

    return model, tokenizer, checkpoint, temperature


@torch.inference_mode()
def judge(
    model,
    tokenizer,
    device,
    context: str,
    assertion: str,
    temperature: float = 1.0,
):
    inputs = tokenizer(
        context,
        assertion,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    logits = model(**inputs)

    probability = torch.sigmoid(
        logits.float()
        / temperature
    ).item()

    return probability


def main():
    parser = argparse.ArgumentParser(
        description="Judge an assertion with CAPE."
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--context",
        default=None,
    )

    parser.add_argument(
        "--assertion",
        default=None,
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

    print(f"Device: {device}")

    model, tokenizer, checkpoint, temperature = load_cape(
        args.checkpoint,
        device,
    )

    print(
        f"Loaded CAPE checkpoint "
        f"from epoch {checkpoint.get('epoch', '?')}"
    )

    print(
        f"Temperature: {temperature:.6f}"
    )

    context = args.context
    assertion = args.assertion

    if context is not None and assertion is not None:
        probability = judge(
            model,
            tokenizer,
            device,
            context,
            assertion,
            temperature,
        )

        print()
        print(f"Context:   {context}")
        print(f"Assertion: {assertion}")
        print()
        print(
            f"CAPE: {probability:.4f} "
            f"({probability:.2%})"
        )

        return

    print()
    print("Interactive CAPE")
    print("Press Ctrl+C to exit.")
    print()

    while True:
        try:
            context = input("Context: ").strip()

            if not context:
                continue

            assertion = input(
                "Assertion: "
            ).strip()

            if not assertion:
                continue

            probability = judge(
                model,
                tokenizer,
                device,
                context,
                assertion,
                temperature,
            )

            print()
            print(
                f"CAPE: {probability:.4f} "
                f"({probability:.2%})"
            )
            print()

        except (KeyboardInterrupt, EOFError):
            print()
            break


if __name__ == "__main__":
    main()

