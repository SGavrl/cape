import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from cape.cape_model import CAPEModel


def load_model(
    checkpoint_path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
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

    model = model.to(device)
    model.eval()

    return model, tokenizer


@torch.inference_mode()
def judge(
    model,
    tokenizer,
    device,
    context,
    assertion,
):
    tokens = tokenizer(
        context,
        assertion,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    tokens = {
        key: value.to(device)
        for key, value in tokens.items()
    }

    logits = model(**tokens)

    return torch.sigmoid(
        logits.float()
    ).item()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best.pt",
    )

    parser.add_argument(
        "--probes",
        default="probes/semantic.jsonl",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
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

    model, tokenizer = load_model(
        args.checkpoint,
        device,
    )

    probes = []

    with Path(args.probes).open() as f:
        for line in f:
            probes.append(
                json.loads(line)
            )

    passed = 0

    print()
    print(
        f"{'result':<8}"
        f"{'prob':>9}  "
        f"name"
    )
    print("-" * 70)

    for probe in probes:
        probability = judge(
            model,
            tokenizer,
            device,
            probe["context"],
            probe["assertion"],
        )

        success = True

        if (
            "min_probability"
            in probe
        ):
            success &= (
                probability
                >= probe[
                    "min_probability"
                ]
            )

        if (
            "max_probability"
            in probe
        ):
            success &= (
                probability
                <= probe[
                    "max_probability"
                ]
            )

        if success:
            passed += 1

        status = (
            "PASS"
            if success
            else "FAIL"
        )

        print(
            f"{status:<8}"
            f"{probability:>8.2%}  "
            f"{probe['name']}"
        )

    total = len(probes)

    print()
    print(
        f"Passed {passed}/{total} "
        f"({passed / total:.1%})"
    )

    if (
        args.strict
        and passed != total
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
