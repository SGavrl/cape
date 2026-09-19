import argparse
import json
from pathlib import Path

from cape import CAPE
from cape.metrics import metrics_from_arrays
from cape.robustness import robustness_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/v1_2_robustness_dev.jsonl")
    parser.add_argument("--output")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    with Path(args.data).open() as file:
        rows = [json.loads(line) for line in file if line.strip()]
    cape = CAPE.from_checkpoint(args.checkpoint, device=args.device)
    probabilities = [
        cape.judge(row["context"], row["assertion"]) for row in rows
    ]
    report = {
        "checkpoint": args.checkpoint,
        "classification": metrics_from_arrays(
            probabilities, [row["target"] for row in rows]
        ),
        "robustness": robustness_metrics(rows, probabilities),
    }
    print(json.dumps(report, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
