import argparse
import json
from pathlib import Path

from cape import CAPE
from cape.system_one import CAPEAdapter, evaluate_tasks


def load_tasks(path):
    with Path(path).open() as file:
        return [json.loads(line) for line in file if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CAPE with the generic System-One task schema"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cape = CAPE.from_checkpoint(args.checkpoint, device=args.device)
    report = evaluate_tasks(CAPEAdapter(cape), load_tasks(args.tasks))
    print(json.dumps(report, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
