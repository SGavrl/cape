import argparse
import json
from pathlib import Path


def robustness_rows():
    groups = [
        (
            "active_passive",
            "The curator approved the exhibit before noon.",
            [
                ("The curator approved the exhibit.", 1),
                ("The exhibit was approved by the curator.", 1),
            ],
            "equivalent",
        ),
        (
            "temporal_inversion",
            "Cataloguing occurred prior to digitization.",
            [
                ("Cataloguing happened before digitization.", 1),
                ("Digitization happened after cataloguing.", 1),
            ],
            "equivalent",
        ),
        (
            "synonym",
            "The committee postponed the hearing.",
            [
                ("The hearing was delayed.", 1),
                ("The hearing was put off until later.", 1),
            ],
            "equivalent",
        ),
        (
            "binary_complement",
            "The hatch is closed.",
            [
                ("The hatch is closed.", 1),
                ("The hatch is not closed.", 0),
            ],
            "complement",
        ),
        (
            "event_complement",
            "The ballot was counted.",
            [
                ("The ballot was counted.", 1),
                ("The ballot was not counted.", 0),
            ],
            "complement",
        ),
    ]
    rows = []
    for group, context, assertions, relation in groups:
        for index, (assertion, target) in enumerate(assertions):
            rows.append(
                {
                    "id": f"robustness-{group}-{index}",
                    "context": context,
                    "assertion": assertion,
                    "target": float(target),
                    "source": "robustness_dev",
                    "capability": "paraphrase_invariance",
                    "robustness_group": group,
                    "robustness_relation": relation,
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/v1_2_robustness_dev.jsonl")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as file:
        for row in robustness_rows():
            file.write(json.dumps(row) + "\n")
    print(f"Wrote {len(robustness_rows())} rows to {output}")


if __name__ == "__main__":
    main()
