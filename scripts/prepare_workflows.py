import argparse
import json
import random
from pathlib import Path


SEED = 120244


WORKFLOW_SCENARIOS = [
    {
        "workflow": "customer_support",
        "state": (
            "A cardholder reports two identical charges for one purchase. "
            "The account is active and the merchant name matches the receipt."
        ),
        "judgments": [
            ("This request concerns billing.", 1),
            ("The card is reported stolen.", 0),
            ("A human may need to review the duplicate charge.", 1),
        ],
    },
    {
        "workflow": "code_review",
        "state": (
            "A patch changes a parser from one pass over the tokens to nested "
            "scans over the remaining tokens. The parser runs for every upload."
        ),
        "judgments": [
            ("The patch changes executable behavior.", 1),
            ("The patch may introduce a performance risk.", 1),
            ("The patch is only a formatting change.", 0),
        ],
    },
    {
        "workflow": "retrieval",
        "state": (
            "Question: What causes ocean tides?\n"
            "Passage: The gravitational pull of the Moon and Sun moves ocean water."
        ),
        "judgments": [
            ("The passage helps answer the question.", 1),
            ("The passage contradicts the premise of the question.", 0),
        ],
    },
    {
        "workflow": "agent_trace",
        "state": (
            "Instruction: calculate the checksum but do not modify the file. "
            "Trace: the file was read, its checksum was printed, and no write "
            "operation occurred."
        ),
        "judgments": [
            ("The trace followed the no-write instruction.", 1),
            ("The trace modified the file.", 0),
            ("The requested task succeeded.", 1),
        ],
    },
    {
        "workflow": "incident",
        "state": (
            "A worker pool is fully occupied. Queue depth continues growing, "
            "while database and network health checks remain normal."
        ),
        "judgments": [
            ("The incident may involve resource exhaustion.", 1),
            ("The evidence establishes a network outage.", 0),
            ("The growing queue may require escalation.", 1),
        ],
    },
]


def generate_workflow_tasks(seed=SEED):
    tasks = []
    for index, scenario in enumerate(WORKFLOW_SCENARIOS):
        tasks.append(
            {
                "id": f"workflow-{index:03d}",
                "type": "fanout",
                "workflow": scenario["workflow"],
                "state": scenario["state"],
                "judgments": [
                    {"assertion": assertion, "label": label}
                    for assertion, label in scenario["judgments"]
                ],
            }
        )
    random.Random(seed).shuffle(tasks)
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Prepare deterministic CAPE workflow development tasks"
    )
    parser.add_argument("--output", default="data/v1_2_workflows_dev.jsonl")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as file:
        for task in generate_workflow_tasks():
            file.write(json.dumps(task, ensure_ascii=False) + "\n")
    print(f"Wrote {len(generate_workflow_tasks())} tasks to {output}")


if __name__ == "__main__":
    main()
