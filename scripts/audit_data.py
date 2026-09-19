import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

try:
    from scripts.challenge import CASES as FROZEN_CHALLENGE_CASES
except ModuleNotFoundError:
    from challenge import CASES as FROZEN_CHALLENGE_CASES


def load_rows(path):
    with Path(path).open() as file:
        return [json.loads(line) for line in file if line.strip()]


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def audit_split(path, rows):
    by_source = defaultdict(list)
    for row in rows:
        by_source[row.get("source", "unknown")].append(row)
    ids = [row.get("id") for row in rows if row.get("id") is not None]
    result = {
        "path": str(path),
        "examples": len(rows),
        "missing_ids": len(rows) - len(ids),
        "duplicate_ids": len(ids) - len(set(ids)),
        "duplicate_pairs": len(rows)
        - len({(row["context"], row["assertion"]) for row in rows}),
        "sources": {},
    }
    for source, examples in sorted(by_source.items()):
        targets = [float(row.get("target", row.get("label"))) for row in examples]
        assertions = Counter(row["assertion"] for row in examples)
        result["sources"][source] = {
            "examples": len(examples),
            "positive_rate": sum(targets) / len(targets),
            "mean_context_tokens": sum(len(words(row["context"])) for row in examples)
            / len(examples),
            "mean_assertion_tokens": sum(len(words(row["assertion"])) for row in examples)
            / len(examples),
            "unique_assertions": len(assertions),
            "most_common_assertions": assertions.most_common(5),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="Audit CAPE dataset shortcuts")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--output")
    args = parser.parse_args()
    splits = {path: load_rows(path) for path in args.paths}
    report = {
        "splits": [audit_split(path, rows) for path, rows in splits.items()],
        "overlap": {},
    }
    paths = list(splits)
    for left_index, left in enumerate(paths):
        for right in paths[left_index + 1 :]:
            left_pairs = {(row["context"], row["assertion"]) for row in splits[left]}
            right_pairs = {(row["context"], row["assertion"]) for row in splits[right]}
            report["overlap"][f"{left} :: {right}"] = len(
                left_pairs & right_pairs
            )
    challenge = {
        (context, assertion)
        for cases in FROZEN_CHALLENGE_CASES.values()
        for context, assertion in cases
    }
    report["frozen_challenge_exact_overlap"] = {
        path: len(
            {(row["context"], row["assertion"]) for row in rows} & challenge
        )
        for path, rows in splits.items()
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
