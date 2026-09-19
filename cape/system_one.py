from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Protocol, Sequence

from .metrics import metrics_from_arrays


class DecisionAdapter(Protocol):
    name: str

    def judge(self, state: str, assertion: str) -> float: ...

    def judge_many(self, state: str, assertions: Sequence[str]) -> list[float]: ...

    def choose(
        self,
        state: str,
        choices: Sequence[str],
        assertion_template: str,
        minimum_probability: float | None,
    ) -> tuple[str | None, dict[str, float]]: ...

    def score(self, state: str, levels: Sequence[str]) -> int: ...


@dataclass
class CAPEAdapter:
    cape: object
    name: str = "cape"

    def judge(self, state, assertion):
        return self.cape.judge(state, assertion)

    def judge_many(self, state, assertions):
        return self.cape.judge_many(state, assertions)

    def choose(
        self,
        state,
        choices,
        assertion_template,
        minimum_probability=None,
    ):
        result = self.cape.choose(
            state,
            choices,
            assertion_template=assertion_template,
            minimum_probability=minimum_probability,
        )
        return result.choice, result.scores

    def score(self, state, levels):
        del state, levels
        raise NotImplementedError(
            "CAPE has no validated ordinal score objective; score tasks are "
            "reported as unsupported"
        )


def _percentile(values, amount):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * amount
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def evaluate_tasks(adapter: DecisionAdapter, tasks):
    boolean_probabilities = []
    boolean_labels = []
    choice_correct = 0
    choice_total = 0
    score_absolute_error = []
    malformed = 0
    unsupported = 0
    latencies = []

    for task in tasks:
        started = time.perf_counter()
        try:
            task_type = task["type"]
            if task_type == "boolean":
                probability = float(
                    adapter.judge(task["state"], task["assertion"])
                )
                if not 0.0 <= probability <= 1.0:
                    raise ValueError("boolean probability outside [0, 1]")
                boolean_probabilities.append(probability)
                boolean_labels.append(int(task["label"]))
            elif task_type == "fanout":
                assertions = [item["assertion"] for item in task["judgments"]]
                probabilities = adapter.judge_many(task["state"], assertions)
                if len(probabilities) != len(assertions):
                    raise ValueError("fanout result count mismatch")
                for probability, item in zip(
                    probabilities, task["judgments"], strict=True
                ):
                    probability = float(probability)
                    if not 0.0 <= probability <= 1.0:
                        raise ValueError("fanout probability outside [0, 1]")
                    boolean_probabilities.append(probability)
                    boolean_labels.append(int(item["label"]))
            elif task_type == "choice":
                choice, scores = adapter.choose(
                    task["state"],
                    task["choices"],
                    task.get(
                        "assertion_template",
                        "The appropriate category is {choice}.",
                    ),
                    task.get("minimum_probability"),
                )
                if set(scores) != set(task["choices"]):
                    raise ValueError("choice score keys do not match choices")
                choice_total += 1
                choice_correct += choice == task.get("answer")
            elif task_type == "score":
                prediction = adapter.score(task["state"], task["levels"])
                if not isinstance(prediction, int) or not 0 <= prediction < len(
                    task["levels"]
                ):
                    raise ValueError("ordered score is not a valid level index")
                score_absolute_error.append(abs(prediction - int(task["answer_index"])))
            else:
                raise ValueError(f"unknown task type: {task_type!r}")
        except NotImplementedError:
            unsupported += 1
        except (KeyError, TypeError, ValueError):
            malformed += 1
        finally:
            latencies.append(time.perf_counter() - started)

    return {
        "adapter": adapter.name,
        "tasks": len(tasks),
        "boolean": (
            metrics_from_arrays(boolean_probabilities, boolean_labels)
            if boolean_labels
            else None
        ),
        "choice": {
            "examples": choice_total,
            "accuracy": choice_correct / choice_total if choice_total else None,
        },
        "ordered_score": {
            "examples": len(score_absolute_error),
            "mean_absolute_level_error": (
                statistics.mean(score_absolute_error)
                if score_absolute_error
                else None
            ),
        },
        "unsupported": unsupported,
        "malformed": malformed,
        "externally_supplied_cost": getattr(adapter, "cost", None),
        "latency": {
            "median_ms": statistics.median(latencies) * 1000 if latencies else None,
            "p95_ms": _percentile(latencies, 0.95) * 1000 if latencies else None,
        },
    }
