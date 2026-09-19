import torch
from types import SimpleNamespace

from scripts.benchmark import DEFAULT_COUNTS, benchmark_count


class TinyTokenizer:
    def __call__(self, text, **kwargs):
        del kwargs
        return {"input_ids": list(range(len(text.split()) + 2))}


class TinyCAPE:
    device = torch.device("cpu")
    architecture = "cross_encoder"
    tokenizer = TinyTokenizer()
    max_length = 32

    def judge_many(self, context, assertions, batch_size=64):
        del context, batch_size
        return [0.5] * len(assertions)

    def choose(
        self,
        context,
        choices,
        assertion_template,
        batch_size=64,
    ):
        del context, assertion_template, batch_size
        return SimpleNamespace(scores={choice: 0.5 for choice in choices})


def test_fanout_benchmark_smoke_and_required_counts():
    assert DEFAULT_COUNTS == [1, 2, 5, 10, 25, 50, 100, 250, 1000]
    result = benchmark_count(
        TinyCAPE(),
        "short context",
        assertion_count=5,
        warmup_runs=1,
        repeat_runs=2,
        batch_size=4,
    )
    assert result["assertions"] == 5
    assert result["p95_ms"] >= 0.0
    assert result["context_tokens"] > 0
    choice_result = benchmark_count(
        TinyCAPE(),
        "short context",
        assertion_count=2,
        warmup_runs=0,
        repeat_runs=1,
        batch_size=2,
        mode="choice",
    )
    assert choice_result["mode"] == "choice"
