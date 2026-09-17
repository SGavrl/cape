import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from cape import CAPE


DEFAULT_CHECKPOINT = "checkpoints/cape_mix_v1.pt"

DEFAULT_COUNTS = [
    1,
    10,
    100,
    1000,
]

DEFAULT_WARMUP_RUNS = 2
DEFAULT_REPEAT_RUNS = 5


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()

    elif (
        device.type == "mps"
        and hasattr(torch, "mps")
    ):
        torch.mps.synchronize()


def reset_peak_memory(device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def get_peak_memory_mb(device: torch.device):
    if device.type != "cuda":
        return None

    return (
        torch.cuda.max_memory_allocated()
        / (1024 ** 2)
    )


def build_assertions(count: int):
    base_assertions = [
        "This change may affect performance.",
        "This change may create concurrency risk.",
        "This change is related to memory allocation.",
        "This change may affect reliability.",
        "This change may introduce a regression.",
        "This change touches a hot execution path.",
        "This change may increase CPU usage.",
        "This change may increase memory usage.",
        "This change may require additional testing.",
        "This change may need manual review.",
    ]

    assertions = []

    while len(assertions) < count:
        index = len(assertions)

        base = base_assertions[
            index % len(base_assertions)
        ]

        assertions.append(
            f"{base} "
            f"[benchmark assertion {index + 1}]"
        )

    return assertions


def benchmark_count(
    cape: CAPE,
    context: str,
    assertion_count: int,
    warmup_runs: int,
    repeat_runs: int,
    batch_size: int,
):
    assertions = build_assertions(
        assertion_count
    )

    device = cape.device

    # Warmup
    for _ in range(warmup_runs):
        cape.judge_many(
            context,
            assertions,
            batch_size=batch_size,
        )

        synchronize(device)

    durations = []

    reset_peak_memory(
        device
    )

    for _ in range(repeat_runs):
        synchronize(
            device
        )

        start = time.perf_counter()

        results = cape.judge_many(
            context,
            assertions,
            batch_size=batch_size,
        )

        synchronize(
            device
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        if len(results) != assertion_count:
            raise RuntimeError(
                "CAPE returned the wrong "
                "number of results."
            )

        durations.append(
            elapsed
        )

    median_seconds = (
        statistics.median(
            durations
        )
    )

    mean_seconds = (
        statistics.mean(
            durations
        )
    )

    min_seconds = min(
        durations
    )

    max_seconds = max(
        durations
    )

    assertions_per_second = (
        assertion_count
        / median_seconds
    )

    milliseconds_per_assertion = (
        median_seconds
        * 1000
        / assertion_count
    )

    peak_memory_mb = (
        get_peak_memory_mb(
            device
        )
    )

    return {
        "assertions": (
            assertion_count
        ),
        "runs": (
            repeat_runs
        ),
        "batch_size": (
            batch_size
        ),
        "median_ms": (
            median_seconds
            * 1000
        ),
        "mean_ms": (
            mean_seconds
            * 1000
        ),
        "min_ms": (
            min_seconds
            * 1000
        ),
        "max_ms": (
            max_seconds
            * 1000
        ),
        "ms_per_assertion": (
            milliseconds_per_assertion
        ),
        "assertions_per_second": (
            assertions_per_second
        ),
        "peak_memory_mb": (
            peak_memory_mb
        ),
    }


def print_result(result):
    assertions = result[
        "assertions"
    ]

    peak_memory = result[
        "peak_memory_mb"
    ]

    if peak_memory is None:
        peak_memory_text = "n/a"
    else:
        peak_memory_text = (
            f"{peak_memory:.1f} MB"
        )

    print(
        f"{assertions:>5} assertions | "
        f"{result['median_ms']:>10.2f} ms total | "
        f"{result['ms_per_assertion']:>9.3f} ms/assertion | "
        f"{result['assertions_per_second']:>10.2f} assertions/s | "
        f"peak mem: {peak_memory_text}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark CAPE inference throughput."
        )
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--counts",
        nargs="+",
        type=int,
        default=DEFAULT_COUNTS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=DEFAULT_WARMUP_RUNS,
    )

    parser.add_argument(
        "--repeat-runs",
        type=int,
        default=DEFAULT_REPEAT_RUNS,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional JSON output path."
        ),
    )

    args = parser.parse_args()

    for count in args.counts:
        if count < 1:
            raise ValueError(
                "All assertion counts must "
                "be at least 1."
            )

    if args.batch_size < 1:
        raise ValueError(
            "batch size must be at least 1"
        )

    context = (
        "The patch introduces a global mutex "
        "around a frequently called allocator "
        "inside a hot execution path. "
        "The code is executed many times per second."
    )

    cape = CAPE.from_checkpoint(
        args.checkpoint,
        device=args.device,
    )

    print()
    print("CAPE benchmark")
    print("=" * 90)

    info = cape.info()

    print(
        f"checkpoint:   "
        f"{info['checkpoint']}"
    )

    print(
        f"model:        "
        f"{info['model_name']}"
    )

    print(
        f"version:      "
        f"{info['model_version']}"
    )

    print(
        f"device:       "
        f"{info['device']}"
    )

    print(
        f"temperature:  "
        f"{info['temperature']:.6f}"
    )

    print(
        f"batch size:   "
        f"{args.batch_size}"
    )

    print(
        f"warmup runs:  "
        f"{args.warmup_runs}"
    )

    print(
        f"repeat runs:  "
        f"{args.repeat_runs}"
    )

    print()
    print("-" * 90)

    results = []

    for count in args.counts:
        result = benchmark_count(
            cape=cape,
            context=context,
            assertion_count=count,
            warmup_runs=args.warmup_runs,
            repeat_runs=args.repeat_runs,
            batch_size=args.batch_size,
        )

        results.append(
            result
        )

        print_result(
            result
        )

    print("-" * 90)

    output = {
        "model": info,
        "benchmark": {
            "context": context,
            "batch_size": (
                args.batch_size
            ),
            "warmup_runs": (
                args.warmup_runs
            ),
            "repeat_runs": (
                args.repeat_runs
            ),
            "results": results,
        },
    }

    if args.output is not None:
        output_path = Path(
            args.output
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w"
        ) as f:
            json.dump(
                output,
                f,
                indent=2,
            )

        print()
        print(
            f"Saved benchmark to "
            f"{output_path}"
        )


if __name__ == "__main__":
    main()
