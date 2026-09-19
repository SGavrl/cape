# CAPE

CAPE is a Calibrated Assertion Probability Estimator. Given a context and a
natural-language assertion, it returns an estimated probability that the
assertion is supported by the context.

```python
from cape import CAPE

cape = CAPE.from_checkpoint("checkpoints/cape_mix_v1.pt")
probability = cape.judge(
    context="The customer says their card was charged twice.",
    assertion="This is a billing issue.",
)
print(probability)
```

CAPE also supports batched scoring with `judge_many()` and choosing among
named alternatives with `choose()`. See `examples/basic.py` for a complete
example.

## Installation

CAPE requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Data preparation

Generated datasets are written under `data/`, which is excluded from Git.

Prepare CAPE-Mix-v1:

```bash
python scripts/prepare_mix.py
```

Prepare the v1.1 datasets after CAPE-Mix-v1 has been generated:

```bash
python scripts/prepare_v1_1.py
```

The v1.1 preparation step combines CAPE-Mix-v1 with public entailment and
question-answering datasets and deterministic synthetic reasoning examples.
It writes `data/v1_1_train.jsonl`, `data/v1_1_validation.jsonl`, and
`data/v1_1_test.jsonl`.

Additional training examples can be merged from ordinary JSONL files:

```bash
python scripts/prepare_v1_1.py \
  --extra-train-data path/to/extra.jsonl
```

The option may be repeated. It can also be configured with the platform path
separator:

```bash
CAPE_EXTRA_TRAIN_DATA=path/to/extra.jsonl python scripts/prepare_v1_1.py
```

Each JSONL row must contain `context`, `assertion`, and a numeric `label`
between 0 and 1. The optional `source` field defaults to `supplemental`.

## Training

Training is configured with environment variables and requires CUDA:

```bash
CAPE_TRAIN_PATH=data/v1_1_train.jsonl \
CAPE_VAL_PATH=data/v1_1_validation.jsonl \
CAPE_CHECKPOINT_DIR=checkpoints/cape_v1_1 \
python scripts/train.py
```

Available settings include `CAPE_MODEL_NAME`, `CAPE_BATCH_SIZE`,
`CAPE_MAX_LENGTH`, `CAPE_EPOCHS`, and `CAPE_LEARNING_RATE`. Set
`CAPE_INIT_CHECKPOINT` to initialize a run from an existing compatible CAPE
checkpoint. Checkpoints are written beneath `checkpoints/` and excluded from
Git.

## Calibration and evaluation

Apply temperature calibration to a trained checkpoint:

```bash
python scripts/calibrate.py \
  --checkpoint checkpoints/cape_v1_1/best.pt \
  --validation data/v1_1_validation.jsonl \
  --output checkpoints/cape_v1_1.pt \
  --model-version cape-v1.1
```

Evaluate one or more checkpoints:

```bash
python scripts/evaluate.py \
  --checkpoints checkpoints/cape_v1_1.pt \
  --data data/v1_1_test.jsonl \
  --output-dir evaluation/cape_v1_1
```

Evaluation output is written beneath `evaluation/`, which is excluded from
Git. Frozen project results that are useful for comparison live in
`benchmarks/`.

## Command-line tools

Run an assertion directly or start the interactive interface:

```bash
python scripts/judge.py \
  --checkpoint checkpoints/cape_mix_v1.pt \
  --context "The deployment completed successfully." \
  --assertion "The deployment failed."
```

Run the frozen challenge, semantic probes, or throughput benchmark:

```bash
python scripts/challenge.py --checkpoint checkpoints/cape_mix_v1.pt
python scripts/run_probes.py --checkpoint checkpoints/cape_mix_v1.pt --strict
python scripts/benchmark.py --checkpoint checkpoints/cape_mix_v1.pt
```

The challenge cases in `scripts/challenge.py` are frozen. Reference results
are stored in `benchmarks/`.

## Tests

```bash
python -m pytest -q
```

## Repository layout

- `cape/`: model, inference SDK, dataset, and device utilities
- `scripts/`: data preparation, training, calibration, evaluation, and tools
- `examples/`: SDK examples
- `tests/`: unit tests
- `benchmarks/`: frozen benchmark and challenge results
- `artifacts/`: model metadata and hashes
- `probes/`: semantic probe definitions

## License

CAPE is available under the Apache License 2.0. See `LICENSE`.
