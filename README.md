# CAPE

CAPE stands for **Calibrated Assertion Probability Estimator**.

The idea is simple:

Given some context and a natural-language assertion, estimate how likely the assertion is to be true.

Example:

```python
cape.judge(
    context="The customer says they were charged twice.",
    assertion="This is a billing issue."
)
```

```text
0.94
```

CAPE is an early experiment in building a small, fast, non-generative model for scoring arbitrary assertions with calibrated probabilities.

The longer-term goal is to use the same primitive for things like classification, routing, ranking, relevance, risk estimation, reward scoring, and other decision tasks without generating text.

The project is still very early and currently focused on getting the core model and evaluation pipeline working.

Work in progress.
