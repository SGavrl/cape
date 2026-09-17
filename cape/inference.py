from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoTokenizer

from .cape_model import CAPEModel


@dataclass(frozen=True)
class ChoiceResult:
    choice: str
    probability: float
    scores: dict[str, float]


def _resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device

    if device != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


class CAPE:
    """
    Calibrated Assertion Probability Estimator.

    Estimates:
        P(assertion is true/applicable | context)

    CAPE-Mix-v1 is currently a cross-encoder. `judge_many()` batches
    many context/assertion pairs, but still re-encodes the context
    for every assertion. The public API is designed so a future CAPE
    architecture can optimize this without changing user code.
    """

    def __init__(
        self,
        model: CAPEModel,
        tokenizer,
        *,
        device: torch.device,
        temperature: float = 1.0,
        max_length: int = 256,
        checkpoint_path: str | None = None,
        metadata: dict | None = None,
    ):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = float(temperature)
        self.max_length = int(max_length)
        self.checkpoint_path = checkpoint_path
        self.metadata = metadata or {}

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "auto",
        max_length: int = 256,
    ) -> "CAPE":
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"CAPE checkpoint not found: {checkpoint_path}"
            )

        resolved_device = _resolve_device(device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        model_name = checkpoint.get("model_name")
        if not model_name:
            raise ValueError(
                "Checkpoint is missing required field 'model_name'."
            )

        model_state = checkpoint.get("model")
        if model_state is None:
            raise ValueError(
                "Checkpoint is missing required field 'model'."
            )

        temperature = float(
            checkpoint.get(
                "temperature",
                1.0,
            )
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        model = CAPEModel(
            model_name
        )

        model.load_state_dict(
            model_state
        )

        model = model.to(
            resolved_device
        )

        model.eval()

        if resolved_device.type == "cuda":
            torch.set_float32_matmul_precision(
                "high"
            )

        metadata = {
            key: value
            for key, value in checkpoint.items()
            if key != "model"
        }

        return cls(
            model,
            tokenizer,
            device=resolved_device,
            temperature=temperature,
            max_length=max_length,
            checkpoint_path=str(checkpoint_path),
            metadata=metadata,
        )

    def _tokenize(
        self,
        contexts: Sequence[str],
        assertions: Sequence[str],
    ):
        return self.tokenizer(
            list(contexts),
            list(assertions),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

    @torch.inference_mode()
    def judge(
        self,
        context: str,
        assertion: str,
    ) -> float:
        return self.judge_many(
            context,
            [assertion],
            batch_size=1,
        )[0]

    @torch.inference_mode()
    def judge_many(
        self,
        context: str,
        assertions: Sequence[str],
        *,
        batch_size: int = 64,
    ) -> list[float]:
        assertions = list(assertions)

        if not assertions:
            return []

        if batch_size < 1:
            raise ValueError(
                "batch_size must be at least 1"
            )

        probabilities: list[float] = []

        for start in range(
            0,
            len(assertions),
            batch_size,
        ):
            batch_assertions = assertions[
                start : start + batch_size
            ]

            batch_contexts = [
                context
            ] * len(batch_assertions)

            tokens = self._tokenize(
                batch_contexts,
                batch_assertions,
            )

            tokens = {
                key: value.to(
                    self.device,
                    non_blocking=(
                        self.device.type == "cuda"
                    ),
                )
                for key, value in tokens.items()
            }

            logits = self.model(
                **tokens
            )

            if not torch.isfinite(
                logits
            ).all():
                raise RuntimeError(
                    "CAPE produced non-finite logits."
                )

            batch_probabilities = torch.sigmoid(
                logits.float()
                / self.temperature
            )

            probabilities.extend(
                batch_probabilities
                .cpu()
                .tolist()
            )

        return [
            float(value)
            for value in probabilities
        ]

    def choose(
        self,
        context: str,
        choices: Sequence[str],
        *,
        assertion_template: str = (
            "The appropriate category is {choice}."
        ),
        batch_size: int = 64,
    ) -> ChoiceResult:
        choices = list(choices)

        if not choices:
            raise ValueError(
                "choices must not be empty"
            )

        assertions = [
            assertion_template.format(
                choice=choice
            )
            for choice in choices
        ]

        probabilities = self.judge_many(
            context,
            assertions,
            batch_size=batch_size,
        )

        scores = {
            choice: probability
            for choice, probability
            in zip(
                choices,
                probabilities,
                strict=True,
            )
        }

        best_choice = max(
            scores,
            key=scores.get,
        )

        return ChoiceResult(
            choice=best_choice,
            probability=scores[
                best_choice
            ],
            scores=scores,
        )

    def info(self) -> dict:
        return {
            "checkpoint": (
                self.checkpoint_path
            ),
            "model_name": (
                self.metadata.get(
                    "model_name"
                )
            ),
            "model_version": (
                self.metadata.get(
                    "model_version"
                )
            ),
            "epoch": (
                self.metadata.get(
                    "epoch"
                )
            ),
            "temperature": (
                self.temperature
            ),
            "device": str(
                self.device
            ),
            "max_length": (
                self.max_length
            ),
        }
