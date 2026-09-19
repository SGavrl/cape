from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoTokenizer

from .calibration import calibrated_probabilities, calibration_from_checkpoint
from .cape_model import SHARED_CONTEXT
from .checkpoints import (
    build_model_from_checkpoint,
    checkpoint_backbone,
)


@dataclass(frozen=True)
class ChoiceResult:
    choice: str | None
    probability: float
    scores: dict[str, float]


@dataclass(frozen=True)
class EncodedContext:
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    checkpoint_path: str | None


def _resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CAPE:
    """Calibrated Assertion Probability Estimator.

    ``judge`` estimates the probability that an assertion is warranted by the
    supplied context, allowing ordinary background knowledge. Plausibility or
    topical association alone is not support.
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        device: torch.device,
        calibration: dict | None = None,
        max_length: int = 256,
        checkpoint_path: str | None = None,
        metadata: dict | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.calibration = calibration or {
            "method": "temperature",
            "temperature": 1.0,
        }
        self.max_length = int(max_length)
        self.checkpoint_path = checkpoint_path
        self.metadata = metadata or {}
        self.architecture = getattr(model, "architecture", "cross_encoder")

    @property
    def temperature(self) -> float:
        if self.calibration.get("method") == "temperature":
            return float(self.calibration.get("temperature", 1.0))
        return 1.0

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "auto",
        max_length: int | None = None,
    ) -> "CAPE":
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"CAPE checkpoint not found: {checkpoint_path}")

        resolved_device = _resolve_device(device)
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        model_name = checkpoint_backbone(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = build_model_from_checkpoint(checkpoint).to(resolved_device)
        model.eval()

        if resolved_device.type == "cuda":
            torch.set_float32_matmul_precision("high")

        metadata = {key: value for key, value in checkpoint.items() if key != "model"}
        return cls(
            model,
            tokenizer,
            device=resolved_device,
            calibration=calibration_from_checkpoint(checkpoint),
            max_length=(
                int(max_length)
                if max_length is not None
                else int(checkpoint.get("max_length", 256))
            ),
            checkpoint_path=str(checkpoint_path),
            metadata=metadata,
        )

    def _tokenize_pairs(
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

    def _to_device(self, tokens):
        return {
            key: value.to(
                self.device,
                non_blocking=self.device.type == "cuda",
            )
            for key, value in tokens.items()
        }

    def _probabilities(self, logits):
        if not torch.isfinite(logits).all():
            raise RuntimeError("CAPE produced non-finite logits")
        return calibrated_probabilities(logits, self.calibration)

    @torch.inference_mode()
    def judge(self, context: str, assertion: str) -> float:
        return self.judge_many(context, [assertion], batch_size=1)[0]

    @torch.inference_mode()
    def encode_context(self, context: str) -> EncodedContext:
        if self.architecture != SHARED_CONTEXT:
            raise RuntimeError(
                "encode_context is available only for shared-context checkpoints"
            )
        tokens = self._to_device(
            self.tokenizer(
                [context],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
        )
        states, mask = self.model.encode_context_tokens(
            tokens["input_ids"], tokens["attention_mask"]
        )
        return EncodedContext(states, mask, self.checkpoint_path)

    @torch.inference_mode()
    def judge_encoded(
        self,
        encoded: EncodedContext,
        assertions: Sequence[str],
        *,
        batch_size: int = 64,
    ) -> list[float]:
        if self.architecture != SHARED_CONTEXT:
            raise RuntimeError(
                "judge_encoded is available only for shared-context checkpoints"
            )
        if encoded.checkpoint_path != self.checkpoint_path:
            raise ValueError("encoded context belongs to a different checkpoint")
        assertions = list(assertions)
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        probabilities = []
        for start in range(0, len(assertions), batch_size):
            batch = assertions[start : start + batch_size]
            tokens = self._to_device(
                self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
            )
            assertion_states, _ = self.model.encode_assertion_tokens(
                tokens["input_ids"], tokens["attention_mask"]
            )
            logits = self.model.score_encoded(
                encoded.hidden_states,
                encoded.attention_mask,
                assertion_states,
            )
            probabilities.extend(self._probabilities(logits).cpu().tolist())
        return [float(value) for value in probabilities]

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
            raise ValueError("batch_size must be at least 1")
        if self.architecture == SHARED_CONTEXT:
            return self.judge_encoded(
                self.encode_context(context),
                assertions,
                batch_size=batch_size,
            )

        probabilities = []
        for start in range(0, len(assertions), batch_size):
            batch_assertions = assertions[start : start + batch_size]
            tokens = self._to_device(
                self._tokenize_pairs(
                    [context] * len(batch_assertions),
                    batch_assertions,
                )
            )
            probabilities.extend(
                self._probabilities(self.model(**tokens)).cpu().tolist()
            )
        return [float(value) for value in probabilities]

    def choose(
        self,
        context: str,
        choices: Sequence[str],
        *,
        assertion_template: str = "The appropriate category is {choice}.",
        batch_size: int = 64,
        minimum_probability: float | None = None,
    ) -> ChoiceResult:
        choices = list(choices)
        if not choices:
            raise ValueError("choices must not be empty")
        if minimum_probability is not None and not 0.0 <= minimum_probability <= 1.0:
            raise ValueError("minimum_probability must be in [0, 1]")
        assertions = [assertion_template.format(choice=choice) for choice in choices]
        probabilities = self.judge_many(
            context,
            assertions,
            batch_size=batch_size,
        )
        scores = dict(zip(choices, probabilities, strict=True))
        best_choice = max(scores, key=scores.get)
        probability = scores[best_choice]
        if minimum_probability is not None and probability < minimum_probability:
            best_choice = None
        return ChoiceResult(
            choice=best_choice,
            probability=probability,
            scores=scores,
        )

    def info(self) -> dict:
        return {
            "checkpoint": self.checkpoint_path,
            "cape_version": self.metadata.get("cape_version"),
            "architecture": self.architecture,
            "model_name": self.metadata.get(
                "backbone", self.metadata.get("model_name")
            ),
            "model_version": self.metadata.get("model_version"),
            "pooling": getattr(self.model, "pooling", "first_token"),
            "epoch": self.metadata.get("epoch"),
            "calibration": self.calibration,
            "temperature": self.temperature,
            "device": str(self.device),
            "max_length": self.max_length,
        }
