from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def calibration_from_checkpoint(checkpoint: dict) -> dict:
    parameters = checkpoint.get("calibration_parameters")
    if parameters:
        return dict(parameters)
    return {
        "method": "temperature",
        "temperature": float(checkpoint.get("temperature", 1.0)),
    }


def calibrated_probabilities(logits: torch.Tensor, parameters: dict):
    method = parameters.get("method", "temperature")
    logits = logits.float()
    if method == "temperature":
        temperature = parameters.get("temperature", 1.0)
        if not torch.is_tensor(temperature) and float(temperature) <= 0:
            raise ValueError("temperature must be positive")
        return torch.sigmoid(logits / temperature)
    if method == "platt":
        return torch.sigmoid(
            logits * parameters["scale"]
            + parameters["bias"]
        )
    if method == "beta":
        probability = torch.sigmoid(logits).clamp(1e-7, 1.0 - 1e-7)
        calibrated_logit = (
            parameters["a"] * torch.log(probability)
            - parameters["b"] * torch.log1p(-probability)
            + parameters["bias"]
        )
        return torch.sigmoid(calibrated_logit)
    raise ValueError(f"unsupported calibration method: {method!r}")


def fit_calibration(logits, labels, method="temperature", max_iter=100):
    logits = logits.detach().float()
    labels = labels.detach().float()
    if method == "temperature":
        raw = torch.zeros((), requires_grad=True)
        parameters = [raw]

        def values():
            return {"method": method, "temperature": raw.exp()}

    elif method == "platt":
        scale = torch.ones((), requires_grad=True)
        bias = torch.zeros((), requires_grad=True)
        parameters = [scale, bias]

        def values():
            return {"method": method, "scale": scale, "bias": bias}

    elif method == "beta":
        raw_a = torch.zeros((), requires_grad=True)
        raw_b = torch.zeros((), requires_grad=True)
        bias = torch.zeros((), requires_grad=True)
        parameters = [raw_a, raw_b, bias]

        def values():
            return {
                "method": method,
                "a": raw_a.exp(),
                "b": raw_b.exp(),
                "bias": bias,
            }

    else:
        raise ValueError(f"unsupported calibration method: {method!r}")

    optimizer = torch.optim.LBFGS(
        parameters,
        lr=0.1,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        probability = calibrated_probabilities(logits, values())
        loss = F.binary_cross_entropy(probability, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    result = values()
    return {
        key: value.detach().item() if torch.is_tensor(value) else value
        for key, value in result.items()
    }


def validate_calibration(parameters: dict):
    values = [value for key, value in parameters.items() if key != "method"]
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        raise ValueError("calibration parameters must be finite")
    if parameters.get("method") == "temperature" and parameters.get(
        "temperature", 0
    ) <= 0:
        raise ValueError("temperature must be positive")
