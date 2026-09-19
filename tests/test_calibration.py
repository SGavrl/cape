import torch

from cape.calibration import (
    calibrated_probabilities,
    calibration_from_checkpoint,
    fit_calibration,
)


def test_all_calibrators_produce_bounded_probabilities():
    logits = torch.tensor([-3.0, -0.5, 0.5, 3.0])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    for method in ("temperature", "platt", "beta"):
        parameters = fit_calibration(logits, labels, method=method, max_iter=20)
        probabilities = calibrated_probabilities(logits, parameters)
        assert torch.isfinite(probabilities).all()
        assert ((0.0 <= probabilities) & (probabilities <= 1.0)).all()


def test_new_and_legacy_calibration_metadata_deserialize():
    assert calibration_from_checkpoint({"temperature": 2.5}) == {
        "method": "temperature",
        "temperature": 2.5,
    }
    parameters = {"method": "platt", "scale": 0.8, "bias": -0.1}
    assert calibration_from_checkpoint(
        {"calibration_parameters": parameters}
    ) == parameters
