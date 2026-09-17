import torch

from cape.inference import (
    ChoiceResult,
    _resolve_device,
)


def test_explicit_cpu_device():
    device = _resolve_device("cpu")

    assert device == torch.device(
        "cpu"
    )


def test_choice_result():
    result = ChoiceResult(
        choice="billing",
        probability=0.9,
        scores={
            "billing": 0.9,
            "networking": 0.1,
        },
    )

    assert result.choice == "billing"
    assert result.probability == 0.9
    assert result.scores[
        "networking"
    ] == 0.1
