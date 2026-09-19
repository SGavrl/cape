import hashlib

import pytest
import torch

from scripts.train import (
    create_optimizer,
    load_initial_weights,
    validate_checkpoint_paths,
)


def test_init_checkpoint_loads_only_weights_and_is_unchanged(tmp_path):
    source = torch.nn.Linear(3, 1)
    with torch.no_grad():
        source.weight.fill_(0.25)
        source.bias.fill_(0.75)

    checkpoint_path = tmp_path / "init.pt"
    torch.save(
        {
            "model": source.state_dict(),
            "model_name": "test-model",
            "temperature": 9.0,
            "optimizer": {"old": "state"},
            "global_step": 999,
        },
        checkpoint_path,
    )
    before = hashlib.sha256(checkpoint_path.read_bytes()).digest()

    target = torch.nn.Linear(3, 1)
    load_initial_weights(
        target,
        checkpoint_path,
        expected_model_name="test-model",
    )
    optimizer = create_optimizer(target)

    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)
    assert optimizer.state == {}
    assert hashlib.sha256(checkpoint_path.read_bytes()).digest() == before


def test_init_checkpoint_cannot_be_training_output(tmp_path):
    checkpoint_dir = tmp_path / "run"
    with pytest.raises(ValueError, match="must not"):
        validate_checkpoint_paths(checkpoint_dir / "last.pt", checkpoint_dir)
