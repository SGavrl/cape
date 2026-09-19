from __future__ import annotations

from .cape_model import CROSS_ENCODER, SHARED_CONTEXT, build_model


def checkpoint_architecture(checkpoint: dict) -> str:
    return checkpoint.get("architecture", CROSS_ENCODER)


def checkpoint_backbone(checkpoint: dict) -> str:
    model_name = checkpoint.get("backbone", checkpoint.get("model_name"))
    if not model_name:
        raise ValueError("checkpoint is missing backbone/model_name")
    return model_name


def build_model_from_checkpoint(checkpoint: dict):
    model = build_model(
        checkpoint_backbone(checkpoint),
        checkpoint_architecture(checkpoint),
        pooling=checkpoint.get("pooling", "first_token"),
    )
    model_state = checkpoint.get("model")
    if model_state is None:
        raise ValueError("checkpoint is missing model")
    model.load_state_dict(model_state)
    return model


def initialize_from_checkpoint(model, checkpoint: dict):
    """Load all compatible parameters and report intentionally new keys."""
    source = checkpoint.get("model")
    if source is None:
        raise ValueError("checkpoint is missing model")
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    result = model.load_state_dict(compatible, strict=False)
    if not compatible:
        raise ValueError("checkpoint has no parameters compatible with model")
    return {
        "loaded": sorted(compatible),
        "missing": sorted(result.missing_keys),
        "unexpected": sorted(result.unexpected_keys),
    }


def supports_context_cache(checkpoint: dict) -> bool:
    return checkpoint_architecture(checkpoint) == SHARED_CONTEXT
