from __future__ import annotations

import torch.nn.functional as F

from .schema import MISSING_LABEL


DEFAULT_AUXILIARY_WEIGHTS = {
    "nli": 0.25,
    "relevance": 0.15,
    "strength": 0.10,
}


def masked_cross_entropy(logits, labels):
    mask = labels != MISSING_LABEL
    if not mask.any():
        return logits.sum() * 0.0, 0
    return F.cross_entropy(logits[mask], labels[mask]), int(mask.sum().item())


def multitask_loss(outputs, batch, auxiliary_weights=None):
    weights = dict(DEFAULT_AUXILIARY_WEIGHTS)
    if auxiliary_weights:
        weights.update(auxiliary_weights)

    support = F.binary_cross_entropy_with_logits(
        outputs["support"],
        batch["labels"],
    )
    total = support
    losses = {"support": support}
    counts = {"support": int(batch["labels"].numel())}

    for head in ("nli", "relevance", "strength"):
        loss, count = masked_cross_entropy(
            outputs[head],
            batch[f"{head}_labels"],
        )
        losses[head] = loss
        counts[head] = count
        total = total + weights[head] * loss

    return total, losses, counts
