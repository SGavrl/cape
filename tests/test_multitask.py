import torch

from cape.losses import masked_cross_entropy, multitask_loss
from cape.schema import MISSING_LABEL


def test_masked_auxiliary_losses_ignore_missing_labels():
    logits = torch.randn(3, 2, requires_grad=True)
    labels = torch.tensor([MISSING_LABEL, MISSING_LABEL, MISSING_LABEL])
    loss, count = masked_cross_entropy(logits, labels)
    assert count == 0
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None


def test_multitask_loss_uses_only_available_auxiliary_targets():
    outputs = {
        "support": torch.tensor([0.2, -0.3], requires_grad=True),
        "nli": torch.randn(2, 3, requires_grad=True),
        "relevance": torch.randn(2, 2, requires_grad=True),
        "strength": torch.randn(2, 3, requires_grad=True),
    }
    batch = {
        "labels": torch.tensor([1.0, 0.0]),
        "nli_labels": torch.tensor([0, 2]),
        "relevance_labels": torch.tensor([MISSING_LABEL, MISSING_LABEL]),
        "strength_labels": torch.tensor([0, MISSING_LABEL]),
    }
    total, losses, counts = multitask_loss(outputs, batch)
    assert torch.isfinite(total)
    assert counts == {"support": 2, "nli": 2, "relevance": 0, "strength": 1}
    assert losses["relevance"].item() == 0.0
    total.backward()
