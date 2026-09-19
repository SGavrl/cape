from types import SimpleNamespace

import torch

import cape.cape_model as model_module
import cape.inference as inference_module
from cape import CAPE
from cape.cape_model import (
    CAPEModel,
    CROSS_ENCODER,
    MULTITASK_CROSS_ENCODER,
    SharedContextCAPEModel,
    build_model,
)


class DummyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = torch.nn.Embedding(32, 8)

    def forward(self, input_ids, attention_mask):
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class DummyTokenizer:
    @classmethod
    def from_pretrained(cls, name):
        del name
        return cls()

    def __call__(
        self,
        first,
        second=None,
        *,
        return_tensors=None,
        padding=False,
        truncation=False,
        max_length=None,
    ):
        del padding, truncation, max_length
        if isinstance(first, str):
            first = [first]
        if second is not None and isinstance(second, str):
            second = [second]
        rows = []
        for index, text in enumerate(first):
            length = 3 + len(text) % 3
            if second is not None:
                length += len(second[index]) % 2
            rows.append([1 + (position % 20) for position in range(length)])
        width = max(map(len, rows))
        ids = [row + [0] * (width - len(row)) for row in rows]
        masks = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(ids),
                "attention_mask": torch.tensor(masks),
            }
        return {"input_ids": ids[0] if len(ids) == 1 else ids}


def patch_backbone(monkeypatch):
    monkeypatch.setattr(
        model_module.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: DummyEncoder(),
    )
    monkeypatch.setattr(inference_module, "AutoTokenizer", DummyTokenizer)


def test_old_checkpoint_without_architecture_still_loads(monkeypatch, tmp_path):
    patch_backbone(monkeypatch)
    model = CAPEModel("dummy", dropout=0.0)
    path = tmp_path / "v1.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_name": "dummy",
            "temperature": 2.0,
        },
        path,
    )
    cape = CAPE.from_checkpoint(path, device="cpu")
    assert cape.architecture == CROSS_ENCODER
    assert cape.temperature == 2.0
    assert 0.0 <= cape.judge("context", "assertion") <= 1.0


def test_multitask_checkpoint_loads_support_api(monkeypatch, tmp_path):
    patch_backbone(monkeypatch)
    model = build_model("dummy", MULTITASK_CROSS_ENCODER, dropout=0.0)
    path = tmp_path / "v1_2.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "architecture": MULTITASK_CROSS_ENCODER,
            "backbone": "dummy",
            "calibration_parameters": {
                "method": "platt",
                "scale": 1.0,
                "bias": 0.0,
            },
        },
        path,
    )
    cape = CAPE.from_checkpoint(path, device="cpu")
    assert cape.architecture == MULTITASK_CROSS_ENCODER
    assert len(cape.judge_many("context", ["one", "two"])) == 2


def test_shared_context_cached_and_automatic_paths_match(monkeypatch):
    patch_backbone(monkeypatch)
    model = SharedContextCAPEModel("dummy", dropout=0.0, attention_heads=2)
    model.eval()
    cape = CAPE(
        model,
        DummyTokenizer(),
        device=torch.device("cpu"),
        checkpoint_path="shared.pt",
    )
    assertions = ["first assertion", "second assertion", "third assertion"]
    encoded = cape.encode_context("one reusable context")
    cached = cape.judge_encoded(encoded, assertions, batch_size=2)
    automatic = cape.judge_many("one reusable context", assertions, batch_size=2)
    repeated = [cape.judge("one reusable context", assertion) for assertion in assertions]
    assert torch.allclose(torch.tensor(cached), torch.tensor(automatic), atol=1e-6)
    assert torch.allclose(torch.tensor(cached), torch.tensor(repeated), atol=1e-6)


def test_choice_can_return_none_without_normalizing_scores(monkeypatch):
    patch_backbone(monkeypatch)
    model = CAPEModel("dummy", dropout=0.0)
    model.eval()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.fill_(-4.0)
    cape = CAPE(model, DummyTokenizer(), device=torch.device("cpu"))
    result = cape.choose(
        "context",
        ["alpha", "beta"],
        minimum_probability=0.5,
    )
    assert result.choice is None
    assert set(result.scores) == {"alpha", "beta"}
    assert sum(result.scores.values()) != 1.0
