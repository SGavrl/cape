from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


CROSS_ENCODER = "cross_encoder"
MULTITASK_CROSS_ENCODER = "cross_encoder_multitask"
SHARED_CONTEXT = "shared_context_multitask"
SUPPORTED_ARCHITECTURES = {
    CROSS_ENCODER,
    MULTITASK_CROSS_ENCODER,
    SHARED_CONTEXT,
}


class CAPEModel(nn.Module):
    """The CAPE v1/v1.1 cross-encoder.

    Parameter names are intentionally unchanged so released checkpoints keep
    loading exactly as before.
    """

    architecture = CROSS_ENCODER

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        dropout: float = 0.1,
        pooling: str = "first_token",
    ):
        super().__init__()
        if pooling not in {"first_token", "mean"}:
            raise ValueError("pooling must be 'first_token' or 'mean'")
        self.pooling = pooling
        self.encoder = AutoModel.from_pretrained(model_name).float()
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def encode_pair(self, input_ids, attention_mask, **kwargs):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        if self.pooling == "first_token":
            representation = outputs.last_hidden_state[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).to(outputs.last_hidden_state.dtype)
            representation = (outputs.last_hidden_state * mask).sum(dim=1)
            representation = representation / mask.sum(dim=1).clamp_min(1.0)
        return self.dropout(representation)

    def forward(self, input_ids, attention_mask, **kwargs):
        representation = self.encode_pair(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        return self.head(representation).squeeze(-1)


class MultiTaskCAPEModel(CAPEModel):
    """Cross-encoder with masked auxiliary semantic heads."""

    architecture = MULTITASK_CROSS_ENCODER

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        dropout: float = 0.1,
        pooling: str = "first_token",
    ):
        super().__init__(
            model_name=model_name,
            dropout=dropout,
            pooling=pooling,
        )
        hidden_size = self.encoder.config.hidden_size
        self.nli_head = nn.Linear(hidden_size, 3)
        self.relevance_head = nn.Linear(hidden_size, 2)
        self.strength_head = nn.Linear(hidden_size, 3)

    def forward(
        self,
        input_ids,
        attention_mask,
        *,
        return_auxiliary: bool = False,
        **kwargs,
    ):
        representation = self.encode_pair(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        support_logits = self.head(representation).squeeze(-1)
        if not return_auxiliary:
            return support_logits
        return {
            "support": support_logits,
            "nli": self.nli_head(representation),
            "relevance": self.relevance_head(representation),
            "strength": self.strength_head(representation),
        }


class SharedContextCAPEModel(nn.Module):
    """Experimental encoder with reusable context representations.

    Context and assertions use one shared backbone. Assertion summaries query
    cached context token states through a lightweight cross-attention layer.
    This remains an experiment until semantic and fan-out measurements justify
    selecting it over the cross-encoder.
    """

    architecture = SHARED_CONTEXT

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        dropout: float = 0.1,
        attention_heads: int = 8,
        pooling: str = "first_token",
    ):
        super().__init__()
        if pooling not in {"first_token", "mean"}:
            raise ValueError("pooling must be 'first_token' or 'mean'")
        self.pooling = pooling
        self.encoder = AutoModel.from_pretrained(model_name).float()
        hidden_size = self.encoder.config.hidden_size
        if hidden_size % attention_heads:
            raise ValueError(
                "hidden size must be divisible by shared-context attention heads"
            )
        self.cross_attention = nn.MultiheadAttention(
            hidden_size,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.interaction = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )
        self.head = nn.Linear(hidden_size, 1)
        self.nli_head = nn.Linear(hidden_size, 3)
        self.relevance_head = nn.Linear(hidden_size, 2)
        self.strength_head = nn.Linear(hidden_size, 3)

    def encode_context_tokens(self, input_ids, attention_mask):
        states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        return states, attention_mask

    def encode_assertion_tokens(self, input_ids, attention_mask):
        states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        if self.pooling == "first_token":
            summary = states[:, :1]
        else:
            mask = attention_mask.unsqueeze(-1).to(states.dtype)
            summary = (
                (states * mask).sum(dim=1)
                / mask.sum(dim=1).clamp_min(1.0)
            ).unsqueeze(1)
        return summary, attention_mask[:, :1]

    def score_encoded(
        self,
        context_states,
        context_attention_mask,
        assertion_states,
        *,
        return_auxiliary: bool = False,
    ):
        batch_size = assertion_states.shape[0]
        if context_states.shape[0] == 1 and batch_size != 1:
            context_states = context_states.expand(batch_size, -1, -1)
            context_attention_mask = context_attention_mask.expand(
                batch_size, -1
            )
        elif context_states.shape[0] != batch_size:
            raise ValueError(
                "encoded context batch must be one or match assertions"
            )

        attended, _ = self.cross_attention(
            query=assertion_states,
            key=context_states,
            value=context_states,
            key_padding_mask=~context_attention_mask.bool(),
            need_weights=False,
        )
        query = assertion_states[:, 0]
        attended = attended[:, 0]
        representation = self.interaction(
            torch.cat(
                [
                    query,
                    attended,
                    torch.abs(query - attended),
                    query * attended,
                ],
                dim=-1,
            )
        )
        support_logits = self.head(representation).squeeze(-1)
        if not return_auxiliary:
            return support_logits
        return {
            "support": support_logits,
            "nli": self.nli_head(representation),
            "relevance": self.relevance_head(representation),
            "strength": self.strength_head(representation),
        }

    def forward(
        self,
        context_input_ids,
        context_attention_mask,
        assertion_input_ids,
        assertion_attention_mask,
        *,
        return_auxiliary: bool = False,
    ):
        context_states, context_mask = self.encode_context_tokens(
            context_input_ids,
            context_attention_mask,
        )
        assertion_states, _ = self.encode_assertion_tokens(
            assertion_input_ids,
            assertion_attention_mask,
        )
        return self.score_encoded(
            context_states,
            context_mask,
            assertion_states,
            return_auxiliary=return_auxiliary,
        )


def build_model(
    model_name: str,
    architecture: str = CROSS_ENCODER,
    **kwargs,
):
    if architecture == CROSS_ENCODER:
        return CAPEModel(model_name, **kwargs)
    if architecture == MULTITASK_CROSS_ENCODER:
        return MultiTaskCAPEModel(model_name, **kwargs)
    if architecture == SHARED_CONTEXT:
        return SharedContextCAPEModel(model_name, **kwargs)
    raise ValueError(
        f"Unsupported CAPE architecture {architecture!r}; "
        f"expected one of {sorted(SUPPORTED_ARCHITECTURES)}"
    )
