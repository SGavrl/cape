import torch
import torch.nn as nn

from transformers import AutoModel


class CAPEModel(nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_ids,
        attention_mask,
        **kwargs,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        # Representation of the first token.
        representation = outputs.last_hidden_state[:, 0]

        representation = self.dropout(representation)

        logits = self.head(representation)

        return logits.squeeze(-1)
