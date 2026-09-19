from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .schema import (
    AUXILIARY_LABELS,
    auxiliary_label,
    normalize_example,
    support_target,
)


class CAPEDataset(Dataset):
    def __init__(self, path: str):
        self.examples = []
        with Path(path).open() as file:
            for line_number, line in enumerate(file, 1):
                try:
                    self.examples.append(normalize_example(json.loads(line)))
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid CAPE row at {path}:{line_number}: {error}"
                    ) from error

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    def summary(self):
        heads = Counter()
        sources = Counter()
        capabilities = Counter()
        targets = Counter()
        for row in self.examples:
            sources[row["source"]] += 1
            capabilities[row["capability"]] += 1
            targets[str(row["target"])] += 1
            for field in AUXILIARY_LABELS:
                if row.get(field) is not None:
                    heads[field] += 1
        return {
            "examples": len(self.examples),
            "sources": dict(sorted(sources.items())),
            "capabilities": dict(sorted(capabilities.items())),
            "targets": dict(sorted(targets.items())),
            "auxiliary_labels": dict(sorted(heads.items())),
        }


def _labels(batch, include_auxiliary):
    result = {
        "labels": torch.tensor(
            [support_target(example) for example in batch],
            dtype=torch.float32,
        )
    }
    if include_auxiliary:
        for field in AUXILIARY_LABELS:
            name = field.removesuffix("_label")
            result[f"{name}_labels"] = torch.tensor(
                [auxiliary_label(example, field) for example in batch],
                dtype=torch.long,
            )
    return result


def make_collate_fn(tokenizer, max_length: int, *, include_auxiliary=False):
    def collate(batch):
        tokens = tokenizer(
            [example["context"] for example in batch],
            [example["assertion"] for example in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens.update(_labels(batch, include_auxiliary))
        return tokens

    return collate


def make_shared_context_collate_fn(
    tokenizer,
    max_length: int,
    *,
    include_auxiliary=True,
):
    def collate(batch):
        contexts = tokenizer(
            [example["context"] for example in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        assertions = tokenizer(
            [example["assertion"] for example in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        result = {
            "context_input_ids": contexts["input_ids"],
            "context_attention_mask": contexts["attention_mask"],
            "assertion_input_ids": assertions["input_ids"],
            "assertion_attention_mask": assertions["attention_mask"],
        }
        result.update(_labels(batch, include_auxiliary))
        return result

    return collate
