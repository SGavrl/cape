import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class CAPEDataset(Dataset):
    def __init__(self, path: str):
        self.examples = []

        with Path(path).open() as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def make_collate_fn(tokenizer, max_length: int):
    def collate(batch):
        contexts = [
            example["context"]
            for example in batch
        ]

        assertions = [
            example["assertion"]
            for example in batch
        ]

        labels = torch.tensor(
            [
                example["label"]
                for example in batch
            ],
            dtype=torch.float32,
        )

        tokens = tokenizer(
            contexts,
            assertions,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        tokens["labels"] = labels

        return tokens

    return collate
