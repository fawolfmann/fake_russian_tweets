"""Covid Dataset"""
from typing import List

import torch
from transformers.tokenization_utils_base import BatchEncoding


class FakeTweetsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: BatchEncoding, labels: List):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> torch.tensor:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)
