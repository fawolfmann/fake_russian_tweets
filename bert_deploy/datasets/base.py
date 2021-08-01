"""Base Dataset class"""
import logging
from typing import List

import torch
from transformers.tokenization_utils_base import BatchEncoding

logger = logging.getLogger(__name__)


class BaseBERTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: BatchEncoding, labels: List):
        """[summary]

        Parameters
        ----------
        encodings : BatchEncoding
            [description]
        labels : List
            [description]
        """

    def __getitem__(self, idx: int) -> torch.tensor:
        """[summary]

        Parameters
        ----------
        idx : int
            [description]

        Returns
        -------
        torch.tensor
            [description]
        """

        return torch.tensor([])

    def __len__(self) -> int:
        """[summary]

        Returns
        -------
        int
            [description]
        """
        return 0
