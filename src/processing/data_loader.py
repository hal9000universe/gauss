#  Copyright (c) 2023. Benjamin Schoofs

from typing import List, Tuple, Callable

import torch
from torch import Tensor
from torch.utils.data import IterDataPipe, DataLoader
import torchtext.transforms as T

from tokenizers import Tokenizer


class EquationDataset(IterDataPipe):
    """Equation dataset.

    Args:
        file (str): file to build the data pipe from
        tokenizer (Tokenizer): tokenizer
    """
    file: str
    tokenizer: Tokenizer

    def __init__(self, file: str, tokenizer: Tokenizer):
        self.file = file
        self.tokenizer = tokenizer

    def __iter__(self):
        for line in open(self.file, "r"):
            # split system and operation
            system_operation = line.split("?")
            # tokenize
            system_tokens = self.tokenizer.encode(system_operation[0]).ids
            operation_tokens = self.tokenizer.encode(system_operation[1]).ids
            for idx in range(0, len(operation_tokens)):
                x = system_tokens + operation_tokens[:idx]
                y = operation_tokens[idx]
                yield x, y


def separate_source_target(sequence_pairs: List[Tuple[List[int], int]]
                           ) -> Tuple[Tuple[List[int], ...], Tuple[int, ...]]:
    """
    input of form: `[(X_1,y_1), (X_2,y_2), (X_3,y_3), (X_4,y_4)]`
    output of form: `((X_1,X_2,X_3,X_4), (y_1,y_2,y_3,y_4))`

    Args:
        sequence_pairs (List[Tuple[List[int], List[int]]]): sequence pairs

    Returns:
        Tuple[Tuple[List[int], ...], Tuple[List[int], ...]]: tuple of sequences
    """
    sources, targets = zip(*sequence_pairs)
    return sources, targets


def generate_collate_fn(padding_value: int) -> Callable:
    """Generates the collate function.

    Args:
        padding_value (int): padding value

    Returns:
        Callable: collate function
    """

    def collate_fn(batch: List[Tuple[List[int], int]]) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        """Collate function for the 2 variable data. ((x, padding_mask), y)

        Args:
            batch (List[Tuple[Tuple[Tensor, Tensor], Tensor]]): batch with a single example

        Returns:
            Tuple[Tuple[Tensor, Tensor], Tensor]: collated data
        """
        sources, targets = separate_source_target(batch)
        # pad the sources and targets
        x = T.ToTensor(padding_value)(list(sources))
        y = T.ToTensor(padding_value)(list(targets))
        # create FloatTensor padding mask for x
        src_key_padding_mask = (x == padding_value).to(torch.bool)
        return (x, src_key_padding_mask), y

    return collate_fn


def generate_data_loader(file: str, tokenizer: Tokenizer, batch_size: int) -> DataLoader:
    """Generates the data loader.

    Args:
        file (str): file to build the data pipe from
        tokenizer (Tokenizer): tokenizer
        batch_size (int): batch size

    Returns:
        DataLoader: data_loader"""
    dataset = EquationDataset(file, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=generate_collate_fn(tokenizer.token_to_id("[PAD]")),
    )
    return dataloader
