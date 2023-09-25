#  Copyright (c) 2023. Benjamin Schoofs

import torch
import torchtext.transforms as T

from torchdata.datapipes.iter import IterableWrapper
from typing import List, Tuple, Iterable, Callable
from tokenizers import Tokenizer

from src.data_engine.tokenizer import fetch_tokenizer


def yield_training_corpus(file: str) -> Iterable[Tuple[List[int], List[int]]]:
    """Reads the training corpus from file line by line.

    Args:
        file (str, optional): file to read from.

    Yields:
        Iterable[Tuple[List[int], List[int]]]: training corpus
    """
    tokenizer = fetch_tokenizer(file)
    with open(file, "r") as f:
        examples = f.readlines()
    for line in examples:
        # split system and operation
        system_operation = line.split("?")
        # tokenize 
        system_tokens = tokenizer.encode(system_operation[0]).ids
        operation_tokens = tokenizer.encode(system_operation[1]).ids
        for idx in range(0, len(operation_tokens)):
            x = system_tokens + operation_tokens[:idx]
            y = operation_tokens[idx]
            yield x, y


# data transformation pipeline
def sort_bucket(bucket: List[Tuple[List[int], List[int]]]) -> List[Tuple[List[int], List[int]]]:
    """
    Function to sort a given bucket. Here, we want to sort based on the length of
    source and target sequence.

    Args:
        bucket (List[Tuple[List[int], List[int]]]): bucket to sort

    Returns:
        List[Tuple[List[int], List[int]]]: sorted bucket
    """
    return sorted(bucket, key=lambda x: len(x[0]))


def separate_source_target(
        sequence_pairs: List[Tuple[List[int], List[int]]]) -> Tuple[Tuple[List[int], ...], Tuple[List[int], ...]]:
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


def gen_apply_padding(padding_value: int) -> Callable[[Tuple[List[int], List[int]]], Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]:
    """Generates the apply padding function.

    Args:
        padding_value (int): padding value.

    Returns:
        Callable[[Tuple[List[int], List[int]]], Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]: apply padding function
    """
    def apply_padding(pair_of_sequences) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Convert sequences to tensors and apply padding

        Args:
            pair_of_sequences (Tuple[List[int], List[int]]): a pair of sequences

        Returns:
            Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: tuple of tensors
        """
        x = T.ToTensor(padding_value)(list(pair_of_sequences[0]))
        y = T.ToTensor(padding_value)(list(pair_of_sequences[1]))
        # create FloatTensor padding mask for x which is added in the forward pass
        src_key_padding_mask = (x == padding_value).to(torch.bool)
        return (x, src_key_padding_mask), y
    return apply_padding


def build_data_pipe(tokenizer: Tokenizer, data_file: str, batch_size: int) -> IterableWrapper:
    """Builds the data pipe.

    Args:
        tokenizer (Tokenizer): tokenizer.
        data_file (str, optional): file to read from.
        batch_size (int, optional): batch size.

    Returns:
        IterableWrapper: data pipe
    """
    # get training corpus
    dp: IterableWrapper = IterableWrapper(yield_training_corpus(file=data_file))
    # bucket batch
    dp = dp.bucketbatch(
        batch_size=batch_size,
        use_in_batch_shuffle=False,
        sort_key=sort_bucket,
    )
    # separate source and target
    dp = dp.map(separate_source_target)
    # apply padding and convert to tensors
    dp = dp.map(gen_apply_padding(tokenizer.token_to_id("[PAD]")))
    dp = dp
    return dp
