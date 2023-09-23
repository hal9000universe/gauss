#  Copyright (c) 2023. Benjamin Schoofs

import torch
import torchtext.transforms as T

from torchdata.datapipes.iter import IterableWrapper
from typing import List, Tuple, Iterable, Dict
from random import randint
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

from src.data_engine.solve import gaussian_elimination
from src.data_engine.repr import repr_record


FILE: str = "data/examples.txt"


def gen_examples(n: int = 10, min_num_vars: int = 2, max_num_vars: int = 10) -> List[str]:
    """Generates n random examples.
    
    Args:
        n (int, optional): number of examples. Defaults to 10.
        min_num_vars (int, optional): minimal number of variables. Defaults to 2.
        max_num_vars (int, optional): maximal number of variables. Defaults to 10.
        
    Returns:
        List[str]: list of examples"""
    examples: List[str] = []
    for i in range(n):
        # generate random matrix
        # 1. choose number of variables and equations
        num_vars = randint(min_num_vars, max_num_vars)
        num_eqs = num_vars
        # 2. generate random matrix
        system = (torch.rand(num_eqs, num_vars + 1) - 0.5) * 2000.
        # 3. compute solution
        solution, record = gaussian_elimination(system)
        # 4. format solution
        names: List[str] = [f"x_({i})" for i in range(num_vars)]
        example = repr_record(record, names)
        examples.append(example)
    return examples


def gen_data(num_examples: int = 100, save_file: str = FILE):
    """Generates the data and saves it to file.

    Args:
        save_file (str, optional): file to save the data to. Defaults to FILE.
        num_examples (int, optional): number of examples. Defaults to 100.
    """
    examples = gen_examples(num_examples, 2, 10)
    # train tokenizer
    with open(save_file, "w") as f:
        for example in examples:
            f.write(example)


def yield_examples(file: str = FILE) -> Iterable[str]:
    """Reads the examples from file line by line.

    Args:
        file (str, optional): file to read from. Defaults to FILE.

    Yields:
        Iterable[str]: example
    """
    with open(file, "r") as f:
        examples = f.readlines()
    for line in examples:
        yield line


def build_tokenizer(save_file: str = "data/tokenizer.json"):
    """Builds a tokenizer."""
    vocab: Dict[str, int] = {
        "[UNK]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "[PAD]": 3,
        "[MASK]": 4,
        "(": 5,
        ")": 6,
        "*": 7,
        "+": 8,
        ",": 9,
        "-": 10,
        ".": 11,
        "/": 12,
        "0": 13,
        "1": 14,
        "2": 15,
        "3": 16,
        "4": 17,
        "5": 18,
        "6": 19,
        "7": 20,
        "8": 21,
        "9": 22,
        "=": 23,
        "?": 24,
        "_": 25,
        "/reduce_rows": 26,
        "/swap_rows": 27,
        "/multiply_row": 28,
        "/n": 29,
        "/div": 30,
        "/rref": 31,
        "#": 32,
        "a": 33,
        "b": 34,
        "c": 35,
        "d": 36,
        "e": 37,
        "f": 38,
        "g": 39,
        "h": 40,
        "i": 41,
        "j": 42,
        "k": 43,
        "l": 44,
        "m": 45,
        "n": 46,
        "o": 47,
        "p": 48,
        "q": 49,
        "r": 50,
        "s": 51,
        "t": 52,
        "u": 53,
        "v": 54,
        "w": 55,
        "x": 56,
        "y": 57,
        "z": 58,
        "[": 59,
        "]": 60,
    }
    tokenizer = Tokenizer(models.WordPiece(vocab=vocab, unk_token="[UNK]", max_input_chars_per_word=100))
    # noinspection PyArgumentList
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.WhitespaceSplit(),
    ])
    tokenizer.pre_tokenizer.pre_tokenize_str("-939.0841674804688*x_(0) + 846.0003051757812*x_(1) + 965.515625*x_(2) + "
                                             "-211.8805694580078*x_(3) = 244.3979949951172 /n215.27731323242188*x_(0) "
                                             "+ -169.37924194335938*x_(1) + -772.15966796875*x_(2) + "
                                             "516.14892578125*x_(3) = -451.181884765625 /n-393.0453186035156*x_(0) + "
                                             "72.83699798583984*x_(1) + 686.2369995117188*x_(2) + -70.29736328125*x_("
                                             "3) = 511.71051025390625 /n202.4686279296875*x_(0) + "
                                             "571.7322998046875*x_(1) + -229.08401489257812*x_(2) + "
                                             "175.90225219726562*x_(3) = 645.262939453125 /n/swap_rows(0,3)")

    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[]"]
    trainer = trainers.WordPieceTrainer(vocab_size=100, special_tokens=special_tokens)

    tokenizer.train_from_iterator(yield_examples(), trainer=trainer)
    
    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
    )

    tokenizer.decoder = decoders.WordPiece(prefix="##")

    tokenizer.save(save_file)


def get_tokenizer(file: str = "data/tokenizer.json") -> Tokenizer:
    """Retrieves the tokenizer.
    
    Returns:
        Tokenizer: tokenizer"""
    return Tokenizer.from_file(file)


def format_decoding(decoding: str) -> str:
    """Formats the decoding.
    
    Args:
        decoding (str): decoding
        
    Returns:
        str: formatted decoding"""
    for i in range(0, 10):
        decoding = decoding.replace(f"{i} ", f"{i}")
    decoding = decoding.replace(". ", ".")
    decoding = decoding.replace("( ", "(")
    decoding = decoding.replace("- ", "-")
    decoding = decoding.replace(", ", ",")
    decoding = decoding.replace("/n ", " /n")
    return decoding


def yield_training_corpus(file: str = FILE) -> Iterable[Tuple[List[int], List[int]]]:
    """Reads the training corpus from file line by line.

    Args:
        file (str, optional): file to read from. Defaults to FILE.

    Yields:
        Iterable[Tuple[List[int], List[int]]]: training corpus
    """
    with open(file, "r") as f:
        examples = f.readlines()
    for line in examples:
        # split system and operation
        system_operation = line.split("?")
        # tokenize 
        system_tokens = get_tokenizer().encode(system_operation[0]).ids
        operation_tokens = get_tokenizer().encode(system_operation[1]).ids
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


def apply_padding(pair_of_sequences) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Convert sequences to tensors and apply padding

    Args:
        pair_of_sequences (Tuple[List[int], List[int]]): a pair of sequences

    Returns:
        Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: tuple of tensors
    """
    padding_value = get_tokenizer().token_to_id("[PAD]")
    x = T.ToTensor(padding_value)(list(pair_of_sequences[0]))
    y = T.ToTensor(padding_value)(list(pair_of_sequences[1]))
    # create FloatTensor padding mask for x which is added in the forward pass
    src_key_padding_mask = (x == padding_value).to(torch.bool)
    return (x, src_key_padding_mask), y


def build_data_pipe(data_file: str = FILE, batch_size: int = 64) -> IterableWrapper:
    """Builds the data pipe.

    Args:
        data_file (str, optional): file to read from. Defaults to FILE.
        batch_size (int, optional): batch size. Defaults to 64.

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
    dp = dp.map(apply_padding)
    dp = dp
    return dp
