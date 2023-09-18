import torch
import torchtext.transforms as T

from torchdata.datapipes.iter import IterableWrapper
from typing import List, Tuple, Iterable
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

from src.data_engine.solve import gaussian_elimination, repr_record


def generate_examples(n: int = 10) -> List[str]:
    """Generates n random examples.
    
    Args:
        n (int, optional): number of examples. Defaults to 10.
        
    Returns:
        List[str]: list of examples"""
    examples: List[str] = []
    for i in range(n):
        # generate random matrix
        # 1. choose number of variables and equations
        num_vars = randint(2, 10)
        num_eqs = num_vars
        # 2. generate random matrix
        mat = (torch.rand(num_eqs, num_vars + 1) - 0.5) * 2000.
        # 3. compute solution
        mat, record = gaussian_elimination(mat)
        # 4. format solution
        names: List[str] = [f"x_({i})" for i in range(num_vars)]
        example = repr_record(record, names)
        examples.append(example)
    return examples


def gen_data():
    """Generates the data and saves it to file."""
    examples = generate_examples(100)
    # train tokenizer
    with open("data/examples.txt", "w") as f:
        for example in examples:
            f.write(example)


def yield_training_corpus() -> Iterable[str]:
    """Reads the training corpus from file line by line."""
    with open("data/examples.txt", "r") as f:
        examples = f.readlines()
    for line in examples:
        yield line


def build_tokenizer() -> Tokenizer:
    """Builds a tokenizer.
    
    Returns:
        Tokenizer: tokenizer"""
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(), 
        normalizers.Lowercase(), 
        normalizers.StripAccents(),
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.WhitespaceSplit(),
    ])
    tokenizer.pre_tokenizer.pre_tokenize_str("-939.0841674804688*x_(0) + 846.0003051757812*x_(1) + 965.515625*x_(2) + -211.8805694580078*x_(3) = 244.3979949951172 /n215.27731323242188*x_(0) + -169.37924194335938*x_(1) + -772.15966796875*x_(2) + 516.14892578125*x_(3) = -451.181884765625 /n-393.0453186035156*x_(0) + 72.83699798583984*x_(1) + 686.2369995117188*x_(2) + -70.29736328125*x_(3) = 511.71051025390625 /n202.4686279296875*x_(0) + 571.7322998046875*x_(1) + -229.08401489257812*x_(2) + 175.90225219726562*x_(3) = 645.262939453125 /n/swap_rows(0,3)")

    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK], []"]
    trainer = trainers.WordPieceTrainer(vocab_size=100, special_tokens=special_tokens)

    tokenizer.train_from_iterator(yield_training_corpus(), trainer=trainer)
    
    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
    )

    tokenizer.decoder = decoders.WordPiece(prefix="##")

    tokenizer.save("data/tokenizer.json")


def get_tokenizer() -> Tokenizer:
    """Retrieves the tokenizer.
    
    Returns:
        Tokenizer: tokenizer"""
    return Tokenizer.from_file("data/tokenizer.json")


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


# data transformation pipeline
def split_x_y(example: str) -> Tuple[str, str]:
    """Splits the example system_operation_pair into the system and the operation.

    Args:
        example (str): example system_operation_pair

    Returns:
        Tuple[str, str]: system, operation
    """
    system, operation = example.split("?")
    operation = operation.replace(" \n", "")
    return (system, operation)

def encode_x_y(system_operation_pair: Tuple[str, str]) -> Tuple[List[int], List[int]]:
    """Encodes the system_operation_pair.

    Args:
        system_operation_pair (Tuple[str, str]): system_operation_pair

    Returns:
        Tuple[List[int], List[int]]: encoded system_operation_pair
    """
    tokenizer = get_tokenizer()
    system, operation = system_operation_pair
    system = tokenizer.encode(system).ids
    operation = tokenizer.encode(operation).ids
    return (system, operation)


def sort_bucket(bucket: List[Tuple[List[int], List[int]]]) -> List[Tuple[List[int], List[int]]]:
    """
    Function to sort a given bucket. Here, we want to sort based on the length of
    source and target sequence.

    Args:
        bucket (List[Tuple[List[int], List[int]]]): bucket to sort

    Returns:
        List[Tuple[List[int], List[int]]]: sorted bucket
    """
    return sorted(bucket, key=lambda x: (len(x[0]), len(x[1])))


def separate_source_target(sequence_pairs: List[Tuple[List[int], List[int]]]) -> Tuple[Tuple[List[int], ...], Tuple[List[int], ...]]:
    """
    input of form: `[(X_1,y_1), (X_2,y_2), (X_3,y_3), (X_4,y_4)]`
    output of form: `((X_1,X_2,X_3,X_4), (y_1,y_2,y_3,y_4))`

    Args:
        sequence_pairs (List[Tuple[List[int], List[int]]]): sequence pairs

    Returns:
        Tuple[Tuple[List[int], ...], Tuple[List[int], ...]]: tuple of sequences
    """
    sources,targets = zip(*sequence_pairs)
    return sources,targets


def apply_padding(pair_of_sequences) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert sequences to tensors and apply padding

    Args:
        pair_of_sequences (Tuple[List[int], List[int]]): pair of sequences

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: pair of tensors
    """
    padding_value = get_tokenizer().token_to_id("[PAD]")
    return (T.ToTensor(padding_value)(list(pair_of_sequences[0])), 
            T.ToTensor(padding_value)(list(pair_of_sequences[1])))


def build_data_pipe() -> IterableWrapper:
    """Builds the data pipe.

    Returns:
        IterableWrapper: data pipe
    """
    # get training corpus
    dp: IterableWrapper = IterableWrapper(yield_training_corpus())
    # split system and operation
    dp = dp.map(split_x_y) 
    # tokenize system and operation
    tokenizer = get_tokenizer()
    dp = dp.map(encode_x_y)
    # bucket batch
    dp = dp.bucketbatch(
        batch_size=64,
        use_in_batch_shuffle=False, 
        sort_key=sort_bucket,
    )
    # separate source and target
    dp = dp.map(separate_source_target)
    # apply padding and convert to tensors
    dp = dp.map(apply_padding)
    dp = dp
    return dp