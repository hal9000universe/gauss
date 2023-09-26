from typing import Dict, Iterable

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers, processors, decoders


def yield_examples(file: str) -> Iterable[str]:
    """Reads the examples from file line by line.

    Args:
        file (str, optional): file to read from.

    Yields:
        Iterable[str]: example
    """
    with open(file, "r") as f:
        examples = f.readlines()
    for line in examples:
        yield line


def build_tokenizer(vocab: Dict[str, int], training_string: str, examples: Iterable[str], save_file: str):
    """Builds a tokenizer.

    Args:
        vocab (Dict[str, int]): vocabulary.
        training_string (str): training string.
        examples (Iterable[str]): examples.
        save_file (str): file to save the tokenizer to.
    """

    tokenizer = Tokenizer(models.WordPiece(vocab=vocab, unk_token="[UNK]", max_input_chars_per_word=100))
    # noinspection PyArgumentList
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.WhitespaceSplit(),
    ])
    tokenizer.pre_tokenizer.pre_tokenize_str(training_string)

    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[]"]
    trainer = trainers.WordPieceTrainer(vocab_size=100, special_tokens=special_tokens)

    tokenizer.train_from_iterator(examples, trainer=trainer)

    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
    )

    tokenizer.decoder = decoders.WordPiece(prefix="##")

    tokenizer.save(save_file)


def fetch_tokenizer(file: str) -> Tokenizer:
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


def build_total_tokenizer():
    """Builds the total tokenizer."""
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
    training_string: str = "-939.0841674804688*x_(0) + 846.0003051757812*x_(1) + 965.515625*x_(2) + " \
                           "-211.8805694580078*x_(3) = 244.3979949951172 /n215.27731323242188*x_(0) " \
                           "+ -169.37924194335938*x_(1) + -772.15966796875*x_(2) + " \
                           "516.14892578125*x_(3) = -451.181884765625 /n-393.0453186035156*x_(0) + " \
                           "72.83699798583984*x_(1) + 686.2369995117188*x_(2) + -70.29736328125*x_(" \
                           "3) = 511.71051025390625 /n202.4686279296875*x_(0) + " \
                           "571.7322998046875*x_(1) + -229.08401489257812*x_(2) + " \
                           "175.90225219726562*x_(3) = 645.262939453125 /n/swap_rows(0,3)"
    examples: Iterable[str] = yield_examples("data/total/train_equations.txt")
    save_file: str = "tokenizer/total_tokenizer.json"
    build_tokenizer(vocab, training_string, examples, save_file)


def build_one_var_tokenizer():
    """Builds the one variable tokenizer."""
    vocab: Dict[str, int] = {
        "[UNK]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "[PAD]": 3,
        "[MASK]": 4,
        "(": 5,
        ")": 6,
        "x": 7,
        "+": 8,
        "-": 9,
        "=": 10,
        "?": 11,
        "/add(": 12,
        "/sub(": 13,
        "/mul(": 14,
        "/div(": 15,
        "/end": 16,
        "/n": 17,
        "0": 18,
        "1": 19,
        "2": 20,
        "3": 21,
        "4": 22,
        "5": 23,
        "6": 24,
        "7": 25,
        "8": 26,
        "9": 27,
        ".": 28,
        " ": 29,
    }
    training_string: str = "10x + 10 = 9.0x + 13?/sub(10)" \
                           "10x = 9.0x + 3.0?/sub(9x)" \
                           "x = 3.0?/div(1.0)" \
                           "x = 3.0?/end"
    examples: Iterable[str] = yield_examples("data/one_var/int/train_equations.txt")
    save_file: str = "tokenizer/one_var_tokenizer.json"
    build_tokenizer(vocab, training_string, examples, save_file)
