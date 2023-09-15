from torch import Tensor, rand
from typing import List
from random import randint

from src.data_engine.solve import gaussian_elimination, repr_record

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

def test_gaussian_elimination():
    system = Tensor([
        [1, 2, 3, 4],
        [2, 1, 3, 7],
        [3, 2, 1, 6]
    ])
    system, record = gaussian_elimination(system)
    print(repr_record(record, ["x", "y", "z"]))


def generate_examples(n: int = 10) -> List[str]:
    examples: List[str] = []
    for i in range(n):
        # generate random matrix
        # 1. choose number of variables and equations
        num_vars = randint(2, 5)
        num_eqs = randint(num_vars, 5)
        # 2. generate random matrix
        mat = rand((num_eqs, num_vars + 1))
        # 3. compute solution
        mat, record = gaussian_elimination(mat)
        # 4. format solution
        names: List[str] = [f"x_({i})" for i in range(num_vars)]
        example = repr_record(record, names)
        examples.append(example)
    return examples


def init_data():
    examples = generate_examples(100)
    # train tokenizer
    with open("data/examples.txt", "w") as f:
        for example in examples:
            f.write(example + "\n")


def get_training_corpus():
    """Reads the training corpus from file line by line."""
    with open("data/examples.txt", "r") as f:
        examples = f.readlines()
    for line in examples:
        yield line


def build_tokenizer():
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
    tokenizer.pre_tokenizer.pre_tokenize_str("0.059598565101623535*x_(0) + 0.008593916893005371*x_(1) + 0.6809395551681519*x_(2) + 0.25285983085632324*x_(3) + 0.23460513353347778*x_(4) = 0.15194106101989746 \n/swap_rows(0,3)")

    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK], []"]
    trainer = trainers.WordPieceTrainer(vocab_size=500, special_tokens=special_tokens)

    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    
    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
    )

    tokenizer.decoder = decoders.WordPiece(prefix="##")

    tokenizer.save("data/tokenizer.json")


def format_decoding(decoding: str) -> str:
    for i in range(0, 10):
        decoding = decoding.replace(f"{i} ", f"{i}")
    decoding = decoding.replace(". ", ".")
    decoding = decoding.replace("( ", "(")
    decoding = decoding.replace("- ", "-")
    decoding = decoding.replace(", ", ",")
    return decoding

def test_tokenizer():
    build_tokenizer()
    tokenizer = Tokenizer.from_file("data/tokenizer.json")
    original = "/reduce_rows(1,0,/div(-0.5246381759643555,1.0))"
    encoding = tokenizer.encode(original)
    decoding = tokenizer.decode(encoding.ids)
    decoding = format_decoding(decoding)
    assert(original == decoding)

def get_tokenizer() -> Tokenizer:
    return Tokenizer.from_file("data/tokenizer.json")


if __name__ == "__main__":
    init_data()
    test_tokenizer()