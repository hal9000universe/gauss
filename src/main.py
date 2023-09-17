from torch.nn import Embedding
from typing import List, Tuple

from src.data_engine.solve import gaussian_elimination, repr_record
from src.data_engine.data_pipe import build_data_pipe, format_decoding, get_tokenizer


def test_gaussian_elimination():
    system = Tensor([
        [1, 2, 3, 4],
        [2, 1, 3, 7],
        [3, 2, 1, 6]
    ])
    system, record = gaussian_elimination(system)
    print(repr_record(record, ["x", "y", "z"]).replace("/n", "\n"))

def test_tokenizer():
    tokenizer = get_tokenizer()
    original = "/reduce_rows(1,0,/div(-0.5246381759643555,1.0))"
    encoding = tokenizer.encode(original)
    decoding = tokenizer.decode(encoding.ids)
    decoding = format_decoding(decoding)
    assert(original == decoding)


def create_embedding() -> Embedding:
    """Creates the embedding layer.

    Returns:
        Embedding: embedding layer
    """
    tokenizer = get_tokenizer()
    embedding = Embedding(tokenizer.get_vocab_size(), 100, padding_idx=tokenizer.token_to_id("[PAD]"))
    return embedding


if __name__ == "__main__":
    data_pipe = build_data_pipe()
    first_batch = next(iter(data_pipe))
    print(first_batch[0].shape, first_batch[1].shape)