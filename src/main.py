import torch
import torch.nn as nn
from torch.nn import Embedding
from typing import List, Tuple

from src.data_engine.solve import *
from src.data_engine.data_pipe import *
from src.gauss_net.transformer import *


def test_gaussian_elimination():
    system = torch.tensor([
        [1, 2, 3, 4],
        [2, 1, 3, 7],
        [3, 2, 1, 6]
    ], dtype=torch.float32)
    system, record = gaussian_elimination(system)
    print(repr_record(record, ["x", "y", "z"]).replace("/n", "\n"))


def test_tokenizer():
    # get tokenizer
    tokenizer = get_tokenizer()
    original = "-312.33953857421875*x_(0) + 856.2227783203125*x_(1) + 352.9467468261719*x_(2) + 529.22119140625*x_(3) = -435.47772216796875 /n789.8570556640625*x_(0) + 19.7451114654541*x_(1) + 642.3790283203125*x_(2) + 339.9715576171875*x_(3) = -719.56396484375 /n309.1565246582031*x_(0) + 782.404296875*x_(1) + -584.4415283203125*x_(2) + 984.8065185546875*x_(3) = -891.5484008789062 /n816.6369018554688*x_(0) + -201.58993530273438*x_(1) + 488.4422912597656*x_(2) + -371.6492614746094*x_(3) = -934.2286376953125?/swap_rows(0,3)"
    encoded = tokenizer.encode(original)
    decoded = format_decoding(tokenizer.decode(encoded.ids))
    print(encoded.ids)
    print(decoded)
    assert original == decoded


def test_data_pipe():
    tokenizer = get_tokenizer()
    data_pipe = build_data_pipe()
    (x, src_key_padding_mask), y = next(iter(data_pipe))
    print(x.shape, x)
    print(y.shape, y)


def test_gauss_net():
    # get input
    data_pipe = build_data_pipe()
    (x, src_key_padding_mask), y = next(iter(data_pipe))
    print(x.shape, y.shape)

    print(src_key_padding_mask)

    model = create_gauss_net()
    output = model(x, src_key_padding_mask=src_key_padding_mask)
    print(output.shape)

    print(x[0])

    prediction = output.argmax(dim=-1)
    print(prediction.shape)

    loss = model(x, targets=y)
    print(loss.shape)


def all():
    tokenizer = get_tokenizer()

    data_pipe = build_data_pipe()
    x, y = next(iter(data_pipe))

    model = create_gauss_net()
    output = model(x)

    prediction = output.argmax(dim=-1)
    
    print(x.shape)
    print(output.shape)
    print(prediction.shape)


if __name__ == "__main__":
    test_gauss_net()