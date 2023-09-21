#  Copyright (c) 2023. Benjamin Schoofs

import torch
from typing import List
from torchdata.datapipes.iter import IterableWrapper

from src.data_engine.solve import gaussian_elimination
from src.data_engine.repr import repr_record
from src.data_engine.data_pipe import build_data_pipe


FILE: str = "data/easy_examples.txt"


def gen_easy_examples(n: int = 10):
    """Generates 2 variable systems of equations."""
    examples: List[str] = []
    names: List[str] = ["x", "y"]
    for i in range(0, n):
        system = torch.rand(2, 3) * 2000. - 1000.
        solution, record = gaussian_elimination(system)
        example = repr_record(record, names)
        examples.append(example)
    return examples


def gen_easy_data():
    """Generates the data and saves it to file."""
    examples = gen_easy_examples(100)
    # save to file
    with open(FILE, "w") as f:
        for example in examples:
            f.write(example)


def build_easy_data_pipe() -> IterableWrapper:
    return build_data_pipe(FILE)
