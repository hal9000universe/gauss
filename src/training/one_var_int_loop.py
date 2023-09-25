#  Copyright (c) 2023. Benjamin Schoofs

import torch
from typing import List, Callable
from torchdata.datapipes.iter import IterableWrapper
from tokenizers import Tokenizer

from src.data_engine.one_var_solve import solve, Equation
from src.data_engine.data_pipe import build_data_pipe


def gen_1var_int_examples(n: int) -> List[str]:
    """Generates 1 variable equations.

    Args:
        n (int): number of examples to generate

    Returns:
        List[str]: list of examples
    """
    examples: List[str] = []
    for i in range(0, n):
        equation = Equation.generate_rand_int_equation(20, -100, 100)
        solution, record = solve(equation)
        example = str(record)
        examples.append(example)

    return examples


def gen_1var_int_data(num_examples: int, save_file: str):
    """Generates the data and saves it to file.

    Args:
        num_examples (int): number of examples to generate
        save_file (str): file to save the examples to
    """
    examples = gen_1var_int_examples(num_examples)
    # save to file
    with open(save_file, "w") as f:
        for example in examples:
            f.write(example)


def gen_1var_int_data_pipe_builder(tokenizer: Tokenizer, file: str, batch_size: int) -> Callable[[], IterableWrapper]:
    """Returns a data pipe builder for the 2 variable data.

    Args:
        tokenizer: Tokenizer
        file (str): file to build the data pipe from
        batch_size (int): batch size

    Returns:
        Callable[[[], IterableWrapper]]: data pipe builder
    """
    def data_pipe_builder() -> IterableWrapper:
        return build_data_pipe(tokenizer, file, batch_size)
    return data_pipe_builder
