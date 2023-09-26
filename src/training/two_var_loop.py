#  Copyright (c) 2023. Benjamin Schoofs

import torch
from typing import List, Callable
from torchdata.datapipes.iter import IterableWrapper
from tokenizers import Tokenizer

from src.data_engine.gaussian_elimination import gaussian_elimination
from src.data_engine.repr import repr_record
from src.data_engine.data_pipe import build_data_pipe
from src.data_engine.tokenizer import fetch_tokenizer
from src.gauss_net.transformer import GaussNet
from src.training.training import training_loop


def gen_2var_examples(n: int) -> List[str]:
    """Generates 2 variable systems of equations.

    Args:
        n (int): number of examples to generate

    Returns:
        List[str]: list of examples
    """
    examples: List[str] = []
    names: List[str] = ["x", "y"]
    for i in range(0, n):
        system = torch.rand(2, 3) * 2000. - 1000.
        solution, record = gaussian_elimination(system)
        example = repr_record(record, names)
        examples.append(example)
    return examples


def gen_2var_data(num_examples: int, save_file: str):
    """Generates the data and saves it to file.

    Args:
        num_examples (int): number of examples to generate
        save_file (str): file to save the examples to
    """
    examples = gen_2var_examples(num_examples)
    # save to file
    with open(save_file, "w") as f:
        for example in examples:
            f.write(example)


def gen_2var_data_pipe_builder(tokenizer: Tokenizer, file: str, batch_size: int) -> Callable[[], IterableWrapper]:
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


def two_var_loop():
    # set up
    train_data_file: str = "data/2var/train_equations.txt"
    test_data_file: str = "data/2var/test_equations.txt"
    model_save_file: str = "models/2var/gauss.pt"
    # fetch tokenizer
    tokenizer = fetch_tokenizer("tokenizer/2var.json")
    # generate data
    num_examples = 10000
    gen_2var_data(num_examples=num_examples, save_file=train_data_file)
    gen_2var_data(num_examples=num_examples // 100, save_file=test_data_file)
    # build data pipe
    train_data_pipe_builder = gen_2var_data_pipe_builder(tokenizer, train_data_file, 64)
    test_data_pipe_builder = gen_2var_data_pipe_builder(tokenizer, test_data_file, 10)
    # create model
    model = GaussNet(
        embed_dim=64,
        dim_feedforward=512,
        num_heads=4,
        num_layers=4,
        tokenizer=tokenizer,
    )
    # create optimizer
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # define training loop
    num_epochs: int = 10000
    monitor_freq = 1000
    evaluation_freq = 10000
    plotting_freq = 10000
    plot_file = f"plots/gauss-2var-easy-min-{num_epochs}-{num_examples}-{lr}.png"

    # train
    training_loop(
        train_data_pipe_builder=train_data_pipe_builder,
        test_data_pipe_builder=test_data_pipe_builder,
        model=model,
        optimizer=optimizer,
        num_epochs=num_epochs,
        monitor_freq=monitor_freq,
        evaluation_freq=evaluation_freq,
        save_file=model_save_file,
        plotting_freq=plotting_freq,
        plot_file=plot_file,
    )
