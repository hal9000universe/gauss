#  Copyright (c) 2023. Benjamin Schoofs

import os
from typing import List, Tuple, Callable

import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from src.data_engine.one_var_solve import solve, Equation
from src.data_engine.data_pipe import build_data_pipe
from src.data_engine.tokenizer import fetch_tokenizer
from src.training.training import training_loop
from src.gauss_net.transformer import GaussNet

from accelerate import Accelerator


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


def collate_fn(batch: List[Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]
               ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Collate function for the 2 variable data. ((x, padding_mask), y)

    Args:
        batch (List[Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]): batch with a single example

    Returns:
        torch.Tensor: collated data
    """
    return batch[0]


def gen_1var_int_data_loader_builder(tokenizer: Tokenizer, file: str, batch_size: int) -> Callable[[], DataLoader]:
    """Returns a data loader for the 2 variable data.

    Args:
        tokenizer: Tokenizer
        file (str): file to build the data pipe from
        batch_size (int): batch size

    Returns:
        DataLoader: data_loader
    """
    def build_1var_int_data_loader() -> DataLoader:
        data_pipe = build_data_pipe(tokenizer, file, batch_size)
        data_loader = DataLoader(data_pipe, collate_fn=collate_fn, shuffle=True)
        return data_loader

    return build_1var_int_data_loader


def one_var_int_loop():
    """One variable integer loop for the GaussNet.

    Training on 10000 examples of Gaussian elimination with a large GaussNet."""
    # make directories
    if not os.path.exists("models/one_var/int"):
        os.makedirs("models/one_var/int")
    if not os.path.exists("plots/one_var/int"):
        os.makedirs("plots/one_var/int")
    if not os.path.exists("data/one_var/int"):
        os.makedirs("data/one_var/int")
    # set up
    train_data_file: str = "data/one_var/int/train_equations.txt"
    test_data_file: str = "data/one_var/int/test_equations.txt"
    num_examples: int = 10000
    batch_size: int = 64
    num_epochs: int = 100
    # generate data
    gen_1var_int_data(num_examples, train_data_file)
    gen_1var_int_data(num_examples // 100, test_data_file)
    # fetch tokenizer
    tokenizer = fetch_tokenizer("tokenizer/one_var_tokenizer.json")
    # build data loaders
    train_data_loader_builder = gen_1var_int_data_loader_builder(tokenizer, train_data_file, batch_size)
    test_data_loader_builder = gen_1var_int_data_loader_builder(tokenizer, test_data_file, batch_size)
    # build model
    model = GaussNet(
        embed_dim=64,
        dim_feedforward=1024,
        num_heads=4,
        num_layers=6,
        tokenizer=tokenizer,
    )
    # load model
    load = True
    if os.path.exists("models/one_var/int/gauss.pt") and load:
        model.load_state_dict(torch.load("models/one_var/int/gauss.pt"))
    # create optimizer
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # define training loop
    monitor_freq = 100
    evaluation_freq = 10000
    save_file = f"models/one_var/int/gauss.pt"
    plotting_freq = 2000
    plot_file = f"plots/one_var/int/{num_epochs}-{num_examples}-{lr}.png"

    # accelerator
    accelerator = Accelerator()
    model, optimizer = accelerator.prepare(
        [model, optimizer]
    )

    # train model
    training_loop(
        accelerator=accelerator,
        train_data_loader_builder=train_data_loader_builder,
        test_data_loader_builder=test_data_loader_builder,
        model=model,
        optimizer=optimizer,
        num_epochs=num_epochs,
        monitor_freq=monitor_freq,
        evaluation_freq=evaluation_freq,
        save_file=save_file,
        plotting_freq=plotting_freq,
        plot_file=plot_file,
    )
