#  Copyright (c) 2023. Benjamin Schoofs

import os
from typing import List

import torch

from src.data_engine.one_var_solve import solve, Equation
from src.processing.tokenizer import fetch_tokenizer
from src.training.training import training_loop
from src.models.transformer import MathFormer
from src.processing.data_loader import generate_data_loader

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
    num_test_examples: int = 50
    batch_size: int = 512
    num_epochs: int = 1000

    # generate data
    gen_1var_int_data(num_examples, train_data_file)
    gen_1var_int_data(num_test_examples, test_data_file)

    # load tokenizer
    tokenizer = fetch_tokenizer("tokenizer/one_var_tokenizer.json")

    # build data loaders
    train_data_loader = generate_data_loader(train_data_file, tokenizer, batch_size)
    test_data_loader = generate_data_loader(test_data_file, tokenizer, batch_size)
    train_eval_data_loader = generate_data_loader(train_data_file, tokenizer, batch_size)

    # build model
    model = MathFormer(
        embed_dim=256,
        dim_feedforward=2048,
        num_heads=8,
        num_layers=12,
        tokenizer=tokenizer,
    )
    # load model
    if os.path.exists("models/one_var/int/gauss.pt"):
        try:
            model.load_state_dict(torch.load("models/one_var/int/gauss.pt"))
        except RuntimeError:
            print("Could not load model. Starting from scratch.")
    # create optimizer
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # define training loop
    monitor_freq = 100
    evaluation_freq = 1000
    save_file = f"models/one_var/int/gauss.pt"
    plotting_freq = 2000
    plot_file = f"plots/one_var/int/{num_epochs}-{num_examples}-{lr}.png"
    save_freq = 10000

    # accelerator
    accelerator = Accelerator()
    model, optimizer, train_data_loader, test_data_loader = accelerator.prepare(
        [model, optimizer, train_data_loader, test_data_loader]
    )

    # train model
    training_loop(
        accelerator=accelerator,
        train_data_loader=train_data_loader,
        test_data_loader=test_data_loader,
        train_eval_data_loader=train_eval_data_loader,
        model=model,
        optimizer=optimizer,
        num_epochs=num_epochs,
        monitor_freq=monitor_freq,
        evaluation_freq=evaluation_freq,
        save_file=save_file,
        plotting_freq=plotting_freq,
        plot_file=plot_file,
        save_freq=save_freq,
    )
