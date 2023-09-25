import torch

from typing import List
from random import randint

from src.gauss_net.transformer import GaussNet
from src.data_engine.data_pipe import build_data_pipe, fetch_tokenizer
from src.training.training import training_loop
from src.data_engine.gaussian_elimination import gaussian_elimination
from src.data_engine.repr import repr_record


def gen_total_examples(n: int, min_num_vars: int, max_num_vars: int) -> List[str]:
    """Generates n random examples.

    Args:
        n (int, optional): number of examples.
        min_num_vars (int, optional): minimal number of variables.
        max_num_vars (int, optional): maximal number of variables.

    Returns:
        List[str]: list of examples"""
    examples: List[str] = []
    for i in range(n):
        # generate random matrix
        # 1. choose number of variables and equations
        num_vars = randint(min_num_vars, max_num_vars)
        num_eqs = num_vars
        # 2. generate random matrix
        system = (torch.rand(num_eqs, num_vars + 1) - 0.5) * 2000.
        # 3. compute solution
        solution, record = gaussian_elimination(system)
        # 4. format solution
        names: List[str] = [f"x_({i})" for i in range(num_vars)]
        example = repr_record(record, names)
        examples.append(example)
    return examples


def gen_total_data(num_examples: int, save_file: str):
    """Generates the data and saves it to file.

    Args:
        save_file (str, optional): file to save the data to.
        num_examples (int, optional): number of examples.
    """
    examples = gen_total_examples(num_examples, 2, 10)
    # train tokenizer
    with open(save_file, "w") as f:
        for example in examples:
            f.write(example)


def total_loop():
    """Total loop for the GaussNet.

    Training on 10000 examples of Gaussian elimination with a large GaussNet."""
    # set up
    train_data_file: str = "data/total/train_equations.txt"
    test_data_file: str = "data/total/test_equations.txt"
    model_save_file: str = "models/total/gauss.pt"
    # generate data
    num_examples = 10000
    gen_total_data(num_examples=num_examples, save_file=train_data_file)
    gen_total_data(num_examples=num_examples // 10, save_file=test_data_file)
    # build data pipe
    data_pipe_builder = build_data_pipe
    # fetch tokenizer
    tokenizer = fetch_tokenizer("tokenizer/total.json")
    # create model
    model = GaussNet(
        embed_dim=64,
        dim_feedforward=512,
        num_heads=8,
        num_layers=10,
        tokenizer=tokenizer,
    )
    # create optimizer
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # define training loop
    num_epochs: int = 50000
    monitor_freq = 1000
    evaluation_freq = 10000
    plotting_freq = 10000
    plot_file = f"plots/big-gauss-{num_epochs}-{num_examples}-{lr}.png"

    # train
    training_loop(
        data_pipe_builder=data_pipe_builder,
        model=model,
        optimizer=optimizer,
        num_epochs=num_epochs,
        monitor_freq=monitor_freq,
        evaluation_freq=evaluation_freq,
        save_file=model_save_file,
        evaluation_file=test_data_file,
        plotting_freq=plotting_freq,
        plot_file=plot_file,
    )
