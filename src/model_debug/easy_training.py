#  Copyright (c) 2023. Benjamin Schoofs

import torch
from typing import List
from torchdata.datapipes.iter import IterableWrapper

from src.data_engine.solve import gaussian_elimination
from src.data_engine.repr import repr_record
from src.data_engine.data_pipe import build_data_pipe
from src.gauss_net.transformer import GaussNet
from src.gauss_net.training import training_loop


# define save file
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


def gen_easy_data(num_examples: int = 100, save_file: str = FILE):
    """Generates the data and saves it to file."""
    examples = gen_easy_examples(num_examples)
    # save to file
    with open(save_file, "w") as f:
        for example in examples:
            f.write(example)


def build_easy_data_pipe() -> IterableWrapper:
    return build_data_pipe(FILE)


def easy_loop():
    # generate data
    num_examples = 400
    gen_easy_data(num_examples=num_examples, save_file=FILE)
    # build data pipe
    data_pipe_builder = build_data_pipe
    # create model
    model = GaussNet(
        embed_dim=64,
        dim_feedforward=512,
        num_heads=4,
        num_layers=4,
    )
    # create optimizer
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # define training loop
    num_epochs: int = 10000
    monitor_freq = 1000
    evaluation_freq = 10000
    save_file = "models/gauss-2var-easy-min.pt"
    evaluation_file = "data/easy_test_examples.txt"
    plotting_freq = 10000
    plot_file = f"plots/gauss-2var-easy-min-{num_epochs}-{num_examples}-{lr}.png"

    # train
    training_loop(
        data_pipe_builder=data_pipe_builder,
        model=model,
        optimizer=optimizer,
        num_epochs=num_epochs,
        monitor_freq=monitor_freq,
        evaluation_freq=evaluation_freq,
        save_file=save_file,
        evaluation_file=evaluation_file,
        plotting_freq=plotting_freq,
        plot_file=plot_file,
    )
