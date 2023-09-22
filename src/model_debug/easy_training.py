#  Copyright (c) 2023. Benjamin Schoofs

import torch
from typing import List
from torchdata.datapipes.iter import IterableWrapper

from src.data_engine.solve import gaussian_elimination
from src.data_engine.repr import repr_record
from src.data_engine.data_pipe import build_data_pipe
from src.gauss_net.transformer import create_gauss_net
from src.gauss_net.training import training_loop


# set seed
torch.manual_seed(0)


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


def gen_easy_data(num_examples: int = 100):
    """Generates the data and saves it to file."""
    examples = gen_easy_examples(num_examples)
    # save to file
    with open(FILE, "w") as f:
        for example in examples:
            f.write(example)


def build_easy_data_pipe() -> IterableWrapper:
    return build_data_pipe(FILE)


def easy_loop():
    # generate data
    gen_easy_data(num_examples=200)
    # data_pipe_builder
    data_pipe_builder = build_easy_data_pipe
    # create model
    model = create_gauss_net(
        embed_dim=64,
        dim_feedforward=512,
        num_heads=4,
        num_layers=5,
    )
    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # define training loop
    num_epochs: int = 50000
    monitor_freq = 5
    evaluation_freq = 10000
    # train
    training_loop(data_pipe_builder, model, optimizer, num_epochs, monitor_freq, evaluation_freq)
