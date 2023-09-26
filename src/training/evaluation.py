#  Copyright (c) 2023. Benjamin Schoofs

from typing import Callable

import torch
from torchdata.datapipes.iter import IterableWrapper

from src.gauss_net.transformer import GaussNet


def evaluate(model: GaussNet, data_pipe_builder: Callable[[], IterableWrapper]) -> float:
    """Evaluate the model on the data.

    Args:
        model (GaussNet): model
        data_pipe_builder (Callable[[], IterableWrapper]): data pipe builder

    Returns:
        float: loss
    """
    # load data
    data_pipe = data_pipe_builder()
    # evaluate
    num_correct = 0
    num_points = 0
    for (x, padding_mask), y in data_pipe:
        # calculate loss
        num_correct += model(x, src_key_padding_mask=padding_mask).argmax(dim=1).eq(y).sum().item()
        num_points += y.size(0)
    # return accuracy
    return num_correct / num_points


def evaluation(model: GaussNet, load_file: str, data_pipe_builder: Callable[[], IterableWrapper]):
    """Evaluate the model.

    Args:
        model (GaussNet): model
        load_file (str): file to load the model from
        data_pipe_builder (Callable[[], IterableWrapper]): data pipe builder
    """
    # load model
    state_dict = torch.load(load_file)
    model.load_state_dict(state_dict)
    # evaluate
    accuracy = evaluate(model, data_pipe_builder)
    print(f"Accuracy: {accuracy}")
