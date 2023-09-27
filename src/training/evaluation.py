#  Copyright (c) 2023. Benjamin Schoofs

import torch
from torch.utils.data import DataLoader

from src.gauss_net.transformer import GaussNet


def evaluate(model: GaussNet, data_loader: DataLoader) -> float:
    """Evaluate the model on the data.

    Args:
        model (GaussNet): model
        data_loader (DataLoader): data loader

    Returns:
        float: loss
    """
    # evaluate
    num_correct = 0
    num_points = 0
    for (x, padding_mask), y in data_loader:
        # calculate loss
        num_correct += model(x, src_key_padding_mask=padding_mask).argmax(dim=1).eq(y).sum().item()
        num_points += y.size(0)
    # return accuracy
    return num_correct / num_points


def evaluation(model: GaussNet, load_file: str, data_loader: DataLoader):
    """Evaluate the model.

    Args:
        model (GaussNet): model
        load_file (str): file to load the model from
        data_loader (DataLoader): data loader
    """
    # load model
    state_dict = torch.load(load_file)
    model.load_state_dict(state_dict)
    # evaluate
    accuracy = evaluate(model, data_loader)
    print(f"Accuracy: {accuracy}")
