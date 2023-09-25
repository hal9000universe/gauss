#  Copyright (c) 2023. Benjamin Schoofs

import torch

from src.gauss_net.transformer import GaussNet
from src.data_engine.data_pipe import build_data_pipe


def evaluate(model: GaussNet, data_file: str, batch_size: int) -> float:
    """Evaluate the model on the data.

    Args:
        model (GaussNet): model
        data_file (str): data file
        batch_size (int): batch size

    Returns:
        float: loss
    """
    # load data
    data_pipe = build_data_pipe(model.get_tokenizer(), data_file, batch_size)
    # evaluate
    num_correct = 0
    num_points = 0
    for (x, padding_mask), y in data_pipe:
        # calculate loss
        num_correct += model(x, src_key_padding_mask=padding_mask).argmax(dim=1).eq(y).sum().item()
        num_points += y.size(0)
    # return accuracy
    return num_correct / num_points


def evaluation(model: GaussNet, load_file: str, data_file: str):
    # load model
    state_dict = torch.load(load_file)
    model.load_state_dict(state_dict)
    # evaluate
    accuracy = evaluate(model, data_file, 10)
    print(f"Accuracy: {accuracy}")
