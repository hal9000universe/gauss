#  Copyright (c) 2023. Benjamin Schoofs

from src.gauss_net.transformer import GaussNet
from src.data_engine.data_pipe import build_data_pipe


def evaluate(data_file: str, model: GaussNet) -> float:
    """Evaluate the model on the data.

    Args:
        data_file (str): data file
        model (GaussNet): model

    Returns:
        float: loss
    """
    # load data
    data_pipe = build_data_pipe(data_file)
    # evaluate
    num_correct = 0
    num_points = 0
    for (x, padding_mask), y in data_pipe:
        # calculate loss
        num_correct += model(x, src_key_padding_mask=padding_mask).argmax(dim=1).eq(y).sum().item()
        num_points += y.size(0)
    # return accuracy
    return num_correct / num_points
