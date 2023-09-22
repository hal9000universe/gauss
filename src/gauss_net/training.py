#  Copyright (c) 2023. Benjamin Schoofs

from typing import Callable

import torch
from torchdata.datapipes.iter import IterableWrapper

from src.data_engine.data_pipe import build_data_pipe
from src.gauss_net.transformer import create_gauss_net, GaussNet
from src.gauss_net.evaluation import evaluate


def training_loop(data_pipe_builder: Callable[[], IterableWrapper], model: GaussNet, optimizer: torch.optim.Optimizer,
                  num_epochs: int = 1000, monitor_freq: int = 100, evaluation_freq: int = 1000):
    """Training loop for the GaussNet.

    Args:
        data_pipe_builder (Callable[[], IterableWrapper]): data pipe
        model (GaussNet): model
        optimizer (torch.optim.Optimizer): optimizer
        num_epochs (int, optional): number of epochs. Defaults to 1000.
        monitor_freq (int, optional): frequency of monitoring. Defaults to 100.
        evaluation_freq (int, optional): frequency of evaluation. Defaults to 1000.
    """
    # train
    step = 0
    for epoch in range(0, num_epochs):
        data_pipe = data_pipe_builder()
        for (x, padding_mask), y in data_pipe:
            # reset gradients
            optimizer.zero_grad()
            # calculate loss
            loss = model(x, targets=y, src_key_padding_mask=padding_mask)
            # backpropagate
            loss.backward()
            # update weights
            optimizer.step()
            # monitor
            if step % monitor_freq == 0:
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")
            # evaluate
            if step % evaluation_freq == 0:
                accuracy = evaluate("data/easy_examples.txt", model)
                print(f"Epoch: {epoch}, Step: {step}, Accuracy: {accuracy}")
            # increment step
            step += 1
        print(f"Epoch {epoch} finished.")
    # save model
    save_file = "models/gauss.pt"
    torch.save(model.state_dict(), save_file)
    print(f"Model saved to {save_file}")


def main_loop():
    # build data pipe
    data_pipe_builder = build_data_pipe
    # create model
    model = create_gauss_net()
    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # define training loop
    num_epochs: int = 10000
    monitor_freq = 100
    # train
    training_loop(data_pipe_builder, model, optimizer, num_epochs, monitor_freq)
