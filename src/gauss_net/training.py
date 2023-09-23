#  Copyright (c) 2023. Benjamin Schoofs

from typing import Callable, Optional, List

import torch
from torchdata.datapipes.iter import IterableWrapper

from src.data_engine.data_pipe import build_data_pipe, gen_data
from src.gauss_net.transformer import GaussNet
from src.gauss_net.evaluation import evaluate

import matplotlib.pyplot as plt


def create_plot(x: List[float], y: List[float], title: str, xlabel: str, ylabel: str, save_file: Optional[str] = None):
    """Create a plot.

    Args:
        x (List[float]): x values
        y (List[float]): y values
        title (str): title
        xlabel (str): x label
        ylabel (str): y label
        save_file (Optional[str], optional): file to save the plot to. Defaults to None.
    """
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    if save_file is not None:
        plt.savefig(save_file)


def training_loop(
        data_pipe_builder: Callable[[], IterableWrapper],
        model: GaussNet,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 1000,
        monitor_freq: int = 100,
        evaluation_freq: int = 1000,
        save_file: Optional[str] = None,
        evaluation_file: Optional[str] = None,
        plotting_freq: int = 1000,
        plot_file: Optional[str] = None,
):
    """Training loop for the GaussNet.

    Args:
        data_pipe_builder (Callable[[], IterableWrapper]): data pipe
        model (GaussNet): model
        optimizer (torch.optim.Optimizer): optimizer
        num_epochs (int, optional): number of epochs. Defaults to 1000.
        monitor_freq (int, optional): frequency of monitoring. Defaults to 100.
        evaluation_freq (int, optional): frequency of evaluation. Defaults to 1000.
        save_file (str, optional): file to save the model to. Defaults to None.
        evaluation_file (str, optional): file to evaluate the model on. Defaults to None.
        plotting_freq (int, optional): frequency of plotting. Defaults to 1000.
        plot_file (str, optional): file to save the plot to. Defaults to None.
    """
    # train
    step = 0
    max_accuracy = 0.
    min_loss = float('inf')

    # plotting
    loss_list = []
    accuracy_list = []
    loss_step_list = []
    accuracy_step_list = []

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

            # plotting
            loss_list.append(loss.item())
            loss_step_list.append(step)

            # save minimal loss
            if loss.item() < 0.5 * min_loss:
                min_loss = loss.item()
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")
                torch.save(model.state_dict(), save_file)
                print(f"Model with minimal loss saved to {save_file}")

            # monitor
            if step % monitor_freq == 0:
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")

            # evaluate
            if step % evaluation_freq == 0:
                accuracy = evaluate(model, evaluation_file)
                print(f"Epoch: {epoch}, Step: {step}, Accuracy: {accuracy}")

                # plotting
                accuracy_list.append(accuracy)
                accuracy_step_list.append(step)

                # save model
                if accuracy > max_accuracy:
                    torch.save(model.state_dict(), save_file)
                    print(f"Model with maximal accuracy saved to {save_file}")
                    max_accuracy = accuracy

            # plot
            if step % plotting_freq == 0:
                # loss plot
                file = plot_file.replace(".png", f"-Loss-{step}.png")
                create_plot(loss_step_list, loss_list, "Loss", "Step", "Loss", file)

                # accuracy plot
                file = plot_file.replace(".png", f"-Accuracy-{step}.png")
                create_plot(accuracy_step_list, accuracy_list, "Accuracy", "Step", "Accuracy", file)

            # increment step
            step += 1
        print(f"Epoch {epoch} finished.")

    # save model
    torch.save(model.state_dict(), save_file)
    print(f"Model saved to {save_file}")

    # loss plot
    create_plot(loss_step_list, loss_list, "Loss", "Step", "Loss", plot_file)

    # accuracy plot
    create_plot(accuracy_step_list, accuracy_list, "Accuracy", "Step", "Accuracy", plot_file)


def main_loop():
    # generate data
    num_examples = 10000
    gen_data(num_examples=num_examples, save_file="data/examples.txt")
    # build data pipe
    data_pipe_builder = build_data_pipe
    # create model
    model = GaussNet(
        embed_dim=64,
        dim_feedforward=512,
        num_heads=8,
        num_layers=10,
    )
    # create optimizer
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # define training loop
    num_epochs: int = 50000
    monitor_freq = 1000
    evaluation_freq = 10000
    save_file = "models/big-gauss.pt"
    evaluation_file = "data/easy_test_examples.txt"
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
        save_file=save_file,
        evaluation_file=evaluation_file,
        plotting_freq=plotting_freq,
        plot_file=plot_file,
    )
