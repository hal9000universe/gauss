#  Copyright (c) 2023. Benjamin Schoofs

from typing import Optional, List, Callable

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

from src.gauss_net.transformer import GaussNet
from src.training.evaluation import evaluate

import matplotlib.pyplot as plt


def plot(x: List[float], y: List[float], title: str, xlabel: str, ylabel: str, save_file: Optional[str] = None):
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


def scatter_accuracy(x: List[float], y1: List[float], y2: List[float],
                     title: str, xlabel: str, ylabel: str, save_file: Optional[str] = None):
    """Create a scatter plot.

    Args:
        x (List[float]): x values
        y1 (List[float]): y values
        y2 (List[float]): y values
        title (str): title
        xlabel (str): x label
        ylabel (str): y label
        save_file (Optional[str], optional): file to save the plot to. Defaults to None.
    """
    plt.scatter(x, y1, label="Training", color="red")
    plt.scatter(x, y2, label="Test", color="blue")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    if save_file is not None:
        plt.savefig(save_file)


def comp_trail_avg_loss(loss_list: List[float]) -> float:
    """Computes the trailing average loss.

    Args:
        loss_list (List[float]): list of losses

    Returns:
        float: trailing average loss
    """
    if len(loss_list) > 50:
        return sum(loss_list[50:]) / len(loss_list[50:])
    else:
        return sum(loss_list) / len(loss_list)


def training_loop(
        accelerator: Accelerator,
        train_data_loader_builder: Callable[[], DataLoader],
        test_data_loader_builder: Callable[[], DataLoader],
        model: GaussNet,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 1000,
        monitor_freq: int = 100,
        evaluation_freq: int = 1000,
        save_file: Optional[str] = None,
        plotting_freq: int = 1000,
        plot_file: Optional[str] = None,
):
    """Training loop for the GaussNet.

    Args:
        accelerator (Accelerator): accelerator
        train_data_loader_builder (DataLoader): train data loader
        test_data_loader_builder (DataLoader): test data loader
        model (GaussNet): model
        optimizer (torch.optim.Optimizer): optimizer
        num_epochs (int, optional): number of epochs. Defaults to 1000.
        monitor_freq (int, optional): frequency of monitoring. Defaults to 100.
        evaluation_freq (int, optional): frequency of evaluation. Defaults to 1000.
        save_file (str, optional): file to save the model to. Defaults to None.
        plotting_freq (int, optional): frequency of plotting. Defaults to 1000.
        plot_file (str, optional): file to save the plot to. Defaults to None.
    """
    # train
    step = 0
    max_accuracy = 0.
    min_loss = float('inf')

    # plotting
    loss_list = []
    avg_loss_list = []
    test_accuracy_list = []
    train_accuracy_list = []
    loss_step_list = []
    accuracy_step_list = []

    # training loop
    for epoch in range(0, num_epochs):
        # build train data loader
        model, optimizer, train_data_loader = accelerator.prepare([model, optimizer, train_data_loader_builder()])
        # iterate over training batches
        for (x, padding_mask), y in train_data_loader:
            # reset gradients
            optimizer.zero_grad()
            # calculate loss
            loss = model(x, targets=y, src_key_padding_mask=padding_mask)
            # backpropagate
            accelerator.backward(loss)
            # update weights
            optimizer.step()

            # plotting
            loss_list.append(loss.item())
            loss_step_list.append(step)
            avg_loss = comp_trail_avg_loss(loss_list)
            avg_loss_list.append(avg_loss)

            # save minimal loss
            if loss.item() < 0.1 * min_loss:
                min_loss = loss.item()
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}, Average Loss: {avg_loss}")
                torch.save(model.state_dict(), save_file)
                print(f"Model with minimal loss saved to {save_file}")

            # monitor
            if step % monitor_freq == 0:
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}, Average Loss: {avg_loss}")

            # evaluation
            if step % evaluation_freq == 0:
                # build train & test data loader
                model, optimizer, train_data_loader, test_data_loader = accelerator.prepare(
                    [model,
                     optimizer,
                     train_data_loader_builder(),
                     test_data_loader_builder(),
                     ]
                )
                # evaluate
                test_accuracy = evaluate(model, test_data_loader)
                train_accuracy = evaluate(model, train_data_loader)
                print(f"Epoch: {epoch}, Step: {step}, Test Accuracy: {test_accuracy}, "
                      f"Training Accuracy: {train_accuracy}")

                # plotting
                test_accuracy_list.append(test_accuracy)
                train_accuracy_list.append(train_accuracy)
                accuracy_step_list.append(step)

                # accuracy plot
                file = plot_file.replace(".png", f"-Accuracy.png")
                scatter_accuracy(accuracy_step_list, test_accuracy_list, train_accuracy_list,
                                 "Accuracy", "Step", "Accuracy", file)

                # save model
                if test_accuracy > max_accuracy:
                    torch.save(model.state_dict(), save_file)
                    print(f"Model with maximal test_accuracy saved to {save_file}")
                    max_accuracy = test_accuracy

            # plot
            if step % plotting_freq == 0:
                # average loss plot
                file = plot_file.replace(".png", f"-Average-Loss.png")
                plot(loss_step_list, avg_loss_list, "Average Loss", "Step", "Average Loss", file)

            # increment step
            step += 1
        print(f"Epoch {epoch} finished.")

    # save model
    torch.save(model.state_dict(), save_file)
    print(f"Model saved to {save_file}")

    # loss plot
    loss_plot_file = plot_file.replace(".png", f"-Averager-Loss.png")
    plot(loss_step_list, avg_loss_list, "Average Loss", "Step", "Loss", loss_plot_file)

    # test_accuracy plot
    accuracy_plot_file = plot_file.replace(".png", f"-Accuracy.png")
    scatter_accuracy(accuracy_step_list, test_accuracy_list, train_accuracy_list,
                     "Accuracy", "Step", "Accuracy", accuracy_plot_file)
