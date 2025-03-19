#  Copyright (c) 2023. Benjamin Schoofs

import os
from typing import List

import torch

from src.data_engine.one_var_solve import solve, Equation
from src.processing.tokenizer import fetch_tokenizer
from src.training.training import training_loop
from src.models.transformer import MathFormer
from src.processing.data_loader import generate_data_loader


def gen_1var_int_examples(n: int) -> List[str]:
    """Generates 1 variable equations.

    Args:
        n (int): number of examples to generate

    Returns:
        List[str]: list of examples
    """
    examples: List[str] = []
    for i in range(0, n):
        equation = Equation.generate_rand_int_equation(20, -100, 100)
        solution, record = solve(equation)
        example = str(record)
        examples.append(example)

    return examples


def gen_1var_int_data(num_examples: int, save_file: str):
    """Generates the data and saves it to file.

    Args:
        num_examples (int): number of examples to generate
        save_file (str): file to save the examples to
    """
    examples = gen_1var_int_examples(num_examples)
    # save to file
    with open(save_file, "w") as f:
        for example in examples:
            f.write(example)


def choose_device() -> torch.device:
    """Chooses the device to use.

    Returns:
        torch.device: device to use
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def one_var_int_loop():
    """One variable integer loop for the GaussNet.

    Training on 10000 examples of Gaussian elimination with a large GaussNet."""
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
    # make directories
    if not os.path.exists("models/one_var/int"):
        os.makedirs("models/one_var/int")
    if not os.path.exists("plots/one_var/int"):
        os.makedirs("plots/one_var/int")
    if not os.path.exists("data/one_var/int"):
        os.makedirs("data/one_var/int")
<<<<<<< Updated upstream

    # set up
    train_data_file: str = "data/one_var/int/train_equations.txt"
    test_data_file: str = "data/one_var/int/test_equations.txt"
    num_examples: int = 6000
    num_test_examples: int = 10
    batch_size: int = 128
    num_epochs: int = 1000
    num_workers: int = 0

=======
    # set up
    train_data_file: str = "data/one_var/int/train_equations.txt"
    test_data_file: str = "data/one_var/int/test_equations.txt"
    num_examples: int = 1000
    batch_size: int = 64
    num_epochs: int = 100
>>>>>>> Stashed changes
    # generate data
    gen_1var_int_data(num_examples, train_data_file)
    gen_1var_int_data(num_test_examples, test_data_file)

    # load tokenizer
    tokenizer = fetch_tokenizer("tokenizer/one_var_tokenizer.json")

    # build data loaders
    train_data_loader = generate_data_loader(train_data_file, tokenizer, batch_size, num_workers)
    test_data_loader = generate_data_loader(test_data_file, tokenizer, batch_size, num_workers)
    train_eval_data_loader = generate_data_loader(train_data_file, tokenizer, batch_size, num_workers)

    # define save files
    model_save_file = f"models/one_var/int/giga-gauss-2.pt"
    opt_save_file = f"models/one_var/int/giga-gauss-opt-2.pt"

    # set up device
    device = choose_device()
    # build model
    model = MathFormer(
        embed_dim=128,
        dim_feedforward=1024,
        num_heads=8,
        num_layers=10,
        tokenizer=tokenizer,
    ).to(device)
    # load model
    if os.path.exists(model_save_file):
        try:
            model.load_state_dict(torch.load(model_save_file))
            print("Loaded model.")
        except RuntimeError:
            print("Could not load model. Starting from scratch.")
    # create optimizer
    lr = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # load optimizer
    if os.path.exists(opt_save_file):
        try:
            optimizer.load_state_dict(torch.load(opt_save_file))
            print("Loaded optimizer.")
        except RuntimeError:
            print("Could not load optimizer. Starting from scratch.")

    # define training loop
    monitor_freq = 100
    evaluation_freq = 10000
    plotting_freq = 2000
    plot_file = f"plots/one_var/int/{num_epochs}-{num_examples}-{lr}.png"
    save_freq = 10000

    # train model
    training_loop(
        device=device,
        train_data_loader=train_data_loader,
        test_data_loader=test_data_loader,
        train_eval_data_loader=train_eval_data_loader,
        model=model,
        optimizer=optimizer,
        num_epochs=num_epochs,
        monitor_freq=monitor_freq,
        evaluation_freq=evaluation_freq,
        model_save_file=model_save_file,
        opt_save_file=opt_save_file,
        plotting_freq=plotting_freq,
        plot_file=plot_file,
        save_freq=save_freq,
    )
