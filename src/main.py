#  Copyright (c) 2023. Benjamin Schoofs

import torch

# from src.training.one_var_int_loop import one_var_int_loop

from src.training.evaluation import evaluation
from src.training.one_var_int_loop import gen_1var_int_data, generate_data_loader, choose_device
from src.models.transformer import MathFormer
from src.processing.tokenizer import fetch_tokenizer


def main():
    # set seed
    torch.manual_seed(0)
    # run
    # one_var_int_loop()

    device = choose_device()
    tokenizer = fetch_tokenizer("tokenizer/one_var_tokenizer.json")
    model = MathFormer(
        embed_dim=128,
        dim_feedforward=1024,
        num_heads=8,
        num_layers=10,
        tokenizer=tokenizer,
    ).to(device)
    gen_1var_int_data(100, "data/one_var/int/giga-gauss-2.pt")
    data_loader = generate_data_loader("data/one_var/int/giga-gauss-2.pt", tokenizer, 64, 0)
    evaluation(device, model, "models/one_var/int/giga-gauss-2.pt", data_loader)


if __name__ == "__main__":
    main()
