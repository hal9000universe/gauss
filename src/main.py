#  Copyright (c) 2023. Benjamin Schoofs

import torch

from src.training.one_var_int_loop import one_var_int_loop

from accelerate import notebook_launcher


def main():
    # set seed
    torch.manual_seed(0)
    # run
    notebook_launcher(one_var_int_loop, num_processes=1)


if __name__ == "__main__":
    main()


# Experiment 1: easy, lr: 0.001, batch_size: 64, num_examples: 100, num_epochs: 100
# model = GaussNet(
#     embed_dim=64,
#     dim_feedforward=512,
#     num_heads=4,
#     num_layers=4,
# )
# Accuracy: 0.668 (on the training set)

# Experiment 2: easy, lr: 0.001, batch_size: 64, num_examples: 1, num_epochs: 10000
# model = GaussNet(
#     embed_dim=64,
#     dim_feedforward=256,
#     num_heads=4,
#     num_layers=3,
# )
# Accuracy: 0.92 (on the training set), 0.08 (on the test set), 0.009 (on the test set without training)

# Experiment 3: easy, lr: 0.001, batch_size: 64, num_examples: 200, num_epochs: 200
# model = GaussNet(
#     embed_dim=64,
#     dim_feedforward=512,
#     num_heads=4,
#     num_layers=5,
# )
# Accuracy: 0.70 (on the training set)

# Experiment 4: easy, lr: 0.001, batch_size: 64, num_examples: 10000, num_epochs: 500
# model = GaussNet(
#     embed_dim=64,
#     dim_feedforward=512,
#     num_heads=4,
#     num_layers=5,
# )
# Accuracy: 0.83 (on the training set)
