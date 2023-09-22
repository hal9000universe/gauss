#  Copyright (c) 2023. Benjamin Schoofs

import torch

from src.gauss_net.transformer import GaussNet
from src.gauss_net.evaluation import evaluate

from src.model_debug.easy_training import easy_loop


def evaluation(model: GaussNet):
    # load model
    state_dict = torch.load("models/gauss-2var-easy-min.pt")
    model.load_state_dict(state_dict)
    # evaluate
    accuracy = evaluate("data/easy_test_examples.txt", model)
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    easy_loop()


# Experiment 1: easy, lr: 0.001, batch_size: 64, num_examples: 100, num_epochs: 100
# model = create_gauss_net(
#     embed_dim=64,
#     dim_feedforward=512,
#     num_heads=4,
#     num_layers=4,
# )
# Accuracy: 0.668 (on the training set)

# Experiment 2: easy, lr: 0.001, batch_size: 64, num_examples: 1, num_epochs: 10000
# model = create_gauss_net(
#     embed_dim=64,
#     dim_feedforward=256,
#     num_heads=4,
#     num_layers=3,
# )
# Accuracy: 0.92 (on the training set), 0.08 (on the test set), 0.009 (on the test set without training)
