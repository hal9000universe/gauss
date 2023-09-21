#  Copyright (c) 2023. Benjamin Schoofs

import torch

from src.data_engine.data_pipe import build_data_pipe
from src.gauss_net.transformer import create_gauss_net


def training_loop(num_epochs: int = 2):
    # build data pipe
    data_pipe = build_data_pipe()
    # create model
    model = create_gauss_net()
    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # define training loop
    monitor_freq = 10
    # train
    for epoch in range(0, num_epochs):
        for (x, padding_mask), y in data_pipe:
            # reset gradients
            optimizer.zero_grad()
            # calculate loss
            loss = model(x, targets=y, src_key_padding_mask=padding_mask)
            # backpropagate
            loss.backward()
            # update weights
            optimizer.step()
            if epoch % monitor_freq == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")
    # save model
    save_file = "model.pt"
    torch.save(model.state_dict(), save_file)
    print(f"Model saved to {save_file}")

