import torch
from torch import nn
import torchdiffeq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=0))


def train(model, data, time, epochs=1000, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data_mean = data.mean(dim=0)
    data_std = data.std(dim=0) + 1e-8
    data_normalized = (data - data_mean) / data_std

    # we know velocity starts at 0
    v_0 = torch.zeros_like(data_normalized[:, 0])

    for epoch in range(epochs):
        optimizer.zero_grad()

        pred = torchdiffeq.odeint(model, v_0, time)
        loss = torch.mean((pred - data_normalized) ** 2)

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} | Loss: {loss.item()}")



if __name__ == "__main__":
    # Load data
    data = pd.read_table("AOog2uHlKxVa.txt", delimiter= " ").to_numpy().astype(np.float32)
    print(data)
    time = data[:, 0]  # time column
    data = data[:, 1:]  # drop time column
    data = torch.from_numpy(data)
    time = torch.from_numpy(time)

    model = ODEFunc()
    train(model, data, time)