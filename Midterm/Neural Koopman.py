import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


class Neural_Koopman(nn.Module):
    def __init__(self, ny, nz):
        super(Neural_Koopman, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(ny, 10),
            nn.LeakyReLU(),
            nn.Linear(10, nz)
        )
        self.decoder = nn.Sequential(
            nn.Linear(nz, 10),
            nn.LeakyReLU(),
            nn.Linear(10, ny)
        )

        self.K = nn.Parameter(torch.eye(nz) + torch.randn(nz, nz) * 0.01)

    def encode(self, y):
        return self.encoder(y)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, y):
        is_2d = (y.dim() == 2)
        if is_2d:
            y = y.unsqueeze(1)  # (ntraj, ny) -> (ntraj, 1, ny)

        z = self.encode(y)
        z_next = torch.matmul(z, self.K)
        y_next = self.decode(z_next)

        if is_2d:
            y_next = y_next.squeeze(1)  # (ntraj, 1, ny) -> (ntraj, ny)

        return y_next


def koopman_loss(Model, Y):
    # Y shape: (nt, ny)
    z = Model.encode(Y[:-1])  # encode current states: (nt-1, nz)

    z_next = torch.matmul(z, Model.K)  # predicted next latent state: (nt-1, nz)
    z_next_true = Model.encode(Y[1:])  # encode actual next states: (nt-1, nz)

    MSE = torch.sqrt(torch.mean((z_next - z_next_true) ** 2))  # how wrong were we?
    return MSE


def midterm_loss(Model, Y):
    z = Model.encode(Y)  # encode stuff
    z_std = torch.std(z, dim=0)
    z_mean = torch.mean(z, dim=0)

    return torch.mean((1 - z_std) ** 2 + z_mean ** 2)


def reconstruction_loss(Model, Y):
    z = Model.encode(Y)  # encode stuff
    y_hat = Model.decode(z)  # decode stuff

    MSE = torch.sqrt(torch.mean((Y - y_hat) ** 2)) # how wrong were we?

    return MSE


def prediction_loss(Model, Y):
    # Y shape: (nt, ny)
    y_pred = []
    y_current = Y[0:1]  # first time step: (1, ny)

    for t in range(1, Y.shape[0]):
        y_next = Model(y_current)  # predict next state
        y_pred.append(y_next)
        y_current = y_next  # update current state

    y_pred = torch.cat(y_pred, dim=0)  # (nt-1, ny)
    y_true = Y[1:]  # true future states: (nt-1, ny)

    MSE = torch.sqrt(torch.mean((y_pred - y_true) ** 2))  # how wrong were we?

    return MSE


def train_up(model, epochs, Ytrain, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    nn.Dropout(p=0.1)
    a1, a2, a3, a4 = 1.0, 1.0, 1.0, 3.0  # weights for the losses

    losses = {"k_loss": [], "r_loss": [], "p_loss": [],"m_loss": [], "big daddy loss": []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        k_loss = koopman_loss(model, Ytrain)
        losses['k_loss'].append(k_loss.item())

        r_loss = reconstruction_loss(model, Ytrain)
        losses['r_loss'].append(r_loss.item())

        p_loss = prediction_loss(model, Ytrain)
        losses['p_loss'].append(p_loss.item())

        m_loss = midterm_loss(model, Ytrain)
        losses['m_loss'].append(m_loss.item())

        loss = a1*k_loss + a2*r_loss + a3*p_loss + a4*m_loss
        losses["big daddy loss"].append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            print(f'Epoch {epoch}, Train Loss: {loss.item():.4f}')

    return losses


def normalize_data(Ytrain, Ytest, eps=1e-8):
    # Compute mean and std over trajectories and time
    mu = Ytrain.mean(axis=(0, 1), keepdims=True)
    sigma = Ytrain.std(axis=(0, 1), keepdims=True) + eps

    Ytrain_norm = (Ytrain - mu) / sigma
    Ytest_norm  = (Ytest  - mu) / sigma

    stats = {"mu": mu, "sigma": sigma}

    return Ytrain_norm, Ytest_norm, stats


def denormalize(Y, stats):
    return Y * stats["sigma"] + stats["mu"]


def data():
    # training data
    nt = 30
    xtrain = torch.zeros(nt, 1)
    xtrain[0] = 0.5
    for i in range(1, nt):
        xtrain[i] = 3.7 * xtrain[i - 1] * (1 - xtrain[i - 1])

    return xtrain

if __name__ == "__main__":
    xTrain = data()

    model = Neural_Koopman(1, 3)
    losses = train_up(model, 10000, xTrain, lr=5e-4)

    for loss_name in losses:
        plt.plot(losses[loss_name], label=loss_name)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()

    print(f"eigenvalues of K: {torch.linalg.eigvals(model.K).detach().numpy()}")


