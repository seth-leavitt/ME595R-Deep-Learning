import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

def load_data():
    ntraj = 2148  # number of trajectories
    nt = 50  # number of time steps
    ny = 7  # number of states

    tvec = np.linspace(0, 350, nt)
    Y = np.loadtxt('kdata.txt').reshape(ntraj, nt, ny)
    Ytrain = Y[:1000, :, :]  # 2048 training trajectories # for now just 100
    Ytest = Y[1000:1100, :, :]  # 100 testing trajectories

    return Ytrain, Ytest, tvec


class Neural_Koopman(nn.Module):
    def __init__(self, ny, nz):
        super(Neural_Koopman, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(ny, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, nz)
        )
        self.decoder = nn.Sequential(
            nn.Linear(nz, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, ny)
        )

        self.K = nn.Parameter(torch.eye(nz) + torch.randn(nz, nz) * 0.01)

    def encode(self, y):
        ntraj, nt, ny = y.shape
        y_flattened = y.reshape(ntraj*nt, ny)
        z_flat = self.encoder(y_flattened)
        z = z_flat.reshape(ntraj, nt, -1)
        return z

    def decode(self, z):
        ntraj, nt, nz = z.shape
        z_flattened = z.reshape(ntraj*nt, nz)
        y_flat = self.decoder(z_flattened)
        y = y_flat.reshape(ntraj, nt, -1)
        return y

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
    z = Model.encode(Y)  # encode stuff
    z_next = torch.matmul(z[:, :-1, :], Model.K)  # apply Koopman operator to get next latent state
    z_next_true = Model.encode(Y[:, 1:, :])  # encode the actual answer

    MSE = torch.sqrt(torch.mean((z_next - z_next_true) ** 2)) # how wrong were we?
    return MSE


def reconstruction_loss(Model, Y):
    z = Model.encode(Y)  # encode stuff
    y_hat = Model.decode(z)  # decode stuff

    MSE = torch.sqrt(torch.mean((Y - y_hat) ** 2)) # how wrong were we?

    return MSE


def prediction_loss(Model, Y):
    y_pred = []
    y_current = Y[:, 0, :]  # initial state

    for t in range(1, Y.shape[1]):
        y_next = Model(y_current)  # predict next state
        y_pred.append(y_next)
        y_current = y_next  # update current state

    y_pred = torch.stack(y_pred, dim=1)  # ntraj x (nt-1) x ny
    y_true = Y[:, 1:, :]  # true future states

    MSE = torch.sqrt(torch.mean((y_pred - y_true) ** 2)) # how wrong were we?

    return MSE


def train_up(model, epochs, Ytrain, Ytest, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    nn.Dropout(p=0.1)
    a1, a2, a3 = 1.0, 1.0, 1.0  # weights for the losses

    losses = {"k_loss": [], "r_loss": [], "p_loss": [], "big daddy loss": []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        k_loss = koopman_loss(model, Ytrain)
        losses['k_loss'].append(k_loss.item())

        r_loss = reconstruction_loss(model, Ytrain)
        losses['r_loss'].append(r_loss.item())

        p_loss = prediction_loss(model, Ytrain)
        losses['p_loss'].append(p_loss.item())
        if epoch > 400:
            loss = a1*k_loss + a2*r_loss + a3*p_loss
            losses["big daddy loss"].append(loss.item())
        elif epoch > 200:
            loss = a1*k_loss + a2*r_loss
            losses["big daddy loss"].append(loss.item())
        else:
            loss = r_loss
            losses["big daddy loss"].append(loss.item())


        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = koopman_loss(model, Ytest) + reconstruction_loss(model, Ytest) + prediction_loss(model, Ytest)
            print(f'Epoch {epoch}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

    return losses


def fancy_pants_plot(model, y_test, stats):
    plt.cla()
    with torch.no_grad():
        # True trajectory in original units
        real = torch.tensor(denormalize(y_test.numpy(), stats), dtype=torch.float32)[:1, :, :3]

        # Roll out predictions in normalized space
        y_hat = []
        y_current = y_test[:1, 0, :]
        for t in range(1, y_test.shape[1]):
            y_next = model(y_current)
            y_hat.append(y_next)
            y_current = y_next

        y_hat = torch.stack(y_hat, dim=1)  # (1, nt-1, ny)

        # Denormalize predictions for plotting
        y_hat_denorm = torch.tensor(denormalize(y_hat.numpy(), stats), dtype=torch.float32)[:, :, :3]

        # Align time lengths: real has nt points, pred has nt-1
        for i in range(3):
            plt.plot(real[0, :, i], label=f"Real State {i+1}")
            plt.plot(range(1, real.shape[1]), y_hat_denorm[0, :, i], linestyle="dashed",
                     label=f"Predicted State {i+1}")
    plt.xlabel("Time step")
    plt.ylabel("State Value")
    plt.title("Real vs Predicted Trajectories (first 3 states)")
    plt.legend()
    plt.grid()
    plt.show()


def normalize_data(Ytrain, Ytest, eps=1e-8):
    # Compute mean and std over trajectories and time
    mu = Ytrain.mean(axis=(0, 1), keepdims=True)   # shape (1,1,ny)
    sigma = Ytrain.std(axis=(0, 1), keepdims=True) + eps

    Ytrain_norm = (Ytrain - mu) / sigma
    Ytest_norm  = (Ytest  - mu) / sigma

    stats = {"mu": mu, "sigma": sigma}

    return Ytrain_norm, Ytest_norm, stats


def denormalize(Y, stats):
    return Y * stats["sigma"] + stats["mu"]


if __name__ == "__main__":
    Ytrain, Ytest, tvec = load_data()
    Ytrain, Ytest, stats = normalize_data(Ytrain, Ytest)

    model = Neural_Koopman(7, 28)
    losses = train_up(model, 1500, torch.tensor(Ytrain, dtype=torch.float32), torch.tensor(Ytest, dtype=torch.float32), lr=5e-4)

    for loss_name in losses:
        plt.plot(losses[loss_name], label=loss_name)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()

    fancy_pants_plot(model, torch.tensor(Ytest, dtype=torch.float32), stats)
