import numpy as np
import torch as t
from torch import nn
import matplotlib.pyplot as plt

# Model
class PINN(nn.Module):
    def __init__(self, hlayers, width):
        super(PINN, self).__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]

        for _ in range(hlayers - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(width, 1))
        self.linear_stack = nn.Sequential(*layers)

    def forward(self, model_input, t):
        return self.linear_stack(model_input)


def boundary_condition_residual(model):
    # BC: u(0) = 1, u(5) = 0
    x_left = 0
    x_right = 5

    y_hat = model(t.tensor([[x_left], [x_right]], dtype=t.float32))  # [2,1]
    u_left = y_hat[0, 0]  # scalar
    u_right = y_hat[1, 0]  # scalar

    return (1 - u_left) + (0 - u_right)  # residuals for both BCs


def physics_residual(model, x,):
    #ODE: d2ydx2 - 2*dy/dx + y = cos(x)
    u = model(x)  # [N,1]
    u_x = t.autograd.grad(u, x, grad_outputs=t.ones_like(u), create_graph=True)[0]  # [N,1]
    u_xx = t.autograd.grad(u_x, x, grad_outputs=t.ones_like(u_x), create_graph=True)[0]  # [N,1]

    f = u_xx - 2 * u_x + u - t.cos(x)  # [N,1]
    return f

def plot_solution(model):
    x = t.linspace(0, 5, 100).reshape(-1, 1)  # [100,1]
    with t.no_grad():
        y_pred = model(x).numpy()  # [100,1]
    plt.style.use('dark_background')
    plt.plot(x.numpy(), y_pred, label='PINN Solve', color = "fuchsia")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.suptitle('PINN Solution to ODE')
    plt.title("(I used a black background to make it look cooler and because I had extra time", fontsize=8)
    plt.legend()
    plt.show()
    plt.savefig("solution.jpeg")

def train(model, epochs, lr):
    optimizer = t.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Sample collocation points for physics residual
        x_phys = t.linspace(0, 5, 100).reshape(-1, 1).requires_grad_(True)  # [100,1]
        f_residual = physics_residual(model, x_phys)  # [100,1]

        bc_residual = boundary_condition_residual(model)  # scalar

        loss = f_residual.pow(2).mean() + bc_residual.pow(2)  # MSE loss
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')


if __name__ == "__main__":
    # Test model
    model = PINN(2, 20)
    train(model, epochs=1000, lr=1e-3)
    plot_solution(model)