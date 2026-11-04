import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path

# hyperparameters
batch_size = 512
lr = 1e-3
epochs = 30
val_split = 0.1
max_train_samples = None  # set to a smaller int to speed up experiments
grad_clip = 1.0
hidden_dim = 64
message_dim = 2
seed = 7

torch.manual_seed(seed)
np.random.seed(seed)

# fully-connected (no self edges) for 4 nodes
EDGE_INDEX = torch.tensor([
    [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
    [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
], dtype=torch.long)


def load_data(data_path):
    data = np.load(data_path)
    X_train = torch.tensor(data['X_train'], dtype=torch.float32)
    y_train = torch.tensor(data['y_train'], dtype=torch.float32)
    X_test = torch.tensor(data['X_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.float32)
    times = torch.tensor(data['times'], dtype=torch.float32)
    return X_train, y_train, X_test, y_test, times


def build_dataset(X, y, edge_index):
    data_list = []
    for i in range(X.shape[0]):
        data_list.append(Data(x=X[i], edge_index=edge_index, y=y[i]))
    return data_list


def split_loaders(X_train, y_train, X_test, y_test, edge_index):
    if max_train_samples:
        X_train = X_train[:max_train_samples]
        y_train = y_train[:max_train_samples]

    dataset = build_dataset(X_train, y_train, edge_index)
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    test_set = build_dataset(X_test, y_test, edge_index)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, val_loader, test_loader


class SpringGNN(MessagePassing):
    def __init__(self, hidden_dim=64, message_dim=2):
        super().__init__(aggr='add')
        self.msg_mlp = nn.Sequential(
            nn.Linear(14, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim)
        )

    def forward(self, x, edge_index):
        pos = x[:, 0:2]
        vel = x[:, 2:4]
        force = self.propagate(edge_index, x=x, pos=pos, vel=vel)
        mass = x[:, 4:5].clamp(min=1e-4)
        return force / mass

    def message(self, x_i, x_j, pos_i, pos_j, vel_i, vel_j):
        rel_pos = pos_j - pos_i
        rel_vel = vel_j - vel_i
        inputs = torch.cat([x_i, x_j, rel_pos, rel_vel], dim=-1)
        return self.msg_mlp(inputs)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.L1Loss()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index)
        loss = loss_fn(pred, batch.y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_samples += batch.num_graphs
    return total_loss / max(1, total_samples)


def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.L1Loss()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index)
            loss = loss_fn(pred, batch.y)
            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs
    return total_loss / max(1, total_samples)


def rollout_trajectory(model, init_state, times, edge_index):
    # init_state: (4,5); times: (T,)
    device = torch.device('cpu')
    model_cpu = model.to(device).eval()
    masses = init_state[:, 4:5].detach().cpu()
    init = init_state[:, :4].detach().cpu()
    num_nodes = init.shape[0]

    def ode_func(t, flat_state):
        state = torch.tensor(flat_state, dtype=torch.float32).view(num_nodes, 4)
        pos = state[:, :2]
        vel = state[:, 2:4]
        node_feat = torch.cat([pos, vel, masses], dim=-1)
        with torch.no_grad():
            accel = model_cpu(node_feat, edge_index)
        deriv = torch.cat([vel, accel], dim=-1)
        return deriv.reshape(-1).numpy()

    sol = solve_ivp(
        ode_func,
        (float(times[0]), float(times[-1])),
        init.reshape(-1).numpy(),
        t_eval=times.numpy(),
        method='RK45'
    )
    traj = torch.tensor(sol.y.T, dtype=torch.float32).view(len(times), num_nodes, 4)
    return traj[:, :, :2]


def plot_trajectories(true_traj, pred_traj, title, out_path):
    colors = ['C0', 'C1', 'C2', 'C3']
    plt.figure(figsize=(6, 6))
    for i in range(true_traj.shape[1]):
        plt.plot(true_traj[:, i, 0], true_traj[:, i, 1], color=colors[i], label=f"true particle {i}")
        plt.plot(pred_traj[:, i, 0], pred_traj[:, i, 1], '--', color=colors[i], label=f"pred particle {i}")
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.legend()
    plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_losses(train_losses, val_losses, out_path):
    plt.figure()
    plt.plot(train_losses, label='train MAE')
    plt.plot(val_losses, label='val MAE')
    plt.xlabel('epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / 'spring_data.npz'
    X_train, y_train, X_test, y_test, times = load_data(data_path)

    train_loader, val_loader, test_loader = split_loaders(X_train, y_train, X_test, y_test, EDGE_INDEX)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpringGNN(hidden_dim=hidden_dim, message_dim=message_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    best_val = float('inf')
    ckpt_path = base_dir / 'spring_gnn.pt'

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
        print(f"Epoch {epoch:03d}: train MAE={train_loss:.4f} | val MAE={val_loss:.4f}")

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    test_mae = evaluate(model, test_loader, device)
    print(f"Test MAE: {test_mae:.4f}")

    plot_losses(train_losses, val_losses, base_dir / 'losses.png')

    nt = len(times)
    train_traj = X_train[:nt]
    test_traj = X_test[:nt]

    train_pred = rollout_trajectory(model, train_traj[0], times, EDGE_INDEX)
    test_pred = rollout_trajectory(model, test_traj[0], times, EDGE_INDEX)

    plot_trajectories(train_traj[:, :, :2], train_pred, 'Train trajectory vs prediction', base_dir / 'train_traj_pred.png')
    plot_trajectories(test_traj[:, :, :2], test_pred, 'Test trajectory vs prediction', base_dir / 'test_traj_pred.png')

    np.savez(
        base_dir / 'predictions.npz',
        train_true=train_traj.numpy(),
        test_true=test_traj.numpy(),
        train_pred=train_pred.numpy(),
        test_pred=test_pred.numpy(),
        times=times.numpy(),
    )


if __name__ == '__main__':
    main()
