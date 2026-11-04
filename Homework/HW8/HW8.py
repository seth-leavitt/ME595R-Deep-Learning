import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from torch import nn

torch.manual_seed(1021)  # for reproducibility


class SuperRes(nn.Module):
    def __init__(self, hlayers, height, width, kernel_size):
        super(SuperRes, self).__init__()
        layers = [nn.Conv2d(2, width, kernel_size, stride=1, padding=2), nn.LeakyReLU()]
        for _ in range(hlayers - 1):
            layers.append(nn.Conv2d(width, width, kernel_size, stride=1, padding=2))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(width, 3, kernel_size, stride=1, padding=2))
        self.conv_stack = nn.Sequential(*layers)
        self.upsample = nn.Upsample(size=(75, 47), mode="bicubic", align_corners=True)

    def forward(self, model_input):
        x = self.upsample(model_input)
        x = self.conv_stack(x)
        return x

    def train_model(self, lf_input, jinv, dxdxi, dxdeta, dydxi, dydeta, h, epochs=5000, lr=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []

        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = super_res_loss(self, lf_input, jinv, dxdxi, dxdeta, dydxi, dydeta, h)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if epoch % 100 == 0:
                print(f"epoch {epoch}/{epochs} | loss: {loss.item():.6f}")

        return losses

def apply_bc(uvp):
    pad = nn.ConstantPad2d((1, 1, 1, 1), 0.0)
    uvp = pad(uvp)

    u = uvp[:, 0:1, :, :]
    v = uvp[:, 1:2, :, :]
    p = uvp[:, 2:3, :, :]

    u[:, :, 0, :] = 0.0
    v[:, :, 0, :] = 1.0
    p[:, :, 0, :] = p[:, :, 1, :].clone()

    u[:, :, :, 0] = 0.0
    v[:, :, :, 0] = 0.0
    p[:, :, :, 0] = p[:, :, :, 1].clone()

    u[:, :, :, -1] = 0.0
    v[:, :, :, -1] = 0.0
    p[:, :, :, -1] = p[:, :, :, -2].clone()

    u[:, :, -1, :] = u[:, :, -2, :].clone()
    v[:, :, -1, :] = v[:, :, -2, :].clone()
    p[:, :, -1, :] = 0.0

    return torch.cat([u, v, p], dim=1)


def super_res_loss(model, lf_input, jinv, dxdxi, dxdeta, dydxi, dydeta, h):
    uvp_interior = model(lf_input)
    uvp = apply_bc(uvp_interior)

    u = uvp[:, 0:1, :, :]
    v = uvp[:, 1:2, :, :]
    p = uvp[:, 2:3, :, :]

    dudxi = ddxi(u, h)
    dudeta = ddeta(u, h)
    dvdxi = ddxi(v, h)
    dvdeta = ddeta(v, h)
    dpdxi = ddxi(p, h)
    dpdeta = ddeta(p, h)

    d2udxi2 = ddxi(dudxi, h)
    d2udeta2 = ddeta(dudeta, h)
    d2vdxi2 = ddxi(dvdxi, h)
    d2vdeta2 = ddeta(dvdeta, h)

    dudx = jinv * (dydeta * dudxi - dydxi * dudeta)
    dudy = jinv * (-dxdeta * dudxi + dxdxi * dudeta)
    dvdx = jinv * (dydeta * dvdxi - dydxi * dvdeta)
    dvdy = jinv * (-dxdeta * dvdxi + dxdxi * dvdeta)
    dpdx = jinv * (dydeta * dpdxi - dydxi * dpdeta)
    dpdy = jinv * (-dxdeta * dpdxi + dxdxi * dpdeta)

    d2udx2 = jinv * (dydeta * d2udxi2 - dydxi * d2udeta2)
    d2udy2 = jinv * (-dxdeta * d2udxi2 + dxdxi * d2udeta2)
    d2vdx2 = jinv * (dydeta * d2vdxi2 - dydxi * d2vdeta2)
    d2vdy2 = jinv * (-dxdeta * d2vdxi2 + dxdxi * d2vdeta2)

    nu = 0.01
    r_cont = dudx + dvdy
    r_xmom = u * dudx + v * dudy + dpdx - nu * (d2udx2 + d2udy2)
    r_ymom = u * dvdx + v * dvdy + dpdy - nu * (d2vdx2 + d2vdy2)

    loss = torch.mean(r_cont ** 2) + torch.mean(r_xmom ** 2) + torch.mean(r_ymom ** 2)
    return loss


def ddxi(f, h):
    dfdx_central = (f[:, :, :, 0:-4] - 8*f[:, :, :, 1:-3] + 8*f[:, :, :, 3:-1] - f[:, :, :, 4:]) / (12*h)
    dfdx_left = (-11*f[:, :, :, 0:2] + 18*f[:, :, :, 1:3] -9*f[:, :, :, 2:4] + 2*f[:, :, :, 3:5]) / (6*h)
    dfdx_right = (-2*f[:, :, :, -5:-3] + 9*f[:, :, :, -4:-2] -18*f[:, :, :, -3:-1] + 11*f[:, :, :, -2:]) / (6*h)

    return torch.cat((dfdx_left, dfdx_central, dfdx_right), dim=3)


def ddeta(f, h):
    dfdy_central = (f[:, :, 0:-4, :] - 8*f[:, :, 1:-3, :] + 8*f[:, :, 3:-1, :] - f[:, :, 4:, :]) / (12*h)
    dfdy_bot = (-11*f[:, :, 0:2, :] + 18*f[:, :, 1:3, :] -9*f[:, :, 2:4, :] + 2*f[:, :, 3:5, :]) / (6*h)
    dfdy_top = (-2*f[:, :, -5:-3, :] + 9*f[:, :, -4:-2, :] -18*f[:, :, -3:-1, :] + 11*f[:, :, -2:, :]) / (6*h)

    return torch.cat((dfdy_bot, dfdy_central, dfdy_top), dim=2)

def import_input_data():
    lf_data = np.load("sr_lfdata.npy")
    lf_x = lf_data[0, :, :]
    lf_y = lf_data[1, :, :]
    lf_u = lf_data[4, :, :]
    lf_v = lf_data[5, :, :]

    plt.figure()
    plt.pcolormesh(lf_x, lf_y, np.sqrt(lf_u ** 2 + lf_v ** 2), cmap=cm.coolwarm, vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.title("Low-Resolution Velocity Magnitude")
    plt.savefig("low_res_original.png")

    hf_data = np.load("sr_hfdata.npy")
    jinv = hf_data[0, :, :]
    dxdxi = hf_data[1, :, :]
    dxdeta = hf_data[2, :, :]
    dydxi = hf_data[3, :, :]
    dydeta = hf_data[4, :, :]
    hf_x = hf_data[5, :, :]
    hf_y = hf_data[6, :, :]

    ny, nx = hf_x.shape
    h = 0.01

    return lf_data, hf_data, jinv, dxdxi, dxdeta, dydxi, dydeta, hf_x, hf_y, ny, nx, h


if __name__ == "__main__":
    lf_data, hf_data, jinv, dxdxi, dxdeta, dydxi, dydeta, hf_x, hf_y, ny, nx, h = import_input_data()

    lf_u = lf_data[4, :, :]
    lf_v = lf_data[5, :, :]
    lf_input = torch.tensor(np.stack([lf_u, lf_v])[np.newaxis, :, :, :], dtype=torch.float32)

    upsample = nn.Upsample(size=(75, 47), mode="bicubic", align_corners=True)
    uv_bicubic = upsample(lf_input)[0].numpy()
    u_bicubic = uv_bicubic[0, :, :]
    v_bicubic = uv_bicubic[1, :, :]

    plt.figure()
    plt.pcolormesh(hf_x[1:-1, 1:-1], hf_y[1:-1, 1:-1], np.sqrt(u_bicubic ** 2 + v_bicubic ** 2), cmap=cm.coolwarm, vmin=0.0, vmax=1.3)
    plt.colorbar()
    plt.title("Bicubic Upsampled Velocity Magnitude")
    plt.savefig("bicubic_upsampled.png")

    jinv_t = torch.tensor(jinv, dtype=torch.float32)
    dxdxi_t = torch.tensor(dxdxi, dtype=torch.float32)
    dxdeta_t = torch.tensor(dxdeta, dtype=torch.float32)
    dydxi_t = torch.tensor(dydxi, dtype=torch.float32)
    dydeta_t = torch.tensor(dydeta, dtype=torch.float32)

    model = SuperRes(hlayers=10, height=75, width=32, kernel_size=5)
    losses = model.train_model(lf_input, jinv_t, dxdxi_t, dxdeta_t, dydxi_t, dydeta_t, h, epochs=5000, lr=1e-3)

    plt.figure()
    plt.plot(losses)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("physics loss")
    plt.title("Training Loss")
    plt.savefig("training_loss.png")

    model.eval()
    with torch.no_grad():
        uvp_interior = model(lf_input)
        uvp = apply_bc(uvp_interior)
        u_pred = uvp[0, 0, :, :].numpy()
        v_pred = uvp[0, 1, :, :].numpy()

    plt.figure()
    plt.pcolormesh(hf_x, hf_y, np.sqrt(u_pred ** 2 + v_pred ** 2), cmap=cm.coolwarm, vmin=0.0, vmax=1.3)
    plt.colorbar()
    plt.title("Super-Resolved Velocity Magnitude (CNN)")
    plt.savefig("super_res_result.png")
    plt.show()

