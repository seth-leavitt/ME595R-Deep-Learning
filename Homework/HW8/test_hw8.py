import numpy as np
import torch
from HW8 import SuperRes, apply_bc, super_res_loss

lfdata = np.load("sr_lfdata.npy")
hfdata = np.load("sr_hfdata.npy")

lfu = lfdata[4, :, :]
lfv = lfdata[5, :, :]
lf_input = torch.tensor(np.stack([lfu, lfv])[np.newaxis, :, :, :], dtype=torch.float32)

Jinv_t   = torch.tensor(hfdata[0, :, :], dtype=torch.float32)
dxdxi_t  = torch.tensor(hfdata[1, :, :], dtype=torch.float32)
dxdeta_t = torch.tensor(hfdata[2, :, :], dtype=torch.float32)
dydxi_t  = torch.tensor(hfdata[3, :, :], dtype=torch.float32)
dydeta_t = torch.tensor(hfdata[4, :, :], dtype=torch.float32)
h = 0.01

print("LF input shape:", lf_input.shape)

torch.manual_seed(1021)
model = SuperRes(hlayers=10, height=75, width=32, kernel_size=5)

uvp_int = model(lf_input)
print("Interior output shape:", uvp_int.shape)

uvp = apply_bc(uvp_int)
print("After BC shape:", uvp.shape)

loss = super_res_loss(model, lf_input, Jinv_t, dxdxi_t, dxdeta_t, dydxi_t, dydeta_t, h)
print("Initial loss:", loss.item())
print("SUCCESS - all shapes and computations work!")

