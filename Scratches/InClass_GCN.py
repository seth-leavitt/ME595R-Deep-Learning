import torch
from torch import functional as F
from torch import nn
import matplotlib as plt

class GCN(nn.Module):
    def __init__(self, x, edge_index):
        super().__init__()
        self.conv1 =