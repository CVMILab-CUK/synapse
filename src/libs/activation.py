import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from torch.nn import functional as F

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from .utils import exists

class Mish(nn.Module):
    def forward(self, x):
         return x * torch.tanh(F.softplus(x))


def softmax_3d(heatmap):
    # heatmap: Tensor of shape (B, K, H, W, D) containing the batched 3D heatmaps for B samples and K keypoints

    # Reshape the heatmap to (B * K, H, W, D) to apply softmax across all keypoints
    heatmap_reshaped = heatmap.view(-1, heatmap.shape[2] * heatmap.shape[3] * heatmap.shape[4])
    # Apply softmax along the spatial dimensions (H, W, D)
    softmax_heatmap = torch.softmax(heatmap_reshaped, dim=-1)

    # Reshape back to the original shape (B, K, H, W, D)
    softmax_heatmap = softmax_heatmap.view(*heatmap.shape)

    return softmax_heatmap