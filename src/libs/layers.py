import numpy as np
from functools import partial


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from .utils import exists
from .activation import Mish, softmax_3d

r"""
    Original Code : 
        - MSA  :  https://github.com/xxxnell/how-do-vits-work/blob/transformer/models/attentions.py
        - Conv : https://github.com/csqiangwen/DeepFillv2_Pytorch
"""
        

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None, norm=None, drop_rate:int=None, bias=False, alpha=0.2):
        super(LinearLayer, self).__init__()
        #Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_dim)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(out_dim)
        elif norm == None:
            self.norm = None
        else:
            assert 0, f"Unsupported normalization: {norm}"
        
        #Initialize the activation Function
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(alpha, inplace = True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == None:
            self.activation = None
        else:
            assert 0, f"Unsupported activation: {activation}"
        
        # Initialize the Drop Out
        if drop_rate == None:
            self.drop_out = None
        elif type(drop_rate) is float:
            self.drop_out = nn.Dropout(drop_rate)
        else:
            assert 0, f"Unsupported Drop Out Rate Type: {drop_rate} is {type(drop_rate)}"
        
        # Initialize Linear Layer
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim, bias=bias)
    
    def forward(self, x):

        x = self.linear(x)
        if self.norm:
            x = self.norm(x)

        if self.activation:
            x = self.activation(x)

        if self.drop_out:
            x = self.drop_out(x)

        return x

class Conv1dLayer(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride = 1, 
                 padding = 0, 
                 dilation = 1, 
                 groups=1, 
                 pad_type = 'zero', 
                 activation = None,
                 norm = None, 
                 bias=False, 
                 alpha=0.2, 
                 gn_groups=4):
        super(Conv1dLayer, self).__init__()
        # Initialize the padding scheme
        if padding == 0:
            self.pad = None
        else:
            if pad_type == 'reflect':
                self.pad = nn.ReflectionPad1d(padding)
            elif pad_type == 'replicate':
                self.pad = nn.ReplicationPad1d(padding)
            elif pad_type == 'zero':
                self.pad = nn.ConstantPad1d(padding, 0.0)
            else:
                assert 0, "Unsupported padding type: {}".format(pad_type)
        
        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(out_channels)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(out_channels)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(gn_groups, out_channels)
        elif norm == None:
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(alpha, inplace = True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "mish":
            self.activation = Mish()
        elif activation == None:
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation, groups=groups, bias=bias)
    
    def forward(self, x, scale_shift=None):
        if self.pad:
            x = self.pad(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        if self.activation:
            x = self.activation(x)
        
        return x

class Conv2dLayer(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride = 1, 
                 padding = 0, 
                 dilation = 1, 
                 groups=1, 
                 pad_type = 'zero', 
                 activation = None,
                 norm = None, 
                 bias=False, 
                 alpha=0.2, 
                 gn_groups=4):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme
        if padding == 0:
            self.pad = None
        else:
            if pad_type == 'reflect':
                self.pad = nn.ReflectionPad2d(padding)
            elif pad_type == 'replicate':
                self.pad = nn.ReplicationPad2d(padding)
            elif pad_type == 'zero':
                self.pad = nn.ConstantPad2d(padding, 0.0)
            else:
                assert 0, "Unsupported padding type: {}".format(pad_type)
        
        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(out_channels)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(gn_groups, out_channels)
        elif norm == None:
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(alpha, inplace = True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "mish":
            self.activation = Mish()
        elif activation == None:
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation, groups=groups, bias=bias)
    
    def forward(self, x, scale_shift=None):
        if self.pad:
            x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        if self.activation:
            x = self.activation(x)
        
        return x

class PositionWiseFeedForwardLayer(nn.Module):
    
    def __init__(self, dim, dff, rate=0.5, activation="gelu", bias=True):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = LinearLayer(dim, dff, activation=activation, bias=bias, drop_rate=rate )
        self.fc2 = LinearLayer(dff, dim, bias=bias)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out
    
class PositionWiseConv1dLayer(nn.Module):
    
    def __init__(self, dim, dff, rate=0.5, activation="gelu", bias=True):
        super(PositionWiseConv1dLayer, self).__init__()
        self.conv1 = Conv1dLayer(dim, dff, kernel_size=1, activation=activation, bias=bias)
        self.drop = nn.Dropout(p=rate)
        self.conv2 = Conv1dLayer(dff, dim, kernel_size=1, bias=bias)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.drop(out)
        out = self.conv2(out)
        return out

class PositionWiseFullyConvolutionalLayer(nn.Module):

    def __init__(self, dim, dff, rate=0.5, activation="gelu", bias=True):
        super(PositionWiseFullyConvolutionalLayer, self).__init__()
        self.conv1 = Conv2dLayer(dim, dff, kernel_size=1, activation=activation, bias=bias)
        self.drop = nn.Dropout(p=rate)
        self.conv2 = Conv2dLayer(dff, dim, kernel_size=1, bias=bias)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.drop(out)
        out = self.conv2(out)
        return out

class PositionWiseSeparableConvolutionalLayer(nn.Module):

    def __init__(self, dim, dff, rate=0.5, activation="gelu", bias=True):
        super(PositionWiseSeparableConvolutionalLayer, self).__init__()
        self.conv1 = Conv2dLayer(dim, dff, kernel_size=1, activation=activation, bias=bias)
        self.drop1 = nn.Dropout(p=rate)
        self.conv2 = Conv2dLayer(dff, dff, kernel_size=3, padding=1, groups=dff, activation=activation, bias=bias)
        self.drop2 = nn.Dropout(p=rate)
        self.conv3 = Conv2dLayer(dff, dim, kernel_size=1, bias=bias)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.drop1(out)
        out = self.conv2(out)
        out = self.drop2(out)
        out = self.conv3(out)
        return out
    

class MultiHeadAttention1dLayer(nn.Module):
    r"""
    Input Shape : [Batch, Sentence Length, dim_in]
    Output Shape : [Batch, Sentence Length, dim_out]
    """

    def __init__(self, dim_in, dim_out=None, num_heads=64,  bias=True):
        super(MultiHeadAttention1dLayer, self).__init__()

        self.num_heads = num_heads
        self.prob      = nn.Softmax(dim=-1)

        dim_out = dim_in if dim_out is None else dim_out

        self.wq = LinearLayer(dim_in, dim_out, bias=bias)
        self.wk = LinearLayer(dim_in, dim_out, bias=bias)
        self.wv = LinearLayer(dim_in, dim_out, bias=bias)

        self.split = Rearrange('b d (n e) -> b n d e', n=self.num_heads)
        self.concat = Rearrange('b n d e -> b d (n e)')

        self.out = LinearLayer(dim_out, dim_out,  bias= bias)
    
    def calc_attn(self, query, key, value, mask=None):
        # Query : [B x nHeads x Seq x Embed]
        # Key   : [B x nHeads x Seq x Embed]
        # Value : [B x nHeads x Seq x Embed]

        dots = einsum('b n i d, b n j d -> b n i j', query, key) # [B  x nHeads x Seq x Seq]
        dots = dots.masked_fill(mask) if mask is not None else dots
        attn = self.prob(dots)
        out = einsum('b n i j, b n j d -> b n i d', attn, value) # [B x nHeads x Seq x Embed]

        return out, attn
    
    def forward(self, query, key, value, mask=None):
        # Query : [B x Seq x Dim_In]
        # Key   : [B x Seq x Dim_In]
        # Value : [B x Seq x Dim_In]

        query  = self.wq(query) # [B x Seq x Dim_Out]
        key    = self.wk(key)   # [B x Seq x Dim_Out]
        value  = self.wv(value) # [B x Seq x Dim_Out]

        split_query = self.split(query) # [B x nHeads x Seq x Embed]
        split_key   = self.split(key)   # [B x nHeads x Seq x Embed]
        split_value = self.split(value) # [B x nHeads x Seq x Embed]

        out, attn = self.calc_attn(split_query, split_key, split_value, mask) # [B x nHeads x Seq x Embed], [B  x nHeads x Seq x Seq]
        out = self.concat(out) # [B x Seq x Dim_Out]
        out = self.out(out)    # [B x Seq x Dim_Out]

        return out, attn

    
class MultiHeadAttention2dLayer(nn.Module):
    r"""
    Input Shape : [Batch, dim_in, Height, Width]
    Output Shape : [Batch, dim_out, Height, Wdith]
    """

    def __init__(self, dim_in, dim_out=None, num_heads=64, img_size=16,  bias=False):
        super(MultiHeadAttention2dLayer, self).__init__()

        self.num_heads = num_heads
        self.prob      = nn.Softmax(dim= -1)

        dim_out = dim_in if dim_out is None else dim_out

        self.wq = Conv2dLayer(dim_in, dim_out, kernel_size=1, stride=1,  bias=bias)
        self.wk = Conv2dLayer(dim_in, dim_out, kernel_size=1, stride=1,  bias=bias)
        self.wv = Conv2dLayer(dim_in, dim_out, kernel_size=1, stride=1,  bias=bias)

        self.split = Rearrange('b (n e) (h) (w) -> b n (h w) e', n=self.num_heads)
        self.concat = Rearrange('b n (h w) e -> b (n e) (h) (w)', h =img_size)

        self.out = Conv2dLayer(dim_out, dim_out, kernel_size=1, stride=1,  bias=bias)
    
    def calc_attn(self, query, key, value, mask=None):
        # Query : [B x nHeads x (H x W) x Embed]
        # Key   : [B x nHeads x (H x W) x Embed]
        # Value : [B x nHeads x (H x W) x Embed]

        dots = einsum('b n i d, b n j d -> b n i j', query, key)     # [B x nHeads x (H x W) x (H x W)]
        dots = dots.masked_fill(mask) if mask is not None else dots
        attn = self.prob(dots)
        out = einsum('b n i j, b n j d -> b n i d', attn, value)     # [B x nHeads x (H x W) x Embed]

        return out, attn
    
    def forward(self, query, key, value, mask=None):
        # Query : [B x Dim_In x H x W]
        # Key   : [B x Dim_In x H x W]
        # Value : [B x Dim_In x H x W]

        query  = self.wq(query) # [B x Dim_Out x H x W]
        key    = self.wk(key)   # [B x Dim_Out x H x W]
        value  = self.wv(value) # [B x Dim_Out x H x W]

        split_query = self.split(query) # [B x nHeads x (H x W) x Embed]
        split_key   = self.split(key)   # [B x nHeads x (H x W) x Embed]
        split_value = self.split(value) # [B x nHeads x (H x W) x Embed]

        out, attn = self.calc_attn(split_query, split_key, split_value, mask)  # [B x nHeads x (H x W) x Embed], [B x nHeads x (H x W) x (H x W)]
        out = self.concat(out) # [B x Dim_Out x H x W]
        out = self.out(out)    # [B x Dim_Out x H x W]

        return out, attn
    


class EEGAttention1dLayer(nn.Module):
    r"""
    Input Shape : [Batch, dim_in, n_channels]
    Output Shape : [Batch, dim_out, n_channels]
    """

    def __init__(self, dim_in, dim_out=None, num_heads=64, n_channels=128, bias=False):
        super(EEGAttention1dLayer, self).__init__()

        self.num_heads = num_heads
        self.prob      = nn.Softmax(dim= -1)

        dim_out = dim_in if dim_out is None else dim_out

        self.wq = Conv1dLayer(dim_in, dim_out, kernel_size=1, stride=1, bias=bias)
        self.wk = Conv1dLayer(dim_in, dim_out, kernel_size=1, stride=1, bias=bias)
        self.wv = Conv1dLayer(dim_in, dim_out, kernel_size=1, stride=1, bias=bias)

        self.split = Rearrange('b (n e) c -> b n c e', n=self.num_heads)
        self.concat = Rearrange('b n c e -> b (n e) c')

        self.out = Conv1dLayer(dim_out, dim_out, kernel_size=1, stride=1, bias=bias)
    
    def calc_attn(self, query, key, value, mask=None):
        # Query : [B x nHeads x in_dim x Embed]
        # Key   : [B x nHeads x in_dim x Embed]
        # Value : [B x nHeads x in_dim x Embed]

        dots = einsum('b n i d, b n j d -> b n i j', query, key)     # [B x nHeads x in_dim x in_dim]
        dots = dots.masked_fill(mask) if mask is not None else dots
        attn = self.prob(dots)
        out = einsum('b n i j, b n j d -> b n i d', attn, value)     # [B x nHeads x in_dim x Embed]

        return out, attn
    
    def forward(self, query, key, value, mask=None):
        # Query : [B x Dim_In x n_channels]
        # Key   : [B x Dim_In x n_channels]
        # Value : [B x Dim_In x n_channels]

        query  = self.wq(query) # [B x Dim_Out x n_channels]
        key    = self.wk(key)   # [B x Dim_Out x n_channels]
        value  = self.wv(value) # [B x Dim_Out x n_channels]

        split_query = self.split(query) # [B x nHeads x Dim_Out x Embed]
        split_key   = self.split(key)   # [B x nHeads x Dim_Out x Embed]
        split_value = self.split(value) # [B x nHeads x Dim_Out x Embed]

        out, attn = self.calc_attn(split_query, split_key, split_value, mask)  # [B x nHeads x Dim_Out x Embed], [B x nHeads x Dim_Out x Dim_Out]
        out = self.concat(out) # [B x Dim_Out x in_channels]
        out = self.out(out)    # [B x Dim_Out x in_channels]

        return out, attn

class LoRA(nn.Module):
    def __init__(self, original_layer, rank=4, alpha=1.0):
        super(LoRA, self).__init__()
        self.original_layer = original_layer  # Pretrained layer
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # Initialize low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(original_layer.in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(rank, original_layer.out_features) * 0.01)

    def forward(self, x):
        # Original layer output + LoRA adjustments
        return self.original_layer(x) + (x @ self.lora_A @ self.lora_B) * self.scale

# Example: Adding LoRA to a Linear layer
class LinearWithLoRA(nn.Module):
    def __init__(self, original_linear, rank=4, alpha=1.0):
        super(LinearWithLoRA, self).__init__()
        self.lora = LoRA(original_linear, rank, alpha)

    def forward(self, x):
        return self.lora(x)

# Stable Diffusion CrossAttention Example
class CrossAttentionWithLoRA(nn.Module):
    def __init__(self, original_attention, rank=4, alpha=1.0):
        super(CrossAttentionWithLoRA, self).__init__()
        self.attention = original_attention  # Pretrained CrossAttention layer
        self.lora = LoRA(original_attention.to_q, rank, alpha)  # Apply LoRA to the Query projection

    def forward(self, x, context=None):
        return self.attention(x, context) + self.lora(x)