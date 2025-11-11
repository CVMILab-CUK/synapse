import os
import json
import numpy as np
from typing import Dict
from typing import Optional

import torch
from torch import nn
from inspect import isfunction



def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def init_weights(m, init_gain=0.02):
    classname = m.__class__.__name__
    if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
        nn.init.normal_(m.weight.data, 0.0, init_gain)

    elif classname.find("Norm") != -1:
        nn.init.normal_(m.weight.data, 1.0, init_gain)
        nn.init.constant_(m.bias.data, 0.0)

def get_angles(pos, i, d_model):
    angle_rates = 1/np.power(10000, (2* (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return torch.from_numpy(pos_encoding.astype(np.float32))

