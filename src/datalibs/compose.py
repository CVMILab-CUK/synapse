import cv2
import numpy as np


class Resize(object):

    # ReSizing Data

    def __init__(self, shape):
        self.shape = shape
    
    def __call__(self, data):
        data = cv2.resize(data, dsize=(self.shape[0], self.shape[1]), interpolation=cv2.INTER_LINEAR)              
        return data

class Normalization(object):
    
    # Normalized Data
    def __init__(self, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data= (data - self.mean) / self.std  
        return data

class Scaling(object):

    def __init__(self, min_value=-1, max_value = 1):
        assert min_value < max_value, f"min value must be smaller than max value check your parameter now {min_value}, {max_value}"
        self.min_value = min_value
        self.max_value = max_value
        self.scale = self.max_value - self.min_value
        if min != 0:
            self.shift = self.scale / 2
        else:
            self.shift = 0
    def __call__(self, data):
        data = data * self.scale - self.shift
        return data

class Channel_control(object):
    def __init__(self, last=False):
        self.last = last
    
    def __call__(self, data):
        if self.last:
            data = data.permute(0, 2, 3, 1)
        return data