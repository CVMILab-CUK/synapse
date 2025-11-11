import time
import torch
import numpy as np

def timechecker(func):
    def f(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time   = time.time()
        print(f"End Time : {end_time-start_time:.4f}s")
        return result
    return f
    


class ToTensor(object):

    r"""
    Make Image data to tensor type
    """

    def __call__(self, data:np.array) -> None:
        data = torch.from_numpy(data.transpose(2, 0, 1).astype(np.float32))
        return data

