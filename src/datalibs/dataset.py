import sys
import os
import glob
import random
import math
import time
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
from scipy.fftpack import fft, rfft, fftfreq, irfft, ifft, rfftfreq
from scipy import signal
import numpy as np
import importlib
import cv2
from .utils import *
from .compose import Resize

from transformers import AutoProcessor

class EEGPrepDataset:

    # Constructor
    @timechecker
    def __init__(self, eeg_pre_path, eeg_data_path, transforms=None, img_size=512):
        # Load EEG signals
        print("Start Load...")
        # loaded = torch.load(eeg_signals_path)

        # split_loaded = torch.load(split_path)
        # if opt.subject!=0:
        #     self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset']) ) if loaded['dataset'][i]['subject']==0]
        # # else:
        # self.data=loaded['dataset']        
        # self.labels = loaded["labels"]
        # self.images = loaded["images"]
        self.image_path = eeg_data_path
        self.data = glob.glob(os.path.join(eeg_pre_path, "*"))

        # Compute size
        self.dataset_size = len(self.data)

        
        # Transforms
        self.transforms = transforms
        self.resize     =   Resize((img_size,img_size))
        self.to_tensor  = ToTensor()

    # Get size
    def __len__(self):
        return self.dataset_size

    # Get item
    def __getitem__(self, i):

        loaded = torch.load(self.data[i], weights_only=False)
        # Process EEG
        eeg = loaded["eeg"]

        # Get label        
        label = loaded["label"]

        # Get Original Image
        image_name = loaded["image"]
        s, _ = image_name.split("_")
        image_raw = cv2.imread(os.path.join(self.image_path, s, image_name+".JPEG"))
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)/255.
        # image = image_raw.copy()/ 255.
        ori_img = image.copy()
        if self.transforms:
            image = self.transforms(image)
            ori_img = self.resize(ori_img)
        
        image = self.to_tensor(image)
        ori_img = self.to_tensor(ori_img)

        # image_raw = self.processor(images=image_raw, return_tensors="pt")
        # image_raw['pixel_values'] = image_raw['pixel_values'].squeeze(0)
        
        # Return
        return {"eeg":eeg, "image":image, "label":label,"ori_img":ori_img, 'name':""}# "image_raw":image}

class EEGPreDataset:

    # Constructor
    @timechecker
    def __init__(self, eeg_pre_path, eeg_data_path, transforms=None):
        # Load EEG signals
        print("Start Load...")
        # loaded = torch.load(eeg_signals_path)

        # split_loaded = torch.load(split_path)
        # if opt.subject!=0:
        #     self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset']) ) if loaded['dataset'][i]['subject']==0]
        # # else:
        # self.data=loaded['dataset']        
        # self.labels = loaded["labels"]
        # self.images = loaded["images"]
        self.image_path = eeg_data_path
        self.data = glob.glob(os.path.join(eeg_pre_path, "*"))

        # Compute size
        self.dataset_size = len(self.data)

        
        # Transforms
        self.transforms = transforms
        self.to_tensor  = ToTensor()

    # Get size
    def __len__(self):
        return self.dataset_size

    # Get item
    def __getitem__(self, i):

        loaded = torch.load(self.data[i], weights_only=False)
        # Process EEG
        eeg = loaded["eeg"]

        # Get label        
        label = loaded["label"]

        # Get Original Image
        image_name = loaded["image"]
        s, _ = image_name.split("_")
        image = cv2.imread(os.path.join(self.image_path, s, image_name+".JPEG"))
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.
        except:
            print(image_name)
        if self.transforms:
            image = self.transforms(image)
        
        image = self.to_tensor(image)
        
        # Return
        return eeg, image, label

class EEGDataset:
    
    # Constructor
    @timechecker
    def __init__(self, eeg_signals_path, eeg_data_path, split_path, split_num=0, split_name="train", transforms=None):
        # Load EEG signals
        print("Start Load...")
        loaded = torch.load(eeg_signals_path)

        split_loaded = torch.load(split_path)
        # if opt.subject!=0:
        #     self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset']) ) if loaded['dataset'][i]['subject']==0]
        # else:
        self.data=loaded['dataset']        
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.image_path = eeg_data_path

        # Compute size
        self.dataset_size = len(self.data)

        
        # Load split
        self.split_idx = split_loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [i for i in self.split_idx if 450 <= self.data[i]["eeg"].size(1) <= 600]

        # Compute size
        self.split_size = len(self.split_idx)
        # Transforms
        self.transforms = transforms
        self.to_tensor  = ToTensor()

    # Get size
    def __len__(self):
        return self.split_size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[self.split_idx[i]]["eeg"].float().t()
        eeg = eeg[20:460,:]

        # if opt.model_type == "model10":
        #     eeg = eeg.t()
        #     eeg = eeg.view(1,128,460-20)
        # Get label        
        label = self.data[self.split_idx[i]]["label"]

        # Get Original Image
        image_name = self.images[self.data[self.split_idx[i]]["image"]]
        s, _ = image_name.split("_")
        image = cv2.imread(os.path.join(self.image_path, s, image_name+".JPEG"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.

        if self.transforms:
            image = self.transforms(image)
        
        image = self.to_tensor(image)
        
        # Return
        return eeg, image, label

# Splitter class
class Splitter:

    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [i for i in self.split_idx if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        # Compute size
        self.size = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        eeg, image, label = self.dataset[self.split_idx[i]]
        # Return
        return eeg, image, label



if __name__ == "__main__":
    eeg_signals_path = "/media/NAS/EEG2IMAGE/eeg_cvpr_2017/data/eeg_5_95_std.pth"
    img_path = '/media/NAS/EEG2IMAGE/eeg_cvpr_2017/image'
    # Load dataset
    dataset = EEGDataset(eeg_signals_path = eeg_signals_path,  eeg_data_path = img_path)
    # Create loaders
    loaders = {split: DataLoader(Splitter(dataset, split_path = "/media/NAS/EEG2IMAGE/eeg_cvpr_2017/data/block_splits_by_image_all.pth", 
                                        split_num = 0, 
                                        split_name = split), 8, drop_last = True, shuffle = True) for split in ["train", "val", "test"]}
