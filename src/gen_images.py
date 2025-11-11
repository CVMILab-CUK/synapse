import warnings
import logging
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("timm").setLevel(logging.ERROR)
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")

import argparse
import torch, os
import torch.nn as nn

from trainer.eeg_ldm2_ddp_trainer import EEGLDM2Trainer as trainer



parser = argparse.ArgumentParser(description="Evaluation of Patch Painting Transformer",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c', '--config', default= "./EEGLDM2_token_Test.json", help="Model's Config", type=str, dest="config")
parser.add_argument("-s", "--shared_path", default=os.path.dirname(os.path.abspath(__file__)), help="Shared File Path For Distributed Learning", dest="shared")
parser.add_argument("-p", "--pretrained_path", default=os.path.join(".", "pretrain_models","EEG_IP_CFG","checkpoint-13000"), dest="pretrained_path")
parser.add_argument("-o", "--output_path", default=os.path.join(".", "output"), dest="output_foloder")
parser.add_argument("-f", "--cfg_scale", default=7.5, dest="cfg_scale")
# ip_adaption_path
args = parser.parse_args()

configFilePath = args.config
sharedFilePath = args.shared
pretrained_path = args.pretrained_path
unet_path       = os.path.join(pretrained_path, "unet", "diffusion_pytorch_model.safetensors")
# cond_path       = os.path.join(pretrained_path, "model_1.safetensors")
ip_adaption_path = os.path.join(pretrained_path, "ip_adaption", "diffusion_pytorch_model.safetensors")
cond_path        = None
ema_path        = os.path.join(pretrained_path, "ema_unet", "diffusion_pytorch_model.safetensors")
output_path     = args.output_foloder
cfg_scale       = float(args.cfg_scale)

if __name__ =="__main__":
    if os.path.exists(sharedFilePath + "/sharedfile"):
        os.remove(sharedFilePath+ "/sharedfile")
    Trainer = trainer(configFilePath, sharedFilePath)
    Trainer.test(unet_path=unet_path, cond_path=cond_path,output_path=output_path, ema_path=ema_path, ip_adaption_path=ip_adaption_path, cfg_scale=cfg_scale)