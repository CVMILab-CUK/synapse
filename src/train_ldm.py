import warnings
import logging
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("timm").setLevel(logging.ERROR)
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")

import argparse
import torch, os
import torch.nn as nn

# from trainer.eeg_ldm2_trainer import EEGLDM2Trainer as trainer



parser = argparse.ArgumentParser(description="Evaluation of Patch Painting Transformer",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c', '--config', default= "./config/Train_LDM.json", help="Model's Config", type=str, dest="config")
parser.add_argument("-s", "--shared_path", default=os.path.dirname(os.path.abspath(__file__)), help="Shared File Path For Distributed Learning", dest="shared")
parser.add_argument("-m", "--train_mode", default="ddp", help="Shared File Path For Distributed Learning", dest="train_mode")

args = parser.parse_args()

configFilePath = args.config
sharedFilePath = args.shared
train_mode     = args.train_mode

def loadTrainer():
    module = "trainer"
    if train_mode == "fsdp":
        exec(f"from {module}.eeg_ldm2_fsdp_trainer import EEGLDM2Trainer as trainer")
    else:
        exec(f"from {module}.eeg_ldm2_ddp_trainer import EEGLDM2Trainer as trainer")
    
    return eval("trainer")

if __name__ =="__main__":
    if os.path.exists(sharedFilePath + "/sharedfile"):
        os.remove(sharedFilePath+ "/sharedfile")
    trainer = loadTrainer()
    Trainer = trainer(configFilePath, sharedFilePath, training_mode = train_mode)
    Trainer.train()