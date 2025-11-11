import gc, os
import numpy as np
from PIL import Image
import datetime

import torch
from torch import nn
import pytorch_lightning as pl
# import lightning.pytorch as pl
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.strategies import FSDPStrategy, DDPStrategy

from einops import rearrange
from omegaconf import OmegaConf

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from models import eLDM
from trainer.base_trainer import BaseTrainer
from trainer.utils import plot_recon_figures, save, CustomDDPStrategy
from libs.losses import l1, l2, SignalDiceLoss, ContrastiveLoss
from libs.metric import SignalDice, get_eval_metric


from dc_ldm.models.diffusion.plms import PLMSSampler
from dc_ldm.models.diffusion.ddim import DDIMSampler
from dc_ldm.modules.encoders.modules import FrozenClipImageEmbedder as ImageClip

from .utils import add_lora_to_model

class EEGLDMTrainer(BaseTrainer):
    def __init__(self, json_file, sharedFilePath):
        self.json_dict = self.json_load(json_file)
        self.sharedFilePath = sharedFilePath
        self.startEpoch     = 0
        self.globalStep     = 0
        self.eps            = 1e-6
        super().__init__(self.json_dict["ckpt_dir"], self.json_dict["log_dir"], self.json_dict["batch_size"], 
                         sharedFilePath, self.json_dict["num_workers"])

        self.json_parser()
        self.__checkDirectory__()
        self.losses = {"sdsc":[], "rec":[], "sim":[],"loss":[]}
        self.accuracy = {"sdsc":[], "mse":[]}
        self.val_losses = {"sdsc":[], "rec":[], "sim":[], "loss":[]}
        self.val_accuracy = {"sdsc":[], "mse":[]}

        
    def json_parser(self):

        ########################################
        #            Train Setting
        ########################################
        self.name               = self.json_dict["name"]
        self.data_name          = self.json_dict["data_name"]
        self.log_dir            = self.json_dict["log_dir"]
        self.ckpt_dir           = self.json_dict["ckpt_dir"]
        self.eeg_train_path     = self.json_dict["eeg_train_path"]
        self.eeg_test_path      = self.json_dict["eeg_test_path"]
        self.eeg_val_path       = self.json_dict["eeg_val_path"]
        self.eeg_pretrian_path  = self.json_dict["eeg_pretrian_path"]
        self.pretrain_path      = self.json_dict["pretrain_path"]
        self.img_path           = self.json_dict["img_path"]
        self.split_path         = self.json_dict["split_path"]

        ########################################
        #            Model Setting
        ########################################
        self.in_seq         = self.json_dict["in_seq"]
        self.in_channels    = self.json_dict["in_channels"]
        self.z_channels     = self.json_dict["z_channels"]
        self.out_seq        = self.json_dict["out_seq"]
        self.dims           = self.json_dict["dims"]
        self.shortcut       = bool(self.json_dict["shortcut"])
        self.dropout        = self.json_dict["dropout"]
        self.groups         = self.json_dict["groups"]
        self.layer_mode     = self.json_dict["layer_mode"]
        self.block_mode     = self.json_dict["block_mode"]
        self.down_mode      = self.json_dict["down_mode"]
        self.up_mode        = self.json_dict["up_mode"]
        self.pos_mode       = self.json_dict["pos_mode"]
        self.skip_mode      = self.json_dict["skip_mode"]
        self.sim_mode       = self.json_dict["sim_mode"]
        self.n_layer        = self.json_dict["n_layer"]
        self.n_head         = self.json_dict["n_head"]
        self.dff_factor     = self.json_dict["dff_factor"]
        self.stride         = self.json_dict["stride"]
        self.sdsc_lambda    = self.json_dict["sdsc_lambda"]
        self.sim_lambda     = self.json_dict["sim_lambda"]
        self.img_size       = self.json_dict["img_size"]


        ########################################
        #         Training Parameters
        ########################################
        self.lr        = self.json_dict["learningRate"]
        self.epochs    = self.json_dict["trainEpochs"]
        self.saveIter  = self.json_dict["saveIter"]
        self.validIter = self.json_dict["validIter"]
        self.logIter   = self.json_dict["logIter"]
        self.restart   = self.json_dict["restart"]
        self.cfg_scale = self.json_dict["cfg_scale"]

        ########################################
        #         LDM Parameters
        ########################################
        self.clip_tune   = bool(self.json_dict["clip_tune"])
        self.cls_tune    = bool(self.json_dict["cls_tune"])
        self.eval_avg    = bool(self.json_dict["eval_avg"])
        self.num_samples = self.json_dict["num_samples"]
        self.ddim_steps  = self.json_dict["ddim_steps"]

    def create_trainer(self, 
                        precision=32, 
                        accumulate_grad_batches=2,
                        logger=None, 
                        check_val_every_n_epoch=2, 
                        # devices=torch.cuda.device_count(), 
                        devices=[0, 1, 2, 3], 
                        strategy="ddp"):
        acc = 'gpu' if torch.cuda.is_available() else 'cpu'
        logger = TensorBoardLogger(self.log_dir) if logger is None else logger
        # torch.set_float32_matmul_precision('medium')
        strategy = CustomDDPStrategy(timeout=datetime.timedelta(seconds=129600000), sharedFilePath=self.sharedFilePath, 
                          find_unused_parameters=True,)
        # strategy = DDPStrategy(find_unused_parameters=True, timeout=datetime.timedelta(seconds=129600000))
        # strategy = FSDPStrategy(timeout=datetime.timedelta(seconds=129600000))
        # strategy = "ddp"


        return pl.Trainer(accelerator=acc, max_epochs=self.epochs, logger=logger,
                          devices=devices, strategy=strategy,#"fsdp",
                          precision=precision, accumulate_grad_batches=accumulate_grad_batches,
                          enable_checkpointing=False, enable_model_summary=False,# gradient_clip_val=0.5,
                          check_val_every_n_epoch=check_val_every_n_epoch)
    
    def model_define(self):
        #Set LDM
        self.MODEL = eLDM(self.eeg_pretrian_path, torch.device("cuda"), self.pretrain_path, 
                          in_seq = self.in_seq, in_channels = self.in_channels, out_seq = self.out_seq,
                          dims = self.dims, shortcut = self.shortcut, dropout = self.dropout,
                          groups = self.groups, layer_mode = self.layer_mode, block_mode = self.block_mode,
                          down_mode = self.down_mode, pos_mode = self.pos_mode, skip_mode = self.skip_mode, 
                          n_layer = self.n_layer, n_head = self.n_head, dff_factor = self.dff_factor,
                          stride =  self.stride, clip_tune = self.clip_tune)
        # self.MODEL.model.clip_tune                     = self.clip_tune
        self.MODEL.model.cls_tune                      = self.cls_tune
        self.MODEL.model.main_config                   = self.json_dict
        self.MODEL.model.output_path                   = self.ckpt_dir
        self.MODEL.model.run_full_validation_threshold = 0.15

        self.MODEL.model.unfreeze_whole_model()
        self.MODEL.model.freeze_first_stage()
        # self.MODEL.model.freeze_whole_model()
        # self.MODEL.model.unfreeze_cond_stage()
        # self.MODEL.model = add_lora_to_model(self.MODEL.model, rank=4, alpha=1.0) # for lora
        self.MODEL.model.learning_rate = self.lr
        self.MODEL.model.train_cond_stage_only = True
        self.MODEL.model.eval_avg = self.eval_avg

        # Set Trainer
        self.trainer = self.create_trainer(check_val_every_n_epoch=1)
    
    def train(self):

        # print('\n##### Stage One: only optimize conditional encoders #####')
                
        self.makeDatasets(self.eeg_train_path, self.eeg_test_path, self.eeg_val_path, self.img_path, self.img_size, min_value=0, ddp=False)
        self.model_define()
        self.trainer.fit(self.MODEL.model, self.loader_train, val_dataloaders=self.loader_valid)
        self.MODEL.model.unfreeze_whole_model()
        torch.save(
            {
                'model_state_dict': self.MODEL.model.state_dict(),
                'state': torch.random.get_rng_state()

            },
            os.path.join(self.ckpt_dir, 'EEG_LDM.pth')
        )
        self.generate_images()

    def generate_valid(self, gen_path = "./ckpt_dir/val_attn.pth", output_path = "./log"):

        self.makeDatasets(self.eeg_train_path, self.eeg_test_path, self.eeg_val_path, self.img_path, self.img_size, min_value=0, ddp=False)        
        self.model_define()
        sd = torch.load(gen_path)
        self.MODEL.model.load_state_dict(sd['model_state_dict'], strict=False)

        state = sd['state']

        print('load ldm successfully')
        grid, _ = self.MODEL.generate(self.loader_train, self.num_samples, 
                   self.ddim_steps, (self.img_size, self.img_size), 10) # generate 10 instances
        grid_imgs = Image.fromarray(grid.astype(np.uint8))
        
        grid_imgs.save(os.path.join(output_path, f'./samples_train.png'))

        grid, samples =self.MODEL.generate(self.loader_valid, self.num_samples, 
                    self.ddim_steps, (self.img_size, self.img_size), state=state, output_path = output_path+"/val") # generate 10 instances
        grid_imgs = Image.fromarray(grid.astype(np.uint8))


        grid_imgs.save(os.path.join(output_path, f'./samples_valid.png'))

        grid, samples =self.MODEL.generate(self.loader_test, self.num_samples, 
                    self.ddim_steps, (self.img_size, self.img_size), limit=30, state=state, output_path = output_path) # generate 10 instances
        grid_imgs = Image.fromarray(grid.astype(np.uint8))


        grid_imgs.save(os.path.join(output_path, f'./samples_test.png'))


    
    def generate_images(self):
        grid, _ = self.MODEL.generate(self.loader_train, self.num_samples, self.ddim_steps, limit=10, state=torch.random.get_rng_state())
        grid_imgs = Image.fromarray(grid.astype(np.uint8))
        grid_imgs.save(os.path.join(self.log_dir, 'samples_train.png'))

        grid, samples = self.MODEL.generate(self.loader_test, self.num_samples, self.ddim_steps, limit=10, state=torch.random.get_rng_state())
        grid_imgs = Image.fromarray(grid.astype(np.uint8))
        grid_imgs.save(os.path.join(self.log_dir, 'samples_train.png'))

        for sp_idx, imgs in enumerate(samples):
            for copy_idx, img in enumerate(imgs[1:]):
                img = rearrange(img, 'c h w -> h w c')
                Image.fromarray(img).save(os.path.join(self.log_dir, 
                                f'./test{sp_idx}-{copy_idx}.png'))
        metric, metric_list = get_eval_metric(samples, avg=self.eval_avg)
        metric_dict = {f'summary/pair-wise_{k}':v for k, v in zip(metric_list[:-2], metric[:-2])}
        metric_dict[f'summary/{metric_list[-2]}'] = metric[-2]
        metric_dict[f'summary/{metric_list[-1]}'] = metric[-1]
    
 ## END LINE ###       