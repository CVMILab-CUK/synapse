# Default Library
import os
import numpy   as np

from PIL       import Image
from typing    import List
from omegaconf import OmegaConf
from einops    import rearrange, repeat

# Torch Library
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torch.utils.data  import DataLoader

# Diffusion Model
from dc_ldm.util import instantiate_from_config
from dc_ldm.models.diffusion.plms import PLMSSampler

# My Model
from models.eeg_AE import eeg_encoder


def create_model_from_config( in_seq:int              = 440,
                            in_channels             = 128,
                            out_seq:int             = 768,
                            dims:List               = [64, 128, 256, 512, 1024],
                            shortcut:bool           = True,
                            dropout:float           = 0.0,
                            groups:int              = 32,
                            layer_mode:str          = 'conv',
                            block_mode:str          = 'res',
                            down_mode:str           = 'max',
                            pos_mode:str            = 'sinusoidal',
                            skip_mode:str           = "conv",
                            n_layer:int             = 2,
                            n_head:int              = 64,
                            dff_factor:int          = 2,
                            stride:int              = 4,):
    
    model = eeg_encoder( in_seq = in_seq, 
                        in_channels = in_channels, 
                        out_seq = out_seq,
                        dims = dims, 
                        shortcut = shortcut, 
                        dropout = dropout, 
                        groups = groups,
                        layer_mode = layer_mode,  
                        block_mode = block_mode, 
                        down_mode = down_mode,
                        pos_mode = pos_mode, 
                        n_layer =n_layer, 
                        n_head = n_head,
                        dff_factor= dff_factor,
                        stride = stride, 
                        skip_mode= skip_mode)
    return model

def contrastive_loss(logits, dim):
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()
    
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)
    return (caption_loss + image_loss) / 2.0

class cond_stage_model(nn.Module):
    def __init__(self, 
                 pre_path:str            = None,
                 in_seq:int              = 440,
                 in_channels             = 128,
                 out_seq:int             = 768,
                 dims:List               = [64, 128, 256, 512, 1024],
                 shortcut:bool           = True,
                 dropout:float           = 0.0,
                 groups:int              = 32,
                 layer_mode:str          = 'conv',
                 block_mode:str          = 'res',
                 down_mode:str           = 'max',
                 pos_mode:str            = 'sinusoidal',
                 skip_mode:str           = "conv",
                 n_layer:int             = 2,
                 n_head:int              = 64,
                 dff_factor:int          = 2,
                 stride:int              = 4):
        super().__init__()
        assert os.path.exists(pre_path) or pre_path is None, f"Check your ckpt path"
        # prepare pretrained AE 
        self.model = eeg_encoder( in_seq = in_seq, 
                        in_channels = in_channels, 
                        out_seq = out_seq,
                        dims = dims, 
                        shortcut = shortcut, 
                        dropout = dropout, 
                        groups = groups,
                        layer_mode = layer_mode,  
                        block_mode = block_mode, 
                        down_mode = down_mode,
                        pos_mode = pos_mode, 
                        n_layer =n_layer, 
                        n_head = n_head,
                        dff_factor= dff_factor,
                        stride = stride, 
                        skip_mode= skip_mode)
        if pre_path:
            state_dict = torch.load(pre_path)
            m, u = self.model.load_state_dict(state_dict["net"], strict=False)
            print('missing keys:', u)
            print('unexpected keys:', m)
        
    def forward(self, x):
        # n, c, w = x.shape
        # print(x.get_device())
        # print(x.get_device())
        latent, _ = self.model(x)
        return latent

    def get_clip_loss(self, x, image_embeds):
        loss = 1 - torch.cosine_similarity(x.squeeze(), image_embeds, dim=-1).mean()
        return loss
    


class eLDM:

    def __init__(self, 
                 pre_path,
                 device                  = torch.device('cpu'),
                 pretrain_root           = './pretrain_models/',
                 logger                  = None, 
                 ddim_steps              = 250, 
                 global_pool             = True, 
                 use_time_cond           = False,
                 in_seq:int              = 440,
                 in_channels             = 128,
                 out_seq:int             = 768,
                 dims:List               = [64, 128, 256, 512, 1024],
                 shortcut:bool           = True,
                 dropout:float           = 0.0,
                 groups:int              = 32,
                 layer_mode:str          = 'conv',
                 block_mode:str          = 'res',
                 down_mode:str           = 'max',
                 pos_mode:str            = 'sinusoidal',
                 skip_mode:str           = "conv",
                 n_layer:int             = 2,
                 n_head:int              = 64,
                 dff_factor:int          = 2,
                 stride:int              = 4,
                 clip_tune               = False
                 ):
        # self.ckp_path = os.path.join(pretrain_root, 'model.ckpt')
        self.ckp_path = os.path.join(pretrain_root, 'v1-5-pruned.ckpt')
        self.config_path = os.path.join('config', 'config15.yaml') 
        config = OmegaConf.load(self.config_path)
        config.model.params.unet_config.params.use_time_cond = use_time_cond
        config.model.params.unet_config.params.global_pool = global_pool

        self.cond_dim = config.model.params.unet_config.params.context_dim

        model = instantiate_from_config(config.model)
        pl_sd = torch.load(self.ckp_path, map_location="cpu")['state_dict']
       
        m, u = model.load_state_dict(pl_sd, strict=False)
        model.cond_stage_trainable = True
        model.cond_stage_model = cond_stage_model(pre_path, in_seq, in_channels, out_seq, 
                                                  dims, shortcut, dropout, groups,layer_mode,
                                                  block_mode, down_mode, pos_mode, skip_mode, 
                                                  n_layer, n_head, dff_factor, stride)

        model.ddim_steps = ddim_steps
        model.re_init_ema()
        self.clip_tune = True
    
        model.p_channels   = config.model.params.channels
        model.p_image_size = config.model.params.image_size
        model.ch_mult      = config.model.params.first_stage_config.params.ddconfig.ch_mult

        # model.freeze_whole_model()
        # model.freeze_first_stage()
        # # model.unfreeze_diffusion_model()
        # model.unfreeze_cond_stage()

        
        self.device = device    
        self.model = model
        
        self.ldm_config = config
        self.pretrain_root = pretrain_root
        self.latent_dim = out_seq


    @torch.no_grad()
    def generate(self, loader, num_samples, ddim_steps, HW=None, limit=None, state=None, output_path = None, num_count=5):
        # loader in "eeg", "image"
        all_samples = []
        if HW is None:
            shape = (self.ldm_config.model.params.channels, 
                self.ldm_config.model.params.image_size, self.ldm_config.model.params.image_size)
        else:
            num_resolutions = len(self.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
            shape = (self.ldm_config.model.params.channels,
                HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        self.model = self.model.to(self.device)
        self.model.cond_stage_model = self.model.cond_stage_model.to(self.device)
        sampler = PLMSSampler(self.model)
        # sampler = DDIMSampler(model)
        if state is not None:
            torch.cuda.set_rng_state(state)
            
        with self.model.ema_scope():
            self.model.eval()
            for count, item in enumerate(loader):
                if limit is not None:
                    if count >= limit:
                        break
                # print(item)
                for idx, sample in enumerate(zip(item['eeg'], item['image'])):
                    latent = sample[0]
                    gt_image = rearrange(sample[1], 'c h w  -> 1 c h w') # h w c
                    print(f"rendering {num_samples} examples in {ddim_steps} steps.")
                    # print(repeat(latent, '1 h w -> c h w', c=num_samples).shape)
                
                    c  = self.model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                    samples_ddim, _ = sampler.sample(S=ddim_steps, 
                                                    conditioning=c,
                                                    batch_size=num_samples,
                                                    shape=shape,
                                                    verbose=False)

                    x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                    gt_image = torch.clamp((gt_image+1.0)/2.0, min=0.0, max=1.0)
                    
                    all_samples.append(torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0)) # put groundtruth at first
                    if output_path is not None:
                        samples_t = (255. * torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0).numpy()).astype(np.uint8)
                        for copy_idx, img_t in enumerate(samples_t):
                            img_t = rearrange(img_t, 'c h w -> h w c')
                            Image.fromarray(img_t).save(os.path.join(output_path, 
                                f'./test{count}-{idx}-{copy_idx}.png'))
        
        # display as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples+1)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        model = self.model.to('cpu')
        
        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)