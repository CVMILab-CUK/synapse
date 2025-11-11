# Default Library
import os
import numpy   as np
from copy   import deepcopy as copy
from typing import Any, Callable, Dict, List, Optional, Union


from PIL       import Image
from typing    import List
from omegaconf import OmegaConf
from einops    import rearrange, repeat
from einops.layers.torch import Rearrange

# Torch Library
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torchvision.utils import make_grid
from torch.utils.data  import DataLoader
from safetensors.torch import load_file

# Accelerator Libraries
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
# from torch.distributed.fsdp.sharding_strategy import ShardingStrategy


from transformers.utils import ContextManagers


# Diffusion Model
import huggingface_hub

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler, DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

# My Model
from models.eeg_AE import eeg_encoder
from models.ip_adapter import EEGAdapter
from libs.blocks import TransformerBlock, IPAdaptationBlock
from .attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor

r"""
Custom By original SD3 pipelines

See Link Below

https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py#L147
"""


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import EEGSD3Pipeline

        >>> pipe = EEGSD3Pipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
        ... )
        >>> pipe.to("cuda")
        >>> eeg = eeg_data
        >>> image = pipe(eeg).images[0]
        >>> image.save("sd3.png")
        ```
"""


def create_model_from_config( in_seq:int              = 440,
                            in_channels             = 128,
                            out_channels            = 1,
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
                        out_channels=out_channels, 
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

class cond_stage_model2(nn.Module):
    def __init__(self, 
                 pre_path:str            = None,
                 in_seq:int              = 440,
                 in_channels             = 128,
                 out_channels:int        = 77,
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
                 global_attn:bool        = False):
        super().__init__()
        assert os.path.exists(pre_path) or pre_path is None, f"Check your ckpt path"
        # prepare pretrained AE 
        self.model = eeg_encoder( in_seq = in_seq, 
                        in_channels = in_channels, 
                        out_channels = out_channels,
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
                        skip_mode= skip_mode,
                        global_attn=global_attn)
        if pre_path:
            state_dict = torch.load(pre_path)
            m, u = self.model.load_state_dict(state_dict["net"], strict=False)
            print('missing keys:', u)
            print('unexpectesc keys:', m)
    
    @property
    def dtype(self):
        return self.type
        
    def forward(self, x):
        # n, c, w = x.shape
        # print(x.get_device())
        # print(x.get_device())
        latent, _ = self.model(x)
        return latent.permute(0, 2, 1)

    def get_clip_loss(self, x, image_embeds):
        loss = 1 - torch.cosine_similarity(x.squeeze(), image_embeds, dim=-1).mean()
        return loss



class eLDM2:

    def __init__(self, 
                 pre_path                = os.path.join(".","pretrain_models", "EEG_encoder_sd2.pth"),
                 device                  = torch.device('cpu'),
                 # CondStage Setting
                 global_pool             = True, 
                 use_time_cond           = False,
                 in_seq:int              = 440,
                 in_channels             = 128,
                 out_channels:int        = 1,
                 out_seq:int             = 1024,
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
                 global_attn:bool        = True,

                 # Diffusers Settings
                 gradient_accumulation_steps:int = 1,
                 mixed_precision:str             = "fp16",
                 report_to:str                   = "tensorboard",
                 use_ema:bool                    = False,
                 foreach_ema:bool                = True,
                #  model_id                        = "stabilityai/stable-diffusion-2-1-base",
                 model_id                        = "stabilityai/stable-diffusion-2-1",
                 output_dir                      = "./ckpt_dir",
                 logging_dir                     = "./log",
                 revision:str                    = None,
                 variant:str                     = None,#"non_ema",
                 non_ema_revision:str            = None,

                 # Training settings
                 learning_rate: int      = 1e-4,
                 beta1:int               = 0.9,
                 beta2:int               = 0.999,
                 lr_scheduler:str        = "constant",
                 lr_warmup_steps:int     = 500,
                 epochs:int              = 150,
                 eps:float               = 1e-8,
                 training_mode           = "ddp",
                 ip_adapter_enabled:bool  = False,
                 ip_adapter_token_num:int = 4,
                 cfg_scale:float = 0.1,

                 ):
        assert training_mode in ["ddp", "fsdp", None], f"Check your training_mode we only support 'ddp', 'fsdp', None. Now : {training_mode} "
        self.use_ema = use_ema
        self.ip_adapter_enabled = ip_adapter_enabled
        self.cfg_scale = cfg_scale
        # Make Accelerator For Training
        accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)
        # For FSDP
        if training_mode == "fsdp":
            fsdp_plugin = FullyShardedDataParallelPlugin(
                state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
                optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
                sync_module_states=True,
                use_orig_params=True,
                sharding_strategy = torch.distributed.fsdp.ShardingStrategy(2)#"SHARD_GRAD_OP"
            )
            mixed_precision = None
        else:
            fsdp_plugin = None

        self.accelerator = Accelerator(gradient_accumulation_steps = gradient_accumulation_steps,
                                #   mixed_precision = mixed_precision,
                                  mixed_precision = mixed_precision,
                                  fsdp_plugin = fsdp_plugin,
                                  log_with = report_to,
                                  project_config=accelerator_project_config)

        self.accelerator.print = lambda *args, **kwargs: None

        self.noise_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        # self.noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
        

        with ContextManagers(self.deepspeed_zero_init_disabled_context_manager()):
            self.vae = AutoencoderKL.from_pretrained(
                model_id, subfolder="vae", revision=revision, variant=variant,
            )

        self.cond_models = cond_stage_model2(pre_path, in_seq, in_channels, out_channels, out_seq, 
                                                  dims, shortcut, dropout, groups,layer_mode,
                                                  block_mode, down_mode, pos_mode, skip_mode, 
                                                  n_layer, n_head, dff_factor, stride, global_attn=global_attn)#.to(dtype=torch.float16) # Chnage for my EEG Models

        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", revision=non_ema_revision,
        )

        # Register buffers
        if not  self.noise_scheduler.config.prediction_type == "v_prediction":
            self.register_schedule(eps)

        # IF IP-Adation modules
        if self.ip_adapter_enabled:
            # Make Aaption Network

            # self.ip_adaption_modules  = IPAdaptationBlock(out_seq, out_seq, n_head, dff_factor, out_channels, 0.5, 0.5, out_attn=False, ip_adapter_token_num=ip_adapter_token_num)
            self.ip_adaption_modules= nn.Sequential(
                nn.Linear(out_seq, ip_adapter_token_num * out_seq),
                Rearrange('b (n e) -> b n e', n=ip_adapter_token_num),
                nn.LayerNorm(out_seq)
            #     Rearrange('b c s -> b s c'),
            #     TransformerBlock(out_seq, out_seq, n_head, dff_factor, out_channels, 0.5, 0.5, out_attn=False),
            #     Rearrange('b s c -> b c s', ),
            )        

            self.ip_adaption_modules.train()
            self.set_ip_adaption()
        else:
            self.ip_adaption_modules = None

        # Freeze vae and text_encoder and set unet to trainable
        self.vae.requires_grad_(False)
        self.cond_models.requires_grad_(False)
  
        self.vae.eval()
        # self.vae.enable_xformers_memory_efficient_attention()

        self.cond_models.eval()
        

        # Set Optimizer
        # # # Set Parameters
        
        cond_params = []

        self.unet.requires_grad_(True)
        for param in self.unet.parameters():
            param.requires_grad = True
        self.cond_models.train()
        self.unet.train()
        # For efficiency models
        self.unet.enable_gradient_checkpointing()
        # self.unet.enable_xformers_memory_efficient_attention()
        # Create EMA for the unet.

        if use_ema:
            # self.ema_unet = UNet2DConditionModel.from_pretrained(
            #     model_id, subfolder="unet", revision=revision, variant=variant
            # )
            self.ema_unet = copy(self.unet)
            self.ema_unet = EMAModel(
                self.ema_unet.parameters(),
                model_cls=self.ema_unet,
                model_config=self.ema_unet.config,
                foreach=foreach_ema,
                # decay=0.99
            )

        for n, p in self.unet.named_parameters():
            if 'attn2' in n or 'time' in n or 'norm2' in n:
                cond_params.append(p)
            else:
                p.requires_grad = False

        # Set IP Adation Parameters
        if self.ip_adapter_enabled:

            # self.unet.requires_grad_(False)
            # self.unet.eval()
            for proc in self.unet.attn_processors.values():
                proc.requires_grad = True
                cond_params += list(proc.parameters())

            cond_params += list(self.ip_adaption_modules.parameters())
            
        # Set Selective Params
        # else:            
       

        self.params = set(cond_params)

        # self.params = self.unet.parameters()
        # self.params = set(list(self.cond_models.parameters()) + cond_params)
        # print(self.params)

        total_params = sum(p.numel() for p in self.params)

        print(f"Number of Trainable Params: {total_params:,}")


        self.optimizer = torch.optim.AdamW(
                                            self.params,
                                            lr=learning_rate,
                                            betas=(beta1, beta2),
                                        )
    
    def set_ip_adaption(self):

        attn_procs = {}
        unet_sd = self.unet.state_dict()
        for name in self.unet.attn_processors.keys():

            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
                attn_procs[name].load_state_dict(weights)
        self.unet.set_attn_processor(attn_procs)
        # self.adapter_modules = torch.nn.ModuleList(self.unet.attn_processors.values())
        
    def register_schedule(self, eps):
        betas  = self.noise_scheduler.betas  # Scheduler? beta ?
        alphas = self.noise_scheduler.alphas
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        alphas_cumprod_prev = torch.cat([torch.tensor([1,]), alphas_cumprod[:-1]])
        posterior_variance = (1. - alphas_cumprod_prev + eps) / (1. - alphas_cumprod + eps) 

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_variance = posterior_variance
        
        # self.register_buffer('betas', betas)
        # self.register_buffer('alphas', alphas)
        # self.register_buffer('alphas_cumprod', alphas_cumprod)
        # self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        # self.register_buffer('posterior_variance', posterior_variance)
        if self.noise_scheduler.config.prediction_type == "epsilon":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * torch.tensor(alphas, dtype=torch.float32) * (1 - self.alphas_cumprod))
        elif self.noise_scheduler.config.prediction_type == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))

        else:
            raise NotImplementedError("mu not supported")

        lvlb_weights[0] = lvlb_weights[1]
        self.lvlb_weights = lvlb_weights
        # self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)


    def from_pretrained(self, unet_path, cond_path, ema_path=None, ipadaption_path=None):
        
        print(f"Pretrained Weights from path Unet Path : {unet_path} Cond Path : {cond_path}" )
        unet_weights = load_file(unet_path)

        for name, param in self.unet.named_parameters():
            before_param = param.clone()
            if name in unet_weights:
                param.data.copy_(unet_weights[name])

            elif "module."+name in unet_weights:
                param.data.copy_(unet_weights["module."+name])

        
        if cond_path is not None:
            cond_weights = load_file(cond_path)
            for name, param in self.cond_models.named_parameters():
                if name in cond_weights:
                    param.data.copy_(cond_weights[name])

                elif "module."+name in cond_weights:
                    param.data.copy_(cond_weights["module."+name])
            
        # Initialize for pretrained ema
        if self.use_ema:
            ema_weights  = load_file(ema_path)

            self.ema_unet = copy(self.unet)
            for name, param in self.ema_unet.named_parameters():
                if name in ema_weights:
                    param.data.copy_(ema_weights[name])

                elif "module."+name in ema_weights:
                    param.data.copy_(ema_weights["module."+name])

            self.ema_unet = EMAModel(
                self.ema_unet.parameters(),
                model_cls=UNet2DConditionModel,
                model_config=self.ema_unet.config,
                foreach=True,
            )
            del ema_weights
        
        if ipadaption_path is not None:
            ipadaption_weights = load_file(ipadaption_path)

            for name, param in self.ip_adaption_modules.named_parameters():
                if name in ipadaption_weights:
                    param.data.copy_(ipadaption_weights[name])

                elif "module."+name in ipadaption_weights:
                    param.data.copy_(ipadaption_weights["module."+name])
            del ipadaption_weights
            self.ip_adaption_modules.eval()


        self.unet.eval()
        self.vae.eval()
        self.cond_models.eval()
        
        # self.ema_unet.eval()

    def deepspeed_zero_init_disabled_context_manager(self):
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return deepspeed_plugin.zero3_init_context_manager(enable=False)