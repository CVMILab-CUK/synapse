import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from .attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor


class EEGAdapter(nn.Module):
    def __init__(self, 
                 unet, 
                 clip_embeddings_dim:int=1024, 
                 cross_attention_dim:int=1024, 
                 clip_extra_context_tokens:int=4,
                 scale:float=1.0)->None:
        super().__init__()

        self.unet = unet 
        self.clip_extra_context_tokens = clip_extra_context_tokens

        # Make Aaption Network
        self.ip_proj = nn.Sequential(
            nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim),
            Rearrange('b (n e) -> b n e', n=self.clip_extra_context_tokens),
            # nn.LayerNorm(cross_attention_dim)
        )
        
        # Setting Unet for IPAdaption
        self.set_unet()
        
    def set_unet(self):
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
        self.adapter_modules = torch.nn.ModuleList(self.unet.attn_processors.values())
        
    def forward(self, noisy_latents, timesteps, eeg_condition_vector, eeg_image_embed, return_dict=False):
        ip_tokens = self.ip_proj(eeg_image_embed)
        encoder_hidden_states = torch.cat([eeg_condition_vector, ip_tokens], dim=1)
        pred = self.unet(noisy_latents,timesteps, encoder_hidden_states, return_dict=False)[0]
        return pred