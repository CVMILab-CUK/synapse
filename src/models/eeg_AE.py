import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from libs import Conv1dLayer, LinearLayer, ResnetBlock, TransformerBlock, ConvNeXtV2Block, DownSampleBlock, UpSampleBlock
from libs.norm import trunc_normal_
from libs.utils import  positional_encoding



class eeg_encoder(nn.Module):
    def __init__(self, 
                in_seq:int              = 440,
                in_channels:int         = 128,
                out_channels:int        = 1,
                out_seq:int             = 768,
                dims:List               = [64, 128, 256, 512, 1024],
                shortcut:bool           = True,
                dropout:float           = 0.0,
                groups:int              = 32,
                layer_mode:str          = 'conv',
                block_mode:str          = 'res',
                down_mode:str           = 'max',
                pos_mode:str            = 'sinusoidal',
                n_layer:int             = 2,
                n_head:int              = 64,
                dff_factor:int          = 2,
                stride:int              = 4,
                skip_mode: str          = "conv",
                global_attn:bool        = False
                )->None:
        super().__init__()
        assert layer_mode in ['conv', 'linear'], f"Layer Mode Can be  Conv or Linaer Now : {layer_mode}"
        assert block_mode in ['res', 'conv'], f"Layer Mode Can be  Conv or Linaer Now : {block_mode}"
        assert down_mode in ["conv", "linear","avg", "max"], f"Mode is must be 'conv' or 'linear' or 'avg' or 'max'! Now:{down_mode}"
        assert pos_mode in ['sinusoidal', 'trunc'], f"Layer Mode Can be  Conv or Linaer Now : {pos_mode}"
        assert skip_mode in  ['conv', 'down', None], f"Skip Mode Can be conv or donw Now : {skip_mode}"
        self.global_attn = global_attn
        # Set Block's
        if block_mode == "res":
            block = partial(ResnetBlock, shortcut=shortcut, dropout=dropout, groups=groups,layer_mode=layer_mode)
        attn_block = partial(TransformerBlock,  num_heads=n_head, rate=dropout, ffn_rate=dropout)
        
        # Make Channel's List
        channels = [(dims[i], dims[i+1]) for i in range(len(dims) -1)]
        eeg_channels = [int((in_channels/2)//(stride**idx)) for idx in range(len(channels))]
        eeg_channels.insert(0, in_channels)


        # Set Positional Encoding
        if pos_mode == "sinusoidal":
            self.posEmbed  = [positional_encoding((in_channels/2)//(stride**idx), dim_out).transpose(2, 1) for idx, (dim_in, dim_out) in enumerate(channels)]
            self.posEmbed.insert(0, positional_encoding(in_channels, dims[0]).transpose(2,1))

        elif pos_mode == "trunc":
            self.posEmbed  = [trunc_normal_(nn.Parameter(torch.zeros([1, dim_out, int((in_channels/2)//(stride**idx))])), std=.02) for idx, (dim_in, dim_out) in enumerate(channels)]
            self.posEmbed.insert(0, trunc_normal_(nn.Parameter(torch.zeros(1, dims[0], in_channels)), std=.02))
            self.posEmbed = nn.ParameterList(self.posEmbed)

        # Set Layer's
        if layer_mode == "linear":
            self.posEmbed = [embed.transpose(2, 1) for embed in self.posEmbed]
            self.in_layer = LinearLayer(in_dim=in_seq, out_dim=dims[0])
            self.out_layer = LinearLayer(in_dim=dims[-1], out_dim=out_seq)
        else:
            self.in_layer = Conv1dLayer(in_channels=in_seq, out_channels=dims[0], kernel_size=1, bias=True)
            self.out_layer = Conv1dLayer(in_channels=dims[-1], out_channels=out_seq, kernel_size=1, bias=True)
        # Set Skip Mode
        if skip_mode == "conv":
            self.skips = nn.ModuleList([Conv1dLayer(dims[0], dims[0], 1)])
            self.skips_agg = None        
        elif skip_mode == "down":
            self.skips = nn.ModuleList([Conv1dLayer(dims[0], out_seq, 1)])
            layer_shape = int(sum([in_channels]+[(in_channels/2)//(stride**idx) for idx in range(len(channels))]))+1
            self.skips_agg = nn.Sequential(
                LinearLayer(layer_shape, 1024),
                nn.ReLU(),
                LinearLayer(1024, 1024),
                nn.ReLU(),
                LinearLayer(1024, out_channels)
            )
            if global_attn:
                self.skips_agg.append(attn_block(out_seq, out_seq, dff=out_channels*dff_factor, in_channels=out_channels))
            
        else:
            self.skips= None
            self.skips_agg = None
        
        # Set Down Sample Block
        downsample = partial(DownSampleBlock, mode=down_mode, stride=stride)
        
        
        # Set Main Flow's First Block
        proj =[block(dims[0], dims[0]) for _ in range(n_layer)]
        proj.append(attn_block(dims[0], dims[0], dff=dims[0] * dff_factor, in_channels=in_channels))
        proj = nn.ModuleList(proj)
        
        # Set Main Flow's Other Block
        self.main_flow = nn.ModuleList([proj])

        if stride == 4:
            self.downs     = nn.ModuleList([downsample(dims[0], dims[0], stride=stride//2)]) if not down_mode == "linear" else nn.ModuleList([downsample(eeg_channels[0], eeg_channels[1], stride=stride)])
        else:
            self.downs     = nn.ModuleList([downsample(dims[0], dims[0])])

        # down 1
        in_channels /= 2


        for idx, (dim_in, dim_out) in enumerate(channels):
            # Expand Dimension Block Mustbe Included
            proj = nn.ModuleList([])
            proj.append(block(dim_in, dim_out))

            # Add additional Blocks
            for _ in range(n_layer-1):
                proj.append(block(dim_out, dim_out))
            
            # Add Attention
            proj.append(attn_block(dim_out, dim_out, dff= dim_out*dff_factor, in_channels=int(in_channels/(stride**idx))))

            # Make Main Flow and downsamples
            self.main_flow.append(proj)
            
            if skip_mode == "conv":
                self.skips.append(Conv1dLayer(dim_out, dim_out, 1, 1))                
            elif skip_mode == "down":
                self.skips.append(Conv1dLayer(dim_out, out_seq, 1, 1))
    
            if idx +1 == len(channels):
                self.downs.append(nn.Identity())
            else:
                if down_mode == "linear":
                    self.downs.append(downsample(eeg_channels[idx+1], eeg_channels[idx+2]))
                else:
                    self.downs.append(downsample(dim_out, dim_out))
                # If add Skip connection

    def forward(self, x):
        device = x.get_device()
        h = x

        # print("inputs : ",h.shape)
        h = self.in_layer(h)
        skip_h = []
        # print(f"After In Layer : {h.shape}")

        for idx, (proj, down) in enumerate(zip(self.main_flow, self.downs)):
            
            # Main Block
            for block_idx in range(len(proj)):
                # If block_idx +1 == len(proj) is mean, it is last block for one dimension => Transformer Block
                # So, Add Positional Encoding and run transformer blocks
                if (block_idx +1) == len(proj):
                    h += self.posEmbed[idx].to(device)
                    h, attn = proj[block_idx](h)
                    # print(f"BLOCK {idx} after attn : {h.shape}")  

                else:
                    h = proj[block_idx](h)
                    # print(f"BLOCK {idx} after block{block_idx} : {h.shape}")

            if self.skips:
                skip_h.append(self.skips[idx](h))
            # DownSampling
            h = down(h)
            # print(f"BLOCK {idx} after down block : {h.shape}")

        h = self.out_layer(h)
        if self.skips_agg:
            skip_h.append(h)
            h = self.skips_agg(torch.cat(skip_h, 2))
            if self.global_attn:
                h, _ = h
        else:
            skip_h.reverse()
        # print(f"After Out Layer : {h.shape}")
        return h, skip_h

class eeg_decoder(nn.Module):
    def __init__(self, 
                in_seq:int              = 768,
                in_channels:int         = 1,
                out_channels:int        = 128,
                out_seq:int             = 440,
                dims:List               = [64, 128, 256, 512, 1024],
                shortcut:bool           = True,
                dropout:float           = 0.0,
                groups:int              = 32,
                layer_mode:str          = 'conv',
                block_mode:str          = 'res',
                up_mode:str             = "trans",
                pos_mode:str            = 'sinusoidal',
                n_layer:int             = 2,
                n_head:int              = 64,
                dff_factor:int          = 2,
                stride:int              = 4,
                skip_mode:str           = "conv"
            ):
        super().__init__()
        assert layer_mode in ['conv', 'linear'], f"Layer Mode Can be  Conv or Linaer Now : {layer_mode}"
        assert block_mode in ['res', 'conv'], f"Layer Mode Can be  Conv or Linaer Now : {block_mode}"
        assert up_mode in ["trans", "near", "conv", "linear"], f"Mode is must be 'trans' or 'near', 'conv'! Now:{up_mode}" 
        assert pos_mode in ['sinusoidal', 'trunc'], f"Layer Mode Can be  Conv or Linaer Now : {pos_mode}"
        assert skip_mode in  ['conv', 'down', None], f"Skip Mode Can be conv or donw Now : {skip_mode}"
        
        #Set Block's
        if block_mode == "res":
            block = partial(ResnetBlock, shortcut=shortcut, dropout=dropout, groups=groups,layer_mode=layer_mode)
        attn_block = partial(TransformerBlock,  num_heads=n_head, rate=dropout, ffn_rate=dropout)
        
        # Make Channel's List     
        channels = [(dims[i+1], dims[i]) for i in range(len(dims) -1)]
        
        # Set Seqeucne
        seqeunces = [int((out_channels/2)//(stride**idx)) for idx in range(len(channels)-1)]
        seqeunces.insert(0, out_channels)
        seqeunces.append(in_channels)

        # Set Positional Encoding
        if pos_mode == "sinusoidal":
            self.posEmbed  = [positional_encoding(seqeunces[idx+1], dim_in).transpose(2, 1) for idx, (dim_in, dim_out) in enumerate(channels)]
            self.posEmbed.insert(0, positional_encoding(out_channels, dims[0]).transpose(2,1))

        elif pos_mode == "trunc":
            self.posEmbed  = [trunc_normal_(nn.Parameter(torch.zeros([1, dim_in, seqeunces[idx+1]])), std=.02) for idx, (dim_in, dim_out) in enumerate(channels)]
            self.posEmbed.insert(0, trunc_normal_(nn.Parameter(torch.zeros(1, dims[0], out_channels)), std=.02))
            
        self.posEmbed.reverse()
        channels.reverse()
        seqeunces.reverse()
        self.posEmbed = nn.ParameterList(self.posEmbed)
        # self.posEmbed = [s.cuda() for s in self.posEmbed]

        # Set Layer's
        if layer_mode == "linear":
            self.posEmbed = [embed.transpose(2, 1) for embed in self.posEmbed]
            self.in_layer = LinearLayer(in_dim=in_seq, out_dim=dims[-1])
            self.out_layer = LinearLayer(in_dim=dims[0], out_dim=out_seq)
        else:
            self.in_layer = Conv1dLayer(in_channels=in_seq, out_channels=dims[-1], kernel_size=1, bias=True)
            self.out_layer = Conv1dLayer(in_channels=dims[0], out_channels=out_seq, kernel_size=1, bias=True)   

        self.skip_mode = skip_mode  
        # self.posEmbed = nn.ModuleList(self.posEmbed)


        # Set up Sample Block
        # Strdie 4 mean, 4x upsamle
        if up_mode == "trans":
            if stride==4:
                upsample = partial(UpSampleBlock, mode=up_mode, kernel_size=8, stride=4, padding=2)
            else:
                upsample = partial(UpSampleBlock, mode=up_mode, kernel_size=4, stride=2, padding=1)
        else:
            upsample = partial(UpSampleBlock, mode=up_mode, stride=stride)

        #  # Set Main Flow's First Block
        if skip_mode == "conv":
            chan = dims[-1] *2
        else:
            chan = dims[-1]

        proj =  nn.ModuleList([block(chan, dims[-1])])
        for _ in range(n_layer -1):
            proj.append(block(dims[-1], dims[-1]))
        proj.append(attn_block(dims[-1], dims[-1], dff=dims[-1] * dff_factor, in_channels=seqeunces[0]))
        
        self.main_flow = nn.ModuleList([proj])
        self.ups       = nn.ModuleList([])
        for idx, (dim_in, dim_out) in enumerate(channels):
            is_last = (idx +1) == len(channels)
            # dim_dff = dim_in * 2 if skip_mode == "conv" else dim_in       # If skip to concat in channel pow 2 
            # # print(dim_dff)
            # Expand Dimension Block Mustbe Included
            proj = nn.ModuleList([])
            proj.append(block(dim_in, dim_out))

            # Add additional Blocks
            for _ in range(n_layer-1):
                proj.append(block(dim_out, dim_out))
            
            # Add Attention
            proj.append(attn_block(dim_out, dim_out, dff= dim_out*dff_factor, in_channels=seqeunces[idx+1]))
            if up_mode == "linear":
                ups_in  = seqeunces[idx] 
                ups_out = seqeunces[idx+1]
            else:
                ups_in = dim_in
                ups_out = dim_out if skip_mode == "conv" else dim_in
            
            if not is_last:
                 # Make upsample
                self.ups.append(upsample(ups_in, ups_out))
            else:
                # Make upsample
                self.ups.append(upsample(ups_in, ups_out, kernel_size=4, stride=2, padding=1))


            # Make Main Flow 
            self.main_flow.append(proj)

        # Set Last Upsample
        if stride ==4:
            self.ups.append(nn.Identity())
        else:
            self.ups.append(upsample(stride=stride))
        
    def forward(self, x, skips=None):
        device = x.get_device()
        h = x
        # print("inputs : ",h.shape)
        h = self.in_layer(h)
        # print(f"After In Layer : {h.shape}")
        for idx, (proj, up) in enumerate(zip(self.main_flow, self.ups)):  
            if skips:
                h = torch.cat([h,skips[idx]], dim=1)          
            # print(f"BLOCK {idx} after concat : {h.shape}")
            # Main Block
            for block_idx in range(len(proj)):
                # If block_idx +1 == len(proj) is mean, it is last block for one dimension => Transformer Block
                # So, Add Positional Encoding and run transformer blocks
                if (block_idx +1) == len(proj):
                    h += self.posEmbed[idx].to(device)
                    h, attn = proj[block_idx](h)  
                    # print(f"BLOCK {idx} after attn : {h.shape}")      
                else:
                    h = proj[block_idx](h)        
                    # print(f"BLOCK {idx} after block{block_idx} : {h.shape}")
    
            # UpSampling
            h = up(h)
            # print(f"BLOCK {idx} after up block : {h.shape}")
        h = self.out_layer(h)
        # print(f"After Out Layer : {h.shape}")
        return h

class eeg_AutoEncoder(nn.Module):

    def __init__(self, 
                in_seq:int              = 440,
                in_channels             = 128,
                z_channels:int          = 1,
                out_seq:int             = 768,
                dims:List               = [64, 128, 256, 512, 1024],
                shortcut:bool           = True,
                dropout:float           = 0.0,
                groups:int              = 32,
                layer_mode:str          = 'conv',
                block_mode:str          = 'res',
                down_mode:str           = 'max',
                up_mode:str             = "trans",
                pos_mode:str            = 'sinusoidal',
                skip_mode:str           = "conv",
                n_layer:int             = 2,
                n_head:int              = 64,
                dff_factor:int          = 2,
                stride:int              = 4,
                global_attn:bool        = False
                ):
        super().__init__()
        self.skip_mode = skip_mode
        
        self.Encoder = eeg_encoder( in_seq = in_seq, in_channels = in_channels, out_channels=z_channels,out_seq = out_seq,
                dims = dims, shortcut = shortcut, dropout = dropout, groups = groups,
                layer_mode = layer_mode,  block_mode = block_mode, down_mode = down_mode,
                pos_mode = pos_mode, n_layer =n_layer, n_head = n_head, dff_factor= dff_factor,
                stride = stride, skip_mode= skip_mode, global_attn=global_attn)
        
        self.Decoder = eeg_decoder(in_seq = out_seq,in_channels=z_channels, out_channels = in_channels, out_seq = in_seq,
                dims = dims, shortcut = shortcut, dropout = dropout, groups = groups,
                layer_mode = layer_mode,  block_mode = block_mode, up_mode = up_mode,
                pos_mode = pos_mode, n_layer =n_layer, n_head = n_head, dff_factor= dff_factor,
                stride = stride, skip_mode= skip_mode)
        

    def forward(self, x, skips=False):
        z, skips = self.Encoder(x)
        if self.skip_mode == "conv":
            rec = self.Decoder(z, skips)
        else:
            rec = self.Decoder(z)

        if skips:
            return z, rec
        
        return z, rec



if __name__ == "__main__":
    en = eeg_encoder()
    # print([pos.shape for pos in en.posEmbed])

    inputs = torch.randn(1, 440, 128)
    outputs = en(inputs)
    # print(outputs.shape)