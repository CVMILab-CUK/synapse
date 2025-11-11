from libs.layers import *
from libs.norm   import *
from libs.utils import exists


from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_, DropPath


class ConvNeXtV2Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = Conv1dLayer(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN1D(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ResnetBlock(nn.Module):
    r"""
    Resnet block 3d code 
    Original :
        https://github.com/huggingface/diffusers/blob/180841bbde4b200be43350164eef80c93a68983a/src/diffusers/models/resnet.py#L367
    """
    def __init__(
        self,
        in_channels,
        out_channels        = None,
        shortcut            = False,
        dropout             = 0.0,
        groups              = 32,
        eps                 = 1e-6,
        output_scale_factor = 1.0,
        layer_mode          = 'conv'
        ):
        super().__init__()
        layer_mode = layer_mode.lower()
        assert layer_mode in ['conv', 'linear'], f"Layer Mode Can be  Conv or Linaer Now : {layer_mode}"

        self.in_channels         = in_channels
        out_channels             = in_channels if out_channels is None else out_channels
        self.out_channels        = out_channels
        self.conv_shortcut       = None
        self.output_scale_factor = output_scale_factor        

        if layer_mode == "conv":
            layer = partial(Conv1dLayer,  kernel_size=1)
        else:
            layer = partial(LinearLayer) 

        self.act_fn = nn.SiLU()
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv1 = layer(in_channels, out_channels)
        self.conv2 = layer(out_channels, out_channels)
        self.dropout =  nn.Dropout(dropout)

        if shortcut:
            self.shortcut = Conv1dLayer(in_channels, out_channels, 1)


    def forward(self, x, time_emb=None):
        h = x
        h = self.norm1(h)
        h = self.act_fn(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act_fn(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.shortcut is not None:
            x = self.shortcut(x)
        return (x + h) / self.output_scale_factor


class DownSampleBlock(nn.Module):

    def __init__(self, 
                 in_channels:int        = None,
                 out_channels:int       = None,
                 mode:str               = "conv", 
                 stride:int             = 4,
                 ):
        super().__init__()
        mode = mode.lower()
        out_channels = in_channels if out_channels is None else out_channels
        assert mode in ["conv", "linear","avg", "max"], f"Mode is must be 'conv' or 'linear' or 'avg' or 'max'! Now:{mode}"
        if mode == "conv":
            self.proj = Conv1dLayer(in_channels, out_channels, stride, stride)
        elif mode == "linear":
            self.proj = LinearLayer(in_channels, out_channels)
        elif mode  == "avg":
            self.proj = nn.AvgPool1d(stride, stride)
        else:
            self.proj = nn.MaxPool1d(stride,  stride)
    
    def forward(self, x):
        return self.proj(x)

class UpSampleBlock(nn.Module):

    def __init__(self, 
                 in_channels:int  = None,
                 out_channels:int =None, 
                 mode:str="trans", 
                 kernel_size:int=4,
                 stride:int = 2,
                 padding:int= 1):
        super().__init__()
        self.mode = mode.lower()
        out_channels = in_channels if out_channels is None else out_channels
        assert self.mode in ["trans", "near", "conv", "linear"], f"Mode is must be 'trans' or 'near', 'conv'! Now:{self.mode}" 
        if self.mode == "trans":
            self.proj = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        elif self.mode == "near":
            self.proj = partial(F.interpolate, scale_factor=stride, mode="nearest")
        elif mode == "linear":
            self.proj = LinearLayer(in_channels, out_channels)
        else: 
            self.proj = partial(F.interpolate, scale_factor=stride, mode="nearest")
            proj = []
            proj.append(nn.ConstantPad1d(padding, 0.0))
            proj.append(nn.Conv1d(in_channels, out_channels, 3))
            self.conv = nn.Sequential(*proj)
    
    def forward(self, x):
        x  = self.proj(x)
        if not self.mode == "conv":
            return x
        return self.conv(x)
    
class TransformerBlock(nn.Module):

    def __init__(self,
                d_embed,
                d_model,
                num_heads,
                dff,
                in_channels:int=128,
                rate=0.1,
                ffn_rate=0.5,
                eps=1e-6,
                out_attn=True):

        super(TransformerBlock, self).__init__()
        self.out_attn = out_attn
        
        self.mha = EEGAttention1dLayer(dim_in=d_embed, dim_out=d_model, num_heads=num_heads) # dim_in, dim_out, heand num
        self.ffn = PositionWiseConv1dLayer(dim=d_model, dff=dff, rate=ffn_rate)

        self.layernorm1 = nn.LayerNorm([d_model, in_channels], eps=eps)
        self.layernorm2 = nn.LayerNorm([d_model, in_channels], eps=eps)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, mask=None):
        x = self.layernorm1(x)
        attn_output, attn_weight = self.mha(x, x, x, mask)
        attn_output    = self.dropout1(attn_output)
        out1           = self.layernorm2(x + attn_output)

        ffn_output     = self.ffn(out1)
        ffn_output     = self.dropout2(ffn_output)
        out2           = out1 + ffn_output
        if self.out_attn:
            return out2, attn_weight
        else:
            return out2
        

class IPAdaptationBlock(nn.Module):
    r"""
    This is IP Adatation Block for EEG Data
    """
    def __init__(self,
                d_embed,
                d_model,
                num_heads,
                dff,
                in_channels:int=128,
                rate=0.1,
                ffn_rate=0.5,
                out_attn=False, 
                ip_adapter_token_num=4):
        super(IPAdaptationBlock, self).__init__()
        self.num_tokens=ip_adapter_token_num

        # For Image Information
        self.adaptation = nn.Sequential(
            nn.Linear(d_model, self.num_tokens * d_model),
            Rearrange('b (n e) -> b n e', n=self.num_tokens),
        )
        self.rearrage    = Rearrange('b c s -> b s c')


        # For Adaptive Generalization
        # self.transformer = TransformerBlock(d_embed, d_model, num_heads, dff, in_channels+self.num_tokens, rate, ffn_rate, out_attn=out_attn)
        self.transformer = TransformerBlock(d_embed, d_model, num_heads, dff, in_channels, rate, ffn_rate, out_attn=out_attn)
    
    def forward(self, x):
        eeg_img_embed = x.mean(dim=1)
        eeg_img_embed = self.adaptation(eeg_img_embed)

        # eeg_condition_vector = torch.cat([x, eeg_img_embed], dim=1)
        # eeg_condition_vector = self.rearrage(eeg_condition_vector)
        eeg_condition_vector = self.rearrage(x)
        eeg_condition_vector = self.transformer(eeg_condition_vector)
        eeg_condition_vector = self.rearrage(eeg_condition_vector)
        eeg_condition_vector = torch.cat([eeg_condition_vector, eeg_img_embed], dim=1)

        return eeg_condition_vector