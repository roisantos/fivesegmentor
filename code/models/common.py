import torch.nn as nn
import torch.nn.functional as F
import torch
from timm.layers import  DropPath
import sys
import os
"""
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/config')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/datasets')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/evaluation')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/inference')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/models')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/scripts')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/utils')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/training')))

"""
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, ROOT_DIR)

from utils.utils import *

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, k_size = 3, dilation=1, layer_num=None):
        super(ConvBlock, self).__init__()
        self.init(in_channels, out_channels, stride, k_size, dilation, layer_num)
        
    def init(self, in_channels, out_channels, stride=1, k_size=3, dilation=1, layer_num=None):
        print("Function not implemented!")
        exit(-1)

class  ResidualBlock(ConvBlock):
    expansion = 1 # expansion coefficient
    def init(self, in_channels, out_channels, stride, k_size, dilation, layer_num=None):
        # super(ResidualBlock, self).__init__()
        p = k_size//2 * dilation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=k_size,
                               stride=stride, padding=p, bias=False, dilation=dilation)
        identifier = f"ResidualBlock{layer_num}_1stConvBlock"
        self.conv1.register_forward_hook(save_activation(f"{identifier}_forward"))
        self.conv1.register_full_backward_hook(save_gradient(f"{identifier}_backward"))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.the_bn1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=k_size,
                               stride=1, padding=p, bias=False, dilation=dilation)
        identifier2 = f"ResidualBlock{layer_num}_2ndConvBlock"
        self.conv2.register_forward_hook(save_activation(f"{identifier2}_forward"))
        self.conv2.register_full_backward_hook(save_gradient(f"{identifier2}_backward"))
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        # shortcut connection method
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1,
                          stride=stride,bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        # x = self.conv1(F.relu(self.the_bn1(x)))
        # x = self.conv2(F.relu(self.bn2(x)))
        # return x + residual

        x = x + residual
        return nn.ReLU()(x)


class RecurrentBlock(nn.Module):
    """
    Recurrent Block for R2Unet_CNN
    """
    def __init__(self, out_ch, k_size, t=2, groups = 1):
        super(RecurrentBlock, self).__init__()

        self.t = t
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=k_size, stride=1, padding=k_size//2, bias=False, groups=groups),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.conv.register_forward_hook(save_activation("RecurrentBlock"))
        self.conv.register_full_backward_hook(save_gradient("RecurrentBlock"))

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return (x1)

class RRCNNBlock(ConvBlock):
    """
    Recurrent Residual Convolutional Neural Network Block
    """
    def init(self, in_channels, out_channels, stride, k_size, dilation, layer_num):
        assert dilation==1
        t=2
        self.RCNN = nn.Sequential(
            RecurrentBlock(out_channels,k_size=k_size, t=t),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            RecurrentBlock(out_channels,k_size=k_size, t=t),
            nn.BatchNorm2d(out_channels),
        )
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU()
        )
        self.Conv.register_forward_hook(save_activation("RRCNNBlock"))
        self.Conv.register_full_backward_hook(save_gradient("RRCNNBlock"))

    def forward(self, x):
        x1 = self.Conv(x)
        x2 = self.RCNN(x1)
        out = x1 + x2
        return nn.ReLU()(out)


class RecurrentConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = RecurrentBlock(dim, k_size=7, groups=dim)
        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = RecurrentBlock(dim, k_size=3)
        # self.pwconv1 = nn.Conv2d(dim,4* dim, kernel_size=3, padding=1)# nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = RecurrentBlock(dim, k_size=3)
        # self.pwconv2 = nn.Conv2d(4*dim, dim, kernel_size=3, padding=1)#nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x = self.gamma * x
            x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        # x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
    
class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.dwconv.register_forward_hook(save_activation("ConvNeXtBlock"))
        self.dwconv.register_full_backward_hook(save_gradient("ConvNeXtBlock"))
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class SimpleResBlock(nn.Module):
    """
    Bloque simple estilo ResNet: una única convolución seguida de BatchNorm y ReLU.
    Se añade la conexión residual, y se incluye el parámetro 'layer_num' para compatibilidad.
    """
    def __init__(self, in_channels, out_channels, stride=1, k_size=9, layer_num=None):
        super(SimpleResBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k_size,
                              stride=stride, padding=k_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        # Cambiamos inplace a False
        self.relu = nn.ReLU(inplace=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        # Evitar suma inplace:
        out = out + self.shortcut(x)
        return out



class SumMergeBlock(nn.Module):
    """
    Bloque para fusionar dos tensores mediante suma seguida de refinamiento.
    Se espera que la suma se realice sobre tensores de igual dimensión.
    """
    def __init__(self, channels_in, channels_out):
        super(SumMergeBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):
    """
    Módulo de atención para skip connections.
    Recibe una skip connection (x) y una señal de gating (g) y
    genera una versión ponderada de la skip connection.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        # x: características de la skip connection
        # g: señal de gating (por ejemplo, salida del decoder tras upsampling)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi