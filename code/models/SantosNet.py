#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set ROOT_DIR and update sys.path so that we can import common blocks
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, ROOT_DIR)
from models.common import *  # This includes ResidualBlock and any other shared modules


###############################################################
# 1. Define the three new network classes
###############################################################

# The new models replicate the RoiNet architecture but start with a channel conversion block.
# In the rest of the network, the effective input channel is 1.

# 1.1 SantosNet_GCh: Uses only the green channel.
class SantosNet_GCh(nn.Module):
    def __init__( self, ch_out, ls_mid_ch=[32, 64, 128, 128, 64, 32],
                 k_size=9, ch_in=3, cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock):
        super(SantosNet_GCh, self).__init__()
        # Initial module: select only the green channel.
        self.channel_convert = GreenChannelBlock()
        # Now, effective input channels is 1.
        ch_in = 1
        
        # Replicate the RoiNet architecture using ch_in=1.
        self.dict_module = nn.ModuleDict()
        # Encoder
        self.dict_module.add_module("conv0", cls_init_block(ch_in, ls_mid_ch[0], k_size=k_size, layer_num=0))
        ch = ls_mid_ch[0]
        self.dict_module.add_module("conv1", cls_init_block(ch, ls_mid_ch[1], k_size=k_size, layer_num=1))
        ch = ls_mid_ch[1]
        self.dict_module.add_module("pool1", nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(ch, ch * 2, kernel_size=1)
        ))
        ch = ch * 2
        self.dict_module.add_module("conv2", cls_init_block(ch, ls_mid_ch[2], k_size=k_size, layer_num=2))
        ch = ls_mid_ch[2]
        self.dict_module.add_module("pool2", nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(ch, ch * 2, kernel_size=1)
        ))
        ch = ch * 2
        # Bottleneck
        self.dict_module.add_module("bottle1", cls_conv_block(ch, ch, k_size=k_size, layer_num="bottle1"))
        self.dict_module.add_module("bottle2", cls_conv_block(ch, ch, k_size=k_size, layer_num="bottle2"))
        self.dict_module.add_module("merge2", nn.Sequential(
            nn.Conv2d(ch * 2, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        ))
        # Decoder
        self.dict_module.add_module("conv3", cls_init_block(ch, ls_mid_ch[3], k_size=k_size, layer_num=3))
        ch = ls_mid_ch[3]
        self.dict_module.add_module("up3", nn.Sequential(
            nn.ConvTranspose2d(ch, ch // 2, kernel_size=2, stride=2)
        ))
        ch = ch // 2
        self.dict_module.add_module("merge3", nn.Sequential(
            nn.Conv2d((ls_mid_ch[3] // 2) + (ls_mid_ch[1] * 2), ls_mid_ch[1],
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ls_mid_ch[1]),
            nn.ReLU(inplace=True)
        ))
        ch = ls_mid_ch[1]
        self.dict_module.add_module("conv4", cls_init_block(ch, ls_mid_ch[4], k_size=k_size, layer_num=4))
        ch = ls_mid_ch[4]
        self.dict_module.add_module("up4", nn.Sequential(
            nn.ConvTranspose2d(ch, ch // 2, kernel_size=2, stride=2)
        ))
        ch = ch // 2
        self.dict_module.add_module("merge4", nn.Sequential(
            nn.Conv2d((ls_mid_ch[4] // 2) + ls_mid_ch[0], ls_mid_ch[0],
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ls_mid_ch[0]),
            nn.ReLU(inplace=True)
        ))
        ch = ls_mid_ch[0]
        self.dict_module.add_module("conv5", cls_init_block(ch, ls_mid_ch[5], k_size=k_size, layer_num=5))
        ch = ls_mid_ch[5]
        self.dict_module.add_module("final", nn.Sequential(
            nn.Conv2d(ch, ch_out, kernel_size=1, bias=False),
            nn.Sigmoid()
        ))
        self.ls_mid_ch = ls_mid_ch

    def forward(self, x):
        # First, convert 3-channel input to 1-channel using the green channel.
        #print("Output shape before the conversion block:", x.shape)
        x = self.channel_convert(x)
        # Then, propagate through the network.
        #print("Output shape after the conversion block:", x.shape)
        out0 = self.dict_module["conv0"](x)
        out1 = self.dict_module["conv1"](out0)
        out1 = self.dict_module["pool1"](out1)
        skip1 = out1
        out2 = self.dict_module["conv2"](out1)
        out2 = self.dict_module["pool2"](out2)
        skip2 = out2
        bottle1 = self.dict_module["bottle1"](out2)
        bottle2 = self.dict_module["bottle2"](bottle1)
        bottle_cat = torch.cat([bottle2, skip2], dim=1)
        bottle_out = self.dict_module["merge2"](bottle_cat)
        out3 = self.dict_module["conv3"](bottle_out)
        out3 = self.dict_module["up3"](out3)
        out3 = torch.cat([out3, skip1], dim=1)
        out3 = self.dict_module["merge3"](out3)
        out4 = self.dict_module["conv4"](out3)
        out4 = self.dict_module["up4"](out4)
        out4 = torch.cat([out4, out0], dim=1)
        out4 = self.dict_module["merge4"](out4)
        out5 = self.dict_module["conv5"](out4)
        final = self.dict_module["final"](out5)
        return final

# 1.2 SantosNet_PCh: Uses a learnable fusion block (a 1x1 conv) to combine channels.
class SantosNet_PCh(nn.Module):
    def __init__(self, ch_out, ls_mid_ch=[32, 64, 128, 128, 64, 32],
                 k_size=9, ch_in=3, cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock):
        super(SantosNet_PCh, self).__init__()
        # Initial learnable fusion block: a 1x1 convolution that converts 3 channels to 1.
        self.channel_convert = LearnableFusionBlock()
        ch_in = 1  # effective channels after fusion
        self.dict_module = nn.ModuleDict()
        # (The rest of the network architecture is identical to RoiNet with ch_in=1)
        self.dict_module.add_module("conv0", cls_init_block(ch_in, ls_mid_ch[0], k_size=k_size, layer_num=0))
        ch = ls_mid_ch[0]
        self.dict_module.add_module("conv1", cls_init_block(ch, ls_mid_ch[1], k_size=k_size, layer_num=1))
        ch = ls_mid_ch[1]
        self.dict_module.add_module("pool1", nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(ch, ch * 2, kernel_size=1)
        ))
        ch = ch * 2
        self.dict_module.add_module("conv2", cls_init_block(ch, ls_mid_ch[2], k_size=k_size, layer_num=2))
        ch = ls_mid_ch[2]
        self.dict_module.add_module("pool2", nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(ch, ch * 2, kernel_size=1)
        ))
        ch = ch * 2
        self.dict_module.add_module("bottle1", cls_conv_block(ch, ch, k_size=k_size, layer_num="bottle1"))
        self.dict_module.add_module("bottle2", cls_conv_block(ch, ch, k_size=k_size, layer_num="bottle2"))
        self.dict_module.add_module("merge2", nn.Sequential(
            nn.Conv2d(ch * 2, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        ))
        self.dict_module.add_module("conv3", cls_init_block(ch, ls_mid_ch[3], k_size=k_size, layer_num=3))
        ch = ls_mid_ch[3]
        self.dict_module.add_module("up3", nn.Sequential(
            nn.ConvTranspose2d(ch, ch // 2, kernel_size=2, stride=2)
        ))
        ch = ch // 2
        self.dict_module.add_module("merge3", nn.Sequential(
            nn.Conv2d((ls_mid_ch[3] // 2) + (ls_mid_ch[1] * 2), ls_mid_ch[1],
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ls_mid_ch[1]),
            nn.ReLU(inplace=True)
        ))
        ch = ls_mid_ch[1]
        self.dict_module.add_module("conv4", cls_init_block(ch, ls_mid_ch[4], k_size=k_size, layer_num=4))
        ch = ls_mid_ch[4]
        self.dict_module.add_module("up4", nn.Sequential(
            nn.ConvTranspose2d(ch, ch // 2, kernel_size=2, stride=2)
        ))
        ch = ch // 2
        self.dict_module.add_module("merge4", nn.Sequential(
            nn.Conv2d((ls_mid_ch[4] // 2) + ls_mid_ch[0], ls_mid_ch[0],
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ls_mid_ch[0]),
            nn.ReLU(inplace=True)
        ))
        ch = ls_mid_ch[0]
        self.dict_module.add_module("conv5", cls_init_block(ch, ls_mid_ch[5], k_size=k_size, layer_num=5))
        ch = ls_mid_ch[5]
        self.dict_module.add_module("final", nn.Sequential(
            nn.Conv2d(ch, ch_out, kernel_size=1, bias=False),
            nn.Sigmoid()
        ))
        self.ls_mid_ch = ls_mid_ch

    def forward(self, x):
        # Apply the learnable fusion: converts input from 3 channels to 1 channel.
        #print("Output shape before the conversion block:", x.shape)
        x = self.channel_convert(x)
        #print("Output shape after the conversion block:", x.shape)
        # Then, same forward pass as in SantosNet_GCh.
        out0 = self.dict_module["conv0"](x)
        out1 = self.dict_module["conv1"](out0)
        out1 = self.dict_module["pool1"](out1)
        skip1 = out1
        out2 = self.dict_module["conv2"](out1)
        out2 = self.dict_module["pool2"](out2)
        skip2 = out2
        bottle1 = self.dict_module["bottle1"](out2)
        bottle2 = self.dict_module["bottle2"](bottle1)
        bottle_cat = torch.cat([bottle2, skip2], dim=1)
        bottle_out = self.dict_module["merge2"](bottle_cat)
        out3 = self.dict_module["conv3"](bottle_out)
        out3 = self.dict_module["up3"](out3)
        out3 = torch.cat([out3, skip1], dim=1)
        out3 = self.dict_module["merge3"](out3)
        out4 = self.dict_module["conv4"](out3)
        out4 = self.dict_module["up4"](out4)
        out4 = torch.cat([out4, out0], dim=1)
        out4 = self.dict_module["merge4"](out4)
        out5 = self.dict_module["conv5"](out4)
        final = self.dict_module["final"](out5)
        return final

# 1.3 SantosNet_CPCh: Uses a custom fixed weighted fusion block.
class SantosNet_CPCh(nn.Module):
    def __init__(self, ch_out, custom_weights, ls_mid_ch=[32, 64, 128, 128, 64, 32],
                 k_size=9, ch_in=3, cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock):
        """
        custom_weights: a list or tuple of three numbers, e.g. [0.1, 0.8, 0.1]
        """
        super(SantosNet_CPCh, self).__init__()
        # Initial module: fixed weighted fusion from 3 channels to 1.
        self.channel_convert = CustomFusionBlock(custom_weights)
        ch_in = 1
        self.dict_module = nn.ModuleDict()
        # Replicate the RoiNet architecture with input channels = 1
        self.dict_module.add_module("conv0", cls_init_block(ch_in, ls_mid_ch[0], k_size=k_size, layer_num=0))
        ch = ls_mid_ch[0]
        self.dict_module.add_module("conv1", cls_init_block(ch, ls_mid_ch[1], k_size=k_size, layer_num=1))
        ch = ls_mid_ch[1]
        self.dict_module.add_module("pool1", nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(ch, ch * 2, kernel_size=1)
        ))
        ch = ch * 2
        self.dict_module.add_module("conv2", cls_init_block(ch, ls_mid_ch[2], k_size=k_size, layer_num=2))
        ch = ls_mid_ch[2]
        self.dict_module.add_module("pool2", nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(ch, ch * 2, kernel_size=1)
        ))
        ch = ch * 2
        self.dict_module.add_module("bottle1", cls_conv_block(ch, ch, k_size=k_size, layer_num="bottle1"))
        self.dict_module.add_module("bottle2", cls_conv_block(ch, ch, k_size=k_size, layer_num="bottle2"))
        self.dict_module.add_module("merge2", nn.Sequential(
            nn.Conv2d(ch * 2, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        ))
        self.dict_module.add_module("conv3", cls_init_block(ch, ls_mid_ch[3], k_size=k_size, layer_num=3))
        ch = ls_mid_ch[3]
        self.dict_module.add_module("up3", nn.Sequential(
            nn.ConvTranspose2d(ch, ch // 2, kernel_size=2, stride=2)
        ))
        ch = ch // 2
        self.dict_module.add_module("merge3", nn.Sequential(
            nn.Conv2d((ls_mid_ch[3] // 2) + (ls_mid_ch[1] * 2), ls_mid_ch[1],
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ls_mid_ch[1]),
            nn.ReLU(inplace=True)
        ))
        ch = ls_mid_ch[1]
        self.dict_module.add_module("conv4", cls_init_block(ch, ls_mid_ch[4], k_size=k_size, layer_num=4))
        ch = ls_mid_ch[4]
        self.dict_module.add_module("up4", nn.Sequential(
            nn.ConvTranspose2d(ch, ch // 2, kernel_size=2, stride=2)
        ))
        ch = ch // 2
        self.dict_module.add_module("merge4", nn.Sequential(
            nn.Conv2d((ls_mid_ch[4] // 2) + ls_mid_ch[0], ls_mid_ch[0],
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ls_mid_ch[0]),
            nn.ReLU(inplace=True)
        ))
        ch = ls_mid_ch[0]
        self.dict_module.add_module("conv5", cls_init_block(ch, ls_mid_ch[5], k_size=k_size, layer_num=5))
        ch = ls_mid_ch[5]
        self.dict_module.add_module("final", nn.Sequential(
            nn.Conv2d(ch, ch_out, kernel_size=1, bias=False),
            nn.Sigmoid()
        ))
        self.ls_mid_ch = ls_mid_ch

    def forward(self, x):
        # Apply fixed weighted fusion to reduce 3 channels to 1.
        #print("Output shape before the conversion block:", x.shape)
        x = self.channel_convert(x)
        #print("Output shape after the conversion block:", x.shape)
        out0 = self.dict_module["conv0"](x)
        out1 = self.dict_module["conv1"](out0)
        out1 = self.dict_module["pool1"](out1)
        skip1 = out1
        out2 = self.dict_module["conv2"](out1)
        out2 = self.dict_module["pool2"](out2)
        skip2 = out2
        bottle1 = self.dict_module["bottle1"](out2)
        bottle2 = self.dict_module["bottle2"](bottle1)
        bottle_cat = torch.cat([bottle2, skip2], dim=1)
        bottle_out = self.dict_module["merge2"](bottle_cat)
        out3 = self.dict_module["conv3"](bottle_out)
        out3 = self.dict_module["up3"](out3)
        out3 = torch.cat([out3, skip1], dim=1)
        out3 = self.dict_module["merge3"](out3)
        out4 = self.dict_module["conv4"](out3)
        out4 = self.dict_module["up4"](out4)
        out4 = torch.cat([out4, out0], dim=1)
        out4 = self.dict_module["merge4"](out4)
        out5 = self.dict_module["conv5"](out4)
        final = self.dict_module["final"](out5)
        return final





class SantosNet_GCh_lite(nn.Module):
    """
    Red Lite para imágenes de 512x512, utilizando solo el canal verde.
    
    Arquitectura:
      - Encoder: se utilizan bloques residuales simples con dos etapas de pooling.
      - Bottleneck: un bloque residual simple que se fusiona con la salida del pooling.
      - Decoder: se realizan dos etapas de upsampling (bilineal) seguidas de bloques que
        reducen el número de canales, fusionándose con skip connections (suma) provenientes
        del encoder.
      - Salida: convolución 1x1 seguida de Sigmoid.
    """
    def __init__(self, ch_out, ls_mid_ch=[16, 32, 64, 64, 32, 16], k_size=3):
        super(SantosNet_GCh_lite, self).__init__()
        # Conversión: de 3 canales a 1 usando solo el canal verde.
        self.channel_convert = GreenChannelBlock()
        
        # Encoder
        # Bloque 0: de 1 a 16 canales (salida completa; se usará en el decoder)
        self.conv0 = SimpleResBlock(1, ls_mid_ch[0], stride=1, k_size=k_size, layer_num=0)
        # Bloque 1: de 16 a 32 canales
        self.conv1 = SimpleResBlock(ls_mid_ch[0], ls_mid_ch[1], stride=1, k_size=k_size, layer_num=1)
        # Pooling: de 512x512 a 256x256
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Bloque 2: de 32 a 64 canales
        self.conv2 = SimpleResBlock(ls_mid_ch[1], ls_mid_ch[2], stride=1, k_size=k_size, layer_num=2)
        # Pooling: de 256x256 a 128x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        # Se aplica un bloque residual simple y se fusiona (por suma) con la salida previa al pooling.
        self.conv_bottleneck = SimpleResBlock(ls_mid_ch[2], ls_mid_ch[2], stride=1, k_size=k_size, layer_num="bottleneck")
        
        # Decoder
        # Primer upsampling: de 128x128 a 256x256
        # Se utiliza interpolación bilineal seguido de un bloque que reduce canales de 64 -> 32.
        self.conv_reduce_up1 = SimpleResBlock(ls_mid_ch[2], ls_mid_ch[1], stride=1, k_size=k_size, layer_num="upreduce1")
        # Se refina la fusión con la skip connection (proveniente de pool1) mediante otro bloque.
        self.conv_merge1 = SimpleResBlock(ls_mid_ch[1], ls_mid_ch[1], stride=1, k_size=k_size, layer_num="merge1")
        
        # Segundo upsampling: de 256x256 a 512x512
        # Se reduce de 32 canales a 16 para fusionarlo con la salida del Bloque 0.
        self.conv_reduce_up2 = SimpleResBlock(ls_mid_ch[1], ls_mid_ch[5], stride=1, k_size=k_size, layer_num="upreduce2")
        self.conv_merge2 = SimpleResBlock(ls_mid_ch[5], ls_mid_ch[5], stride=1, k_size=k_size, layer_num="merge2")
        
        # Capa final: conv 1x1 para mapear a ch_out y función Sigmoid para normalizar la salida.
        self.conv_final = nn.Sequential(
            nn.Conv2d(ls_mid_ch[5], ch_out, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        self.ls_mid_ch = ls_mid_ch

    def forward(self, x):
        # 1. Conversión del canal: (B, 3, 512, 512)  (B, 1, 512, 512)
        x = self.channel_convert(x)
        
        # 2. Encoder
        out0 = self.conv0(x)         # (B, 16, 512, 512)  se usará en el decoder (skip0)
        out1 = self.conv1(out0)        # (B, 32, 512, 512)
        skip1 = self.pool1(out1)       # (B, 32, 256, 256)  skip connection para el decoder
        out2 = self.conv2(skip1)       # (B, 64, 256, 256)
        skip2 = self.pool2(out2)       # (B, 64, 128, 128)
        
        # 3. Bottleneck
        bottle = self.conv_bottleneck(skip2)  # (B, 64, 128, 128)
        bottle = bottle + skip2               # Fusión mediante suma
        
        # 4. Decoder
        # Primer upsampling: se lleva de 128x128 a 256x256
        up1 = F.interpolate(bottle, scale_factor=2, mode='bilinear', align_corners=True)  # (B, 64, 256, 256)
        up1 = self.conv_reduce_up1(up1)   # Reduce a 32 canales  (B, 32, 256, 256)
        merge1 = up1 + skip1              # Fusiona con skip1 (suma elementwise)
        merge1 = self.conv_merge1(merge1) # Refinamiento: (B, 32, 256, 256)
        
        # Segundo upsampling: se lleva de 256x256 a 512x512
        up2 = F.interpolate(merge1, scale_factor=2, mode='bilinear', align_corners=True)  # (B, 32, 512, 512)
        up2 = self.conv_reduce_up2(up2)   # Reduce a 16 canales  (B, 16, 512, 512)
        merge2 = up2 + out0               # Fusiona con skip0 (elementwise sum)
        merge2 = self.conv_merge2(merge2) # Refinamiento: (B, 16, 512, 512)
        
        # 5. Salida final
        final = self.conv_final(merge2)   # (B, ch_out, 512, 512)
        return final


class SantosNet_GCh_lite_v2(nn.Module):
    """
    Variante mejorada de la red lite para imágenes 512x512.
    
    Esta versión utiliza:
      - ResidualBlock en lugar de bloques simples,
      - Kernel de tamaño 5 en todas las convoluciones.
    
    Arquitectura:
      * Encoder:
          - Stage 0: Dos bloques consecutivos sobre el canal verde (salida skip0).
          - Stage 1: Dos bloques para incrementar de 16 a 32 canales, seguido de pool (skip1).
          - Stage 2: Dos bloques para incrementar a 64 canales, seguidos de pool (skip2).
      * Bottleneck: Dos bloques residuales (con 64 canales) en la resolución 128x128.
      * Decoder:
          - Upsample de 128x128 a 256x256, seguido de bloque que reduce canales a 32 y fusión (suma) con skip1.
          - Upsample de 256x256 a 512x512, seguido de bloque que reduce canales a 16 y fusión (suma) con skip0.
      * Capa final: Convolución 1x1 que mapea a la salida deseada y activación Sigmoid.
    """
    def __init__(self, ch_out, ls_mid_ch=[16, 32, 64, 64, 32, 16], k_size=5):
        super(SantosNet_GCh_lite_v2, self).__init__()
        # Extracción del canal verde
        self.channel_convert = GreenChannelBlock()
        
        # --- Encoder ---
        # Stage 0: Dos bloques consecutivos (de 1 a 16 canales)
        self.conv0a = ResidualBlock(1, ls_mid_ch[0], stride=1, k_size=k_size, dilation=1, layer_num="0a")
        self.conv0b = ResidualBlock(ls_mid_ch[0], ls_mid_ch[0], stride=1, k_size=k_size, dilation=1, layer_num="0b")
        # Stage 1: Dos bloques para pasar de 16 a 32 canales
        self.conv1a = ResidualBlock(ls_mid_ch[0], ls_mid_ch[1], stride=1, k_size=k_size, dilation=1, layer_num="1a")
        self.conv1b = ResidualBlock(ls_mid_ch[1], ls_mid_ch[1], stride=1, k_size=k_size, dilation=1, layer_num="1b")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 512x512  256x256
        
        # Stage 2: Dos bloques para incrementar a 64 canales
        self.conv2a = ResidualBlock(ls_mid_ch[1], ls_mid_ch[2], stride=1, k_size=k_size, dilation=1, layer_num="2a")
        self.conv2b = ResidualBlock(ls_mid_ch[2], ls_mid_ch[2], stride=1, k_size=k_size, dilation=1, layer_num="2b")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256x256  128x128
        
        # --- Bottleneck ---
        self.bottleneck1 = ResidualBlock(ls_mid_ch[2], ls_mid_ch[2], stride=1, k_size=k_size, dilation=1, layer_num="bottle1")
        self.bottleneck2 = ResidualBlock(ls_mid_ch[2], ls_mid_ch[2], stride=1, k_size=k_size, dilation=1, layer_num="bottle2")
        
        # --- Decoder ---
        # Primer upsampling: 128x128  256x256
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decode1 = ResidualBlock(ls_mid_ch[2], ls_mid_ch[1], stride=1, k_size=k_size, dilation=1, layer_num="decode1")
        self.fuse1 = ResidualBlock(ls_mid_ch[1], ls_mid_ch[1], stride=1, k_size=k_size, dilation=1, layer_num="fuse1")
        
        # Segundo upsampling: 256x256  512x512
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decode0 = ResidualBlock(ls_mid_ch[1], ls_mid_ch[0], stride=1, k_size=k_size, dilation=1, layer_num="decode0")
        self.fuse0 = ResidualBlock(ls_mid_ch[0], ls_mid_ch[0], stride=1, k_size=k_size, dilation=1, layer_num="fuse0")
        
        # Capa final: Convolución 1x1 y activación Sigmoid
        self.conv_final = nn.Sequential(
            nn.Conv2d(ls_mid_ch[0], ch_out, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 1. Convertir la imagen de 3 canales a 1 canal (usando solo el canal verde)
        x = self.channel_convert(x)
        
        # --- Encoder ---
        # Stage 0
        x0 = self.conv0a(x)
        x0 = self.conv0b(x0)  # Salida: (B, 16, 512, 512)  skip0
        # Stage 1
        x1 = self.conv1a(x0)
        x1 = self.conv1b(x1)  # (B, 32, 512, 512)
        skip1 = self.pool1(x1)  # (B, 32, 256, 256)  skip1
        # Stage 2
        x2 = self.conv2a(skip1)
        x2 = self.conv2b(x2)  # (B, 64, 256, 256)
        skip2 = self.pool2(x2)  # (B, 64, 128, 128)
        
        # --- Bottleneck ---
        bn = self.bottleneck1(skip2)
        bn = self.bottleneck2(bn)  # (B, 64, 128, 128)
        
        # --- Decoder ---
        # Upsample de 128x128 a 256x256 y fusión con skip1
        up1 = self.up1(bn)
        d1 = self.decode1(up1)  # (B, 32, 256, 256)
        fuse1 = d1 + skip1      # Fusión mediante suma
        fuse1 = self.fuse1(fuse1)
        # Upsample de 256x256 a 512x512 y fusión con skip0
        up2 = self.up2(fuse1)
        d0 = self.decode0(up2)  # (B, 16, 512, 512)
        fuse0 = d0 + x0         # Fusión con salida de Stage 0
        fuse0 = self.fuse0(fuse0)
        
        # Capa final
        out = self.conv_final(fuse0)
        return out