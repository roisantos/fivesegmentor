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


