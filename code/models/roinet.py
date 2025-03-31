import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, ROOT_DIR)
from models.common import *

class RoiNet(nn.Module):

    def __init__(self, ch_in, ch_out, ls_mid_ch=[32, 64, 128, 128, 64, 32],
                 k_size=9,
                 cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock):
        super().__init__()
        self.dict_module = nn.ModuleDict()
        ch = ch_in  # current channel count

        # ------------------ Encoder ------------------
        # Block 0: Full resolution features.
        self.dict_module.add_module("conv0", cls_init_block(ch, ls_mid_ch[0], k_size=k_size, layer_num=0))
        ch = ls_mid_ch[0]  # 32

        # Block 1: Downsample once.
        self.dict_module.add_module("conv1", cls_init_block(ch, ls_mid_ch[1], k_size=k_size, layer_num=1))
        ch = ls_mid_ch[1]  # 64
        # Downsample & double channels.
        self.dict_module.add_module("pool1", nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(ch, ch * 2, kernel_size=1)
        ))
        ch = ch * 2  # becomes 128
        # We'll save the output after pool1 as skip connection "skip1"

        # Block 2: Further encoding.
        self.dict_module.add_module("conv2", cls_init_block(ch, ls_mid_ch[2], k_size=k_size, layer_num=2))
        ch = ls_mid_ch[2]  # 128
        # Downsample & double channels.
        self.dict_module.add_module("pool2", nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(ch, ch * 2, kernel_size=1)
        ))
        ch = ch * 2  # becomes 256
        # Save output after pool2 as skip connection "skip2"

        # ------------------ Bottleneck (Deepened) ------------------
        # Add extra blocks to deepen the bottleneck.
        self.dict_module.add_module("bottle1", cls_conv_block(ch, ch, k_size=k_size, layer_num="bottle1"))
        self.dict_module.add_module("bottle2", cls_conv_block(ch, ch, k_size=k_size, layer_num="bottle2"))
        # Merge skip2 (from encoder) with the deepened bottleneck output.
        self.dict_module.add_module("merge2", nn.Sequential(
            nn.Conv2d(ch * 2, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        ))
        # After merging, our tensor will have 256 channels at 1/4 resolution.

        # ------------------ Decoder ------------------
        # Block 3: Upsample from bottleneck.
        self.dict_module.add_module("conv3", cls_init_block(ch, ls_mid_ch[3], k_size=k_size, layer_num=3))
        ch = ls_mid_ch[3]  # 128 originally, 192 in the scaled version
        self.dict_module.add_module("up3", nn.Sequential(
            nn.ConvTranspose2d(ch, ch // 2, kernel_size=2, stride=2)
        ))
        ch = ch // 2  # becomes 64 originally, 96 in the scaled version
        # Merge with skip connection from Block 1 (pool1 output has ls_mid_ch[1]*2 channels)
        self.dict_module.add_module("merge3", nn.Sequential(
            nn.Conv2d((ls_mid_ch[3] // 2) + (ls_mid_ch[1] * 2), ls_mid_ch[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ls_mid_ch[1]),
            nn.ReLU(inplace=True)
        ))
        ch = ls_mid_ch[1]  # now set to ls_mid_ch[1]

        # Block 4: Further upsampling
        self.dict_module.add_module("conv4", cls_init_block(ch, ls_mid_ch[4], k_size=k_size, layer_num=4))
        ch = ls_mid_ch[4]
        self.dict_module.add_module("up4", nn.Sequential(
            nn.ConvTranspose2d(ch, ch // 2, kernel_size=2, stride=2)
        ))
        ch = ch // 2
        self.dict_module.add_module("merge4", nn.Sequential(
            nn.Conv2d((ls_mid_ch[4] // 2) + ls_mid_ch[0], ls_mid_ch[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ls_mid_ch[0]),
            nn.ReLU(inplace=True)
        ))
        ch = ls_mid_ch[0]

        # ---- ADD THIS BLOCK 5 ----
        self.dict_module.add_module("conv5", cls_init_block(ch, ls_mid_ch[5], k_size=k_size, layer_num=5))
        ch = ls_mid_ch[5]

        # ---- Final Classification ----
        self.dict_module.add_module("final", nn.Sequential(
            nn.Conv2d(ch, ch_out, kernel_size=1, bias=False),
            nn.Sigmoid()
        ))

        self.ls_mid_ch = ls_mid_ch

    def forward(self, x):
        # Encoder
        out0 = self.dict_module["conv0"](x)           # (B, 32, H, W)   -> skip0
        out1 = self.dict_module["conv1"](out0)          # (B, 64, H, W)
        out1 = self.dict_module["pool1"](out1)          # (B, 128, H/2, W/2) -> skip1
        skip1 = out1

        out2 = self.dict_module["conv2"](out1)          # (B, 128, H/2, W/2)
        out2 = self.dict_module["pool2"](out2)          # (B, 256, H/4, W/4) -> skip2
        skip2 = out2

        # Bottleneck (deepened)
        bottle1 = self.dict_module["bottle1"](out2)       # (B, 256, H/4, W/4)
        bottle2 = self.dict_module["bottle2"](bottle1)      # (B, 256, H/4, W/4)
        # Merge the original skip2 with the deepened features.
        bottle_cat = torch.cat([bottle2, skip2], dim=1)     # (B, 512, H/4, W/4)
        bottle_out = self.dict_module["merge2"](bottle_cat) # (B, 256, H/4, W/4)

        # Decoder
        out3 = self.dict_module["conv3"](bottle_out)       # (B, 128, H/4, W/4)
        out3 = self.dict_module["up3"](out3)               # (B, 64, H/2, W/2)
        # Merge with skip1 (from pool1)
        out3 = torch.cat([out3, skip1], dim=1)             # (B, 64+128=192, H/2, W/2)
        out3 = self.dict_module["merge3"](out3)            # (B, 64, H/2, W/2)

        out4 = self.dict_module["conv4"](out3)             # (B, 64, H/2, W/2)
        out4 = self.dict_module["up4"](out4)               # (B, 32, H, W)
        # Merge with skip0 (from conv0)
        out4 = torch.cat([out4, out0], dim=1)              # (B, 32+32=64, H, W)
        out4 = self.dict_module["merge4"](out4)            # (B, 32, H, W)

        out5 = self.dict_module["conv5"](out4)             # (B, 32, H, W)
        final = self.dict_module["final"](out5)            # (B, ch_out, H, W)
        return final






class RoiNetAblationOneBlock(RoiNet): #Decreases pre-bottleneck channels
    """
    Variante de RoiNet con un solo bloque en el bottleneck.
    Se utiliza la lista de canales [32, 64, 128, 64, 32].
    """
    def __init__(self, ch_in, ch_out, k_size=9, 
                 cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock):
        # Sobrescribimos la configuración de canales
        ls_mid_ch = [32, 64, 128, 64, 32]
        super().__init__(ch_in, ch_out, ls_mid_ch=ls_mid_ch, k_size=k_size,
                         cls_init_block=cls_init_block, cls_conv_block=cls_conv_block)
        # En el bottleneck original se tenían 2 bloques: "bottle1" y "bottle2".
        # Eliminamos "bottle2" y modificamos la parte de merge.
        if "bottle2" in self.dict_module:
            del self.dict_module["bottle2"]
        # Como ya no concatenamos con skip2, podemos reemplazar merge2 por una función identidad
        self.dict_module["merge2"] = nn.Identity()

    def forward(self, x):
        # Encoder
        out0 = self.dict_module["conv0"](x)
        out1 = self.dict_module["conv1"](out0)
        out1 = self.dict_module["pool1"](out1)
        skip1 = out1
        out2 = self.dict_module["conv2"](out1)
        out2 = self.dict_module["pool2"](out2)
        # Bottleneck modificado: solo un bloque
        bottle1 = self.dict_module["bottle1"](out2)
        bottle_out = self.dict_module["merge2"](bottle1)  # Identidad

        # Decoder
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
        final = torch.max(final, dim=1, keepdim=True)[0]
        return final


class RoiNetAblationThreeBlock(RoiNet): #Decreases pre-bottleneck channels
    """
    Variante de RoiNet con tres bloques en el bottleneck.
    Se utiliza la lista de canales [32, 64, 128, 128, 128, 64, 32].
    """
    def __init__(self, ch_in, ch_out, k_size=9, 
                 cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock):
        ls_mid_ch = [32, 64, 128, 128, 128, 64, 32]
        super().__init__(ch_in, ch_out, ls_mid_ch=ls_mid_ch, k_size=k_size,
                         cls_init_block=cls_init_block, cls_conv_block=cls_conv_block)
        # Reemplazamos el bottleneck original por tres bloques.
        # Asumiendo que originalmente se definían bottle1 y bottle2,
        # vamos a redefinirlos (o agregar uno nuevo) para tener tres.
        self.dict_module["bottle1"] = cls_conv_block(ls_mid_ch[2], ls_mid_ch[2], k_size=k_size, layer_num="bottle1")
        self.dict_module["bottle2"] = cls_conv_block(ls_mid_ch[2], ls_mid_ch[2], k_size=k_size, layer_num="bottle2")
        self.dict_module["bottle3"] = cls_conv_block(ls_mid_ch[2], ls_mid_ch[2], k_size=k_size, layer_num="bottle3")
        # Actualizamos merge2 para concatenar bottle3 y skip2.
        # En este ejemplo, asumimos que skip2 proviene de pool2 y tiene 2*ls_mid_ch[2] canales.
        self.dict_module["merge2"] = nn.Sequential(
            nn.Conv2d(ls_mid_ch[2] + 2 * ls_mid_ch[2], ls_mid_ch[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ls_mid_ch[2]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        out0 = self.dict_module["conv0"](x)
        out1 = self.dict_module["conv1"](out0)
        out1 = self.dict_module["pool1"](out1)
        skip1 = out1
        out2 = self.dict_module["conv2"](out1)
        out2 = self.dict_module["pool2"](out2)
        skip2 = out2
        # Bottleneck de 3 bloques
        bottle1 = self.dict_module["bottle1"](out2)
        bottle2 = self.dict_module["bottle2"](bottle1)
        bottle3 = self.dict_module["bottle3"](bottle2)
        bottle_cat = torch.cat([bottle3, skip2], dim=1)
        bottle_out = self.dict_module["merge2"](bottle_cat)
        # Decoder (igual que en RoiNet original)
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
        final = torch.max(final, dim=1, keepdim=True)[0]
        return final



class RoiNetAblationOneBlock_similar(RoiNet): #Keeps the channels as in RoiNet
    """
    Variante de RoiNet con un único bloque en el bottleneck.
    La configuración de canales es la misma que la original:
      [32, 64, 128, 128, 64, 32]
    La única diferencia es que se elimina "bottle2", concatenando
    directamente la salida de "bottle1" con el skip2 para la fusión.
    """
    def __init__(self, ch_in, ch_out, k_size=9, 
                 cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock):
        # Usamos la misma lista de canales que la original.
        ls_mid_ch = [32, 64, 128, 128, 64, 32]
        super().__init__(ch_in, ch_out, ls_mid_ch=ls_mid_ch, k_size=k_size,
                         cls_init_block=cls_init_block, cls_conv_block=cls_conv_block)
        # Eliminamos el segundo bloque del bottleneck, ya que sólo usaremos uno.
        if "bottle2" in self.dict_module:
            del self.dict_module["bottle2"]
        # Conservamos merge2 tal como está, ya que sigue esperando 512 canales
        # (256 de bottle1 y 256 del skip2).

    def forward(self, x):
        # Encoder
        out0 = self.dict_module["conv0"](x)
        out1 = self.dict_module["conv1"](out0)
        out1 = self.dict_module["pool1"](out1)
        skip1 = out1

        out2 = self.dict_module["conv2"](out1)
        out2 = self.dict_module["pool2"](out2)
        skip2 = out2

        # Bottleneck modificado: solo se aplica "bottle1"
        bottle1 = self.dict_module["bottle1"](out2)
        # Se concatena bottle1 (256 canales) con skip2 (256 canales)  512 canales
        bottle_cat = torch.cat([bottle1, skip2], dim=1)
        bottle_out = self.dict_module["merge2"](bottle_cat)

        # Decoder (igual que en RoiNet original)
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
        # Mantener la reducción de la dimensión de canales final, como en RoiNetAblationOneBlock
        final = torch.max(final, dim=1, keepdim=True)[0]
        return final




class RoiNetAblationThreeBlock_similar(RoiNet):
    """
    Variante de RoiNet con tres bloques en el bottleneck.
    La configuración de canales se mantiene en:
      [32, 64, 128, 128, 64, 32]
    La única modificación es que, en lugar de dos bloques (bottle1 y bottle2),
    se aplican tres bloques secuenciales (bottle1, bottle2 y bottle3) en el bottleneck.
    Posteriormente, se concatena la salida de bottle3 con skip2 para usar merge2.
    """
    def __init__(self, ch_in, ch_out, k_size=9, 
                 cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock):
        ls_mid_ch = [32, 64, 128, 128, 64, 32]
        super().__init__(ch_in, ch_out, ls_mid_ch=ls_mid_ch, k_size=k_size,
                         cls_init_block=cls_init_block, cls_conv_block=cls_conv_block)
        # Reemplazamos los bloques del bottleneck para tener tres bloques.
        self.dict_module["bottle1"] = cls_conv_block(256, 256, k_size=k_size, layer_num="bottle1")
        self.dict_module["bottle2"] = cls_conv_block(256, 256, k_size=k_size, layer_num="bottle2")
        # Se agrega el tercer bloque:
        self.dict_module.add_module("bottle3", cls_conv_block(256, 256, k_size=k_size, layer_num="bottle3"))
        # Conservamos merge2 sin modificaciones (espera 512 canales).

    def forward(self, x):
        # Encoder
        out0 = self.dict_module["conv0"](x)
        out1 = self.dict_module["conv1"](out0)
        out1 = self.dict_module["pool1"](out1)
        skip1 = out1

        out2 = self.dict_module["conv2"](out1)
        out2 = self.dict_module["pool2"](out2)
        skip2 = out2

        # Bottleneck de 3 bloques:
        bottle1 = self.dict_module["bottle1"](out2)
        bottle2 = self.dict_module["bottle2"](bottle1)
        bottle3 = self.dict_module["bottle3"](bottle2)
        # Concatenamos la salida del último bloque (256 canales) con skip2 (256 canales)
        bottle_cat = torch.cat([bottle3, skip2], dim=1)
        bottle_out = self.dict_module["merge2"](bottle_cat)

        # Decoder (igual que en RoiNet original)
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
        final = torch.max(final, dim=1, keepdim=True)[0]
        return final
