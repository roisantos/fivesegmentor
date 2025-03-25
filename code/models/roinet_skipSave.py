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






class RoiNet_1bottleneck(nn.Module):
    """
    Modelo similar a RoiNet, con la misma configuración de canales:
      ls_mid_ch = [32, 64, 128, 128, 64, 32]
    La única diferencia es que en el bottleneck se utiliza UNICAMENTE
    un bloque (bottle1) en lugar de dos.
    
    La estructura es la siguiente:
      Encoder: conv0 -> conv1 (con pool1) -> conv2 (con pool2)
      Bottleneck: un bloque bottle1, seguido de la concatenación con skip2 y fusión (merge2)
      Decoder: conv3 + up3 + merge3, conv4 + up4 + merge4, conv5 y final.
    """
    def __init__(self, ch_in, ch_out, ls_mid_ch=[32, 64, 128, 128, 64, 32],
                 k_size=9,
                 cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock):
        super(RoiNet_1bottleneck, self).__init__()
        
        self.dict_module = nn.ModuleDict()
        ch = ch_in  # número de canales de entrada
        
        # ------------------ Encoder ------------------
        # Bloque 0: Extrae características a resolución completa.
        self.dict_module.add_module("conv0", cls_init_block(ch, ls_mid_ch[0], k_size=k_size, layer_num=0))
        ch = ls_mid_ch[0]  # 32 canales
        
        # Bloque 1: Conv1 y downsampling
        self.dict_module.add_module("conv1", cls_init_block(ch, ls_mid_ch[1], k_size=k_size, layer_num=1))
        ch = ls_mid_ch[1]  # 64 canales
        # Pooling + Conv 1x1 que duplica canales: 64 -> 128
        self.dict_module.add_module("pool1", nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(ch, ch * 2, kernel_size=1)
        ))
        ch = ch * 2  # ahora ch = 128
        # Guardamos skip1 para fusionar en el decoder
        # (skip1: salida de pool1, con 128 canales)
        
        # Bloque 2: Conv2 y segundo downsampling
        self.dict_module.add_module("conv2", cls_init_block(ch, ls_mid_ch[2], k_size=k_size, layer_num=2))
        ch = ls_mid_ch[2]  # 128 canales
        # Segundo pooling: duplica canales: 128 -> 256
        self.dict_module.add_module("pool2", nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(ch, ch * 2, kernel_size=1)
        ))
        ch = ch * 2  # ch = 256
        # Guardamos skip2 para la fusión en el bottleneck
        
        # ------------------ Bottleneck (Con 1 bloque) ------------------
        # Único bloque en el bottleneck
        self.dict_module.add_module("bottle1", cls_conv_block(ch, ch, k_size=k_size, layer_num="bottle1"))
        # Luego se concatena la salida de bottle1 (256 canales) con skip2 (256 canales)
        # y se aplica merge2: reduce 512 canales a 256.
        self.dict_module.add_module("merge2", nn.Sequential(
            nn.Conv2d(ch * 2, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        ))
        # Después de merge2, la salida tiene 256 canales y resolución 1/4
        
        # ------------------ Decoder ------------------
        # Bloque 3: Upsample desde el bottleneck
        self.dict_module.add_module("conv3", cls_init_block(ch, ls_mid_ch[3], k_size=k_size, layer_num=3))
        ch = ls_mid_ch[3]  # en ls_mid_ch[3] es 128
        self.dict_module.add_module("up3", nn.Sequential(
            nn.ConvTranspose2d(ch, ch // 2, kernel_size=2, stride=2)
        ))
        ch = ch // 2  # ahora ch = 64
        # Merge con skip1: skip1 tiene 128 canales (salida de pool1)
        self.dict_module.add_module("merge3", nn.Sequential(
            nn.Conv2d((ls_mid_ch[3] // 2) + (ls_mid_ch[1] * 2), ls_mid_ch[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ls_mid_ch[1]),
            nn.ReLU(inplace=True)
        ))
        ch = ls_mid_ch[1]  # ch = 64
        
        # Bloque 4: Upsample adicional
        self.dict_module.add_module("conv4", cls_init_block(ch, ls_mid_ch[4], k_size=k_size, layer_num=4))
        ch = ls_mid_ch[4]  # ch = 64
        self.dict_module.add_module("up4", nn.Sequential(
            nn.ConvTranspose2d(ch, ch // 2, kernel_size=2, stride=2)
        ))
        ch = ch // 2  # ch = 32
        self.dict_module.add_module("merge4", nn.Sequential(
            nn.Conv2d((ls_mid_ch[4] // 2) + ls_mid_ch[0], ls_mid_ch[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ls_mid_ch[0]),
            nn.ReLU(inplace=True)
        ))
        ch = ls_mid_ch[0]  # ch = 32
        
        # Bloque 5: Última convolución de procesamiento
        self.dict_module.add_module("conv5", cls_init_block(ch, ls_mid_ch[5], k_size=k_size, layer_num=5))
        ch = ls_mid_ch[5]  # ch = 32
        
        # Clasificación final: reduce a ch_out (por ejemplo, 1 canal para segmentación binaria)
        self.dict_module.add_module("final", nn.Sequential(
            nn.Conv2d(ch, ch_out, kernel_size=1, bias=False),
            nn.Sigmoid()
        ))
        
        self.ls_mid_ch = ls_mid_ch

    def forward(self, x):
        # Encoder
        out0 = self.dict_module["conv0"](x)          # (B, 32, H, W)  -> skip0
        out1 = self.dict_module["conv1"](out0)         # (B, 64, H, W)
        out1 = self.dict_module["pool1"](out1)         # (B, 128, H/2, W/2) -> skip1
        skip1 = out1

        out2 = self.dict_module["conv2"](out1)         # (B, 128, H/2, W/2)
        out2 = self.dict_module["pool2"](out2)         # (B, 256, H/4, W/4) -> skip2
        skip2 = out2

        # Bottleneck: se aplica solo bottle1
        bottle1 = self.dict_module["bottle1"](out2)      # (B, 256, H/4, W/4)
        # Concatenamos bottle1 y skip2: 256 + 256 = 512 canales
        bottle_cat = torch.cat([bottle1, skip2], dim=1)
        bottle_out = self.dict_module["merge2"](bottle_cat)  # (B, 256, H/4, W/4)

        # Decoder
        out3 = self.dict_module["conv3"](bottle_out)     # (B, 128, H/4, W/4)
        out3 = self.dict_module["up3"](out3)             # (B, 64, H/2, W/2)
        out3 = torch.cat([out3, skip1], dim=1)           # (B, 64 + 128 = 192, H/2, W/2)
        out3 = self.dict_module["merge3"](out3)          # (B, 64, H/2, W/2)

        out4 = self.dict_module["conv4"](out3)           # (B, 64, H/2, W/2)
        out4 = self.dict_module["up4"](out4)             # (B, 32, H, W)
        out4 = torch.cat([out4, out0], dim=1)            # (B, 32 + 32 = 64, H, W)
        out4 = self.dict_module["merge4"](out4)          # (B, 32, H, W)

        out5 = self.dict_module["conv5"](out4)           # (B, 32, H, W)
        final = self.dict_module["final"](out5)          # (B, ch_out, H, W)
        return final




class RoiNet_3bottleneck(nn.Module):
    """
    RoiNet_3bottleneck es un modelo basado en RoiNet que conserva la misma
    arquitectura de encoder y decoder (con ls_mid_ch = [32, 64, 128, 128, 64, 32]),
    pero en el bottleneck utiliza TRES bloques (bottle1, bottle2 y bottle3)
    en lugar de dos. Luego, se concatena la salida del último bloque (256 canales)
    con el skip2 (256 canales), formando 512 canales, que se reducen a 256 mediante merge2.
    """
    def __init__(self, ch_in, ch_out, ls_mid_ch=[32, 64, 128, 128, 64, 32],
                 k_size=9,
                 cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock):
        super(RoiNet_3bottleneck, self).__init__()
        self.dict_module = nn.ModuleDict()
        ch = ch_in  # canales de entrada

        # ------------------ Encoder ------------------
        # Bloque 0: conv0 - extrae características a resolución completa.
        self.dict_module.add_module("conv0", cls_init_block(ch, ls_mid_ch[0], k_size=k_size, layer_num=0))
        ch = ls_mid_ch[0]  # 32 canales

        # Bloque 1: conv1 y pooling
        self.dict_module.add_module("conv1", cls_init_block(ch, ls_mid_ch[1], k_size=k_size, layer_num=1))
        ch = ls_mid_ch[1]  # 64 canales
        self.dict_module.add_module("pool1", nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(ch, ch * 2, kernel_size=1)
        ))
        ch = ch * 2  # Ahora ch = 128
        # skip1 se guarda para el decoder

        # Bloque 2: conv2 y segundo pooling
        self.dict_module.add_module("conv2", cls_init_block(ch, ls_mid_ch[2], k_size=k_size, layer_num=2))
        ch = ls_mid_ch[2]  # 128 canales
        self.dict_module.add_module("pool2", nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(ch, ch * 2, kernel_size=1)
        ))
        ch = ch * 2  # Ahora ch = 256
        # skip2 se guarda para la fusión en el bottleneck

        # ------------------ Bottleneck (3 bloques) ------------------
        # Se aplican tres bloques secuenciales sobre 256 canales.
        self.dict_module.add_module("bottle1", cls_conv_block(ch, ch, k_size=k_size, layer_num="bottle1"))
        self.dict_module.add_module("bottle2", cls_conv_block(ch, ch, k_size=k_size, layer_num="bottle2"))
        self.dict_module.add_module("bottle3", cls_conv_block(ch, ch, k_size=k_size, layer_num="bottle3"))
        # Se concatena la salida del tercer bloque con skip2.
        # Ambos tienen 256 canales  256 + 256 = 512.
        self.dict_module.add_module("merge2", nn.Sequential(
            nn.Conv2d(ch * 2, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        ))
        # La salida de merge2 tendrá 256 canales y resolución 1/4.

        # ------------------ Decoder ------------------
        # Bloque 3: conv3 y upsampling
        self.dict_module.add_module("conv3", cls_init_block(ch, ls_mid_ch[3], k_size=k_size, layer_num=3))
        ch = ls_mid_ch[3]  # según ls_mid_ch[3] es 128
        self.dict_module.add_module("up3", nn.Sequential(
            nn.ConvTranspose2d(ch, ch // 2, kernel_size=2, stride=2)
        ))
        ch = ch // 2  # ahora ch = 64
        # Se concatena con skip1 (que tiene 128 canales)
        self.dict_module.add_module("merge3", nn.Sequential(
            nn.Conv2d((ls_mid_ch[3] // 2) + (ls_mid_ch[1] * 2), ls_mid_ch[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ls_mid_ch[1]),
            nn.ReLU(inplace=True)
        ))
        ch = ls_mid_ch[1]  # ch = 64

        # Bloque 4: conv4 y upsampling adicional
        self.dict_module.add_module("conv4", cls_init_block(ch, ls_mid_ch[4], k_size=k_size, layer_num=4))
        ch = ls_mid_ch[4]  # ch = 64
        self.dict_module.add_module("up4", nn.Sequential(
            nn.ConvTranspose2d(ch, ch // 2, kernel_size=2, stride=2)
        ))
        ch = ch // 2  # ch = 32
        self.dict_module.add_module("merge4", nn.Sequential(
            nn.Conv2d((ls_mid_ch[4] // 2) + ls_mid_ch[0], ls_mid_ch[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ls_mid_ch[0]),
            nn.ReLU(inplace=True)
        ))
        ch = ls_mid_ch[0]  # ch = 32

        # Bloque 5: conv5 final de procesamiento
        self.dict_module.add_module("conv5", cls_init_block(ch, ls_mid_ch[5], k_size=k_size, layer_num=5))
        ch = ls_mid_ch[5]  # ch = 32

        # Clasificación final: convolución 1x1 y Sigmoid para producir la salida
        self.dict_module.add_module("final", nn.Sequential(
            nn.Conv2d(ch, ch_out, kernel_size=1, bias=False),
            nn.Sigmoid()
        ))
        
        self.ls_mid_ch = ls_mid_ch

    def forward(self, x):
        # Encoder
        out0 = self.dict_module["conv0"](x)           # (B, 32, H, W)  -> skip0
        out1 = self.dict_module["conv1"](out0)          # (B, 64, H, W)
        out1 = self.dict_module["pool1"](out1)          # (B, 128, H/2, W/2) -> skip1
        skip1 = out1

        out2 = self.dict_module["conv2"](out1)          # (B, 128, H/2, W/2)
        out2 = self.dict_module["pool2"](out2)          # (B, 256, H/4, W/4) -> skip2
        skip2 = out2

        # Bottleneck: Se aplican tres bloques secuenciales.
        bottle1 = self.dict_module["bottle1"](out2)       # (B, 256, H/4, W/4)
        bottle2 = self.dict_module["bottle2"](bottle1)      # (B, 256, H/4, W/4)
        bottle3 = self.dict_module["bottle3"](bottle2)      # (B, 256, H/4, W/4)
        # Concatenación con skip2: 256 + 256 = 512 canales.
        bottle_cat = torch.cat([bottle3, skip2], dim=1)
        bottle_out = self.dict_module["merge2"](bottle_cat)  # (B, 256, H/4, W/4)

        # Decoder
        out3 = self.dict_module["conv3"](bottle_out)       # (B, 128, H/4, W/4)
        out3 = self.dict_module["up3"](out3)               # (B, 64, H/2, W/2)
        out3 = torch.cat([out3, skip1], dim=1)             # (B, 64+128=192, H/2, W/2)
        out3 = self.dict_module["merge3"](out3)            # (B, 64, H/2, W/2)

        out4 = self.dict_module["conv4"](out3)             # (B, 64, H/2, W/2)
        out4 = self.dict_module["up4"](out4)               # (B, 32, H, W)
        out4 = torch.cat([out4, out0], dim=1)              # (B, 32+32=64, H, W)
        out4 = self.dict_module["merge4"](out4)            # (B, 32, H, W)

        out5 = self.dict_module["conv5"](out4)             # (B, 32, H, W)
        final = self.dict_module["final"](out5)            # (B, ch_out, H, W)
        return final



class RoiNetTest1bottleneck(nn.Module):

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
        #self.dict_module.add_module("bottle2", cls_conv_block(ch, ch, k_size=k_size, layer_num="bottle2"))
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
        #bottle2 = self.dict_module["bottle2"](bottle1)      # (B, 256, H/4, W/4)
        # Merge the original skip2 with the deepened features.
        bottle_cat = torch.cat([bottle1, skip2], dim=1)     # (B, 512, H/4, W/4)
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



class RoiNetTest3bottleneck(nn.Module):

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
        self.dict_module.add_module("bottle3", cls_conv_block(ch, ch, k_size=k_size, layer_num="bottle3"))
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
        bottle3 = self.dict_module["bottle3"](bottle2)
        # Merge the original skip2 with the deepened features.
        bottle_cat = torch.cat([bottle3, skip2], dim=1)     # (B, 512, H/4, W/4)
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




class RoiNetTest2bottleneck(nn.Module):

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



class RoiNetNoSkip(RoiNet):
    """
    Variante de RoiNet en la que se eliminan las skip connections.
    Se redefinen los módulos de fusión para que no requieran concatenar
    características de las rutas laterales, de modo que la red procese
    únicamente la señal de upsampling.
    """
    def __init__(self, ch_in, ch_out, ls_mid_ch=[32, 64, 128, 128, 64, 32],
                 k_size=9,
                 cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock):
        super().__init__(ch_in, ch_out, ls_mid_ch=ls_mid_ch,
                         k_size=k_size, cls_init_block=cls_init_block,
                         cls_conv_block=cls_conv_block)
        # Reconfiguramos merge3 y merge4 para procesar solo la señal de upsampling.
        # En merge3, se espera una entrada de dimensión (C = ls_mid_ch[3]//2)
        self.dict_module["merge3"] = nn.Sequential(
            nn.Conv2d(ls_mid_ch[3] // 2, ls_mid_ch[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ls_mid_ch[1]),
            nn.ReLU(inplace=True)
        )
        # En merge4, la entrada será la salida de up4 con canales ls_mid_ch[4]//2.
        self.dict_module["merge4"] = nn.Sequential(
            nn.Conv2d(ls_mid_ch[4] // 2, ls_mid_ch[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ls_mid_ch[0]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        out0 = self.dict_module["conv0"](x)
        out1 = self.dict_module["conv1"](out0)
        out1 = self.dict_module["pool1"](out1)
        out2 = self.dict_module["conv2"](out1)
        out2 = self.dict_module["pool2"](out2)
        
        # Bottleneck: se utilizan dos bloques
        bottle1 = self.dict_module["bottle1"](out2)
        bottle2 = self.dict_module["bottle2"](bottle1)
        bottle_out = bottle2  # No se utiliza skip2
        
        # Decoder (sin skip connections)
        out3 = self.dict_module["conv3"](bottle_out)
        out3 = self.dict_module["up3"](out3)
        out3 = self.dict_module["merge3"](out3)
        out4 = self.dict_module["conv4"](out3)
        out4 = self.dict_module["up4"](out4)
        out4 = self.dict_module["merge4"](out4)
        out5 = self.dict_module["conv5"](out4)
        final = self.dict_module["final"](out5)

        return final


# 2. Variante con fusión por suma en lugar de concatenación: RoiNetSumFusion
class RoiNetSumFusion(RoiNet):
    """
    Variante de RoiNet que fusiona las skip connections mediante suma en vez de concatenación.
    Se incorpora un bloque de proyección para igualar dimensiones antes de la suma,
    y se redefinen los módulos merge3 y merge4 para operar sobre la suma.
    """
    def __init__(self, ch_in, ch_out, ls_mid_ch=[32, 64, 128, 128, 64, 32],
                 k_size=9,
                 cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock):
        super().__init__(ch_in, ch_out, ls_mid_ch=ls_mid_ch,
                         k_size=k_size, cls_init_block=cls_init_block,
                         cls_conv_block=cls_conv_block)
        # En la arquitectura original, skip1 (salida de pool1) tiene 128 canales,
        # mientras que la salida de up3 tiene ls_mid_ch[3]//2 canales (64 si ls_mid_ch[3]==128).
        # Se añade una capa de proyección para transformar skip1 a 64 canales.
        self.proj_skip1 = nn.Conv2d(ls_mid_ch[1] * 2, ls_mid_ch[3] // 2, kernel_size=1, bias=False)
        # Se redefinen merge3 y merge4 para trabajar con la suma.
        self.dict_module["merge3"] = nn.Sequential(
            nn.Conv2d(ls_mid_ch[3] // 2, ls_mid_ch[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ls_mid_ch[1]),
            nn.ReLU(inplace=True)
        )
        self.dict_module["merge4"] = nn.Sequential(
            nn.Conv2d(ls_mid_ch[4] // 2, ls_mid_ch[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ls_mid_ch[0]),
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

        # Bottleneck: se usan dos bloques
        bottle1 = self.dict_module["bottle1"](out2)
        bottle2 = self.dict_module["bottle2"](bottle1)
        bottle_cat = torch.cat([bottle2, skip2], dim=1)
        bottle_out = self.dict_module["merge2"](bottle_cat)

        # Decoder con fusión por suma:
        out3 = self.dict_module["conv3"](bottle_out)
        out3 = self.dict_module["up3"](out3)  # Supongamos que out3 tiene, por ejemplo, 64 canales
        # Proyectamos skip1 para igualar dimensiones:
        skip1_proj = self.proj_skip1(skip1)  # skip1 originalmente tiene 128 canales
        out3 = self.dict_module["merge3"](out3 + skip1_proj)
        
        out4 = self.dict_module["conv4"](out3)
        out4 = self.dict_module["up4"](out4)  # out4: 32 canales
        # Sumamos directamente con out0 (ya que éste posee 32 canales)
        out4 = self.dict_module["merge4"](out4 + out0)
        
        out5 = self.dict_module["conv5"](out4)
        final = self.dict_module["final"](out5)
        return final


# 3. Variante con atención en las skip connections: RoiNetAttnSkip
class RoiNetAttnSkip(RoiNet):
    """
    Variante de RoiNet que incorpora un mecanismo de atención en las skip connections.
    Se utiliza un módulo AttentionGate (definido en common.py) para ponderar la skip connection
    antes de fusionarla con la señal del decoder.
    """
    def __init__(self, ch_in, ch_out, ls_mid_ch=[32, 64, 128, 128, 64, 32],
                 k_size=9,
                 cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock):
        super().__init__(ch_in, ch_out, ls_mid_ch=ls_mid_ch,
                         k_size=k_size, cls_init_block=cls_init_block,
                         cls_conv_block=cls_conv_block)
        # Se definen módulos de atención para cada skip connection.
        # Para skip1: skip1 tiene 128 canales y la señal de gating (de up3) tiene 64 canales.
        self.attn_skip1 = AttentionGate(F_g=ls_mid_ch[3] // 2, F_l=ls_mid_ch[1] * 2, F_int=ls_mid_ch[3] // 2)
        # Para skip0: skip0 tiene 32 canales y la señal de gating (de up4) tiene 32 canales.
        self.attn_skip0 = AttentionGate(F_g=ls_mid_ch[4] // 2, F_l=ls_mid_ch[0], F_int=ls_mid_ch[4] // 2)

        # Se mantienen los módulos merge3 y merge4 originales.
            
    def forward(self, x):
        # Encoder
        out0 = self.dict_module["conv0"](x)
        out1 = self.dict_module["conv1"](out0)
        out1 = self.dict_module["pool1"](out1)
        skip1 = out1
        out2 = self.dict_module["conv2"](out1)
        out2 = self.dict_module["pool2"](out2)
        skip2 = out2

        # Bottleneck: se aplican dos bloques
        bottle1 = self.dict_module["bottle1"](out2)
        bottle2 = self.dict_module["bottle2"](bottle1)
        bottle_cat = torch.cat([bottle2, skip2], dim=1)
        bottle_out = self.dict_module["merge2"](bottle_cat)

        # Decoder con atención en las skip connections:
        out3 = self.dict_module["conv3"](bottle_out)
        out3 = self.dict_module["up3"](out3)  # out3: 64 canales
        # Aplicamos atención a skip1 usando out3 como señal de gating:
        attn_skip1 = self.attn_skip1(skip1, out3)
        out3 = torch.cat([out3, attn_skip1], dim=1)
        out3 = self.dict_module["merge3"](out3)
        
        out4 = self.dict_module["conv4"](out3)
        out4 = self.dict_module["up4"](out4)  # out4: 32 canales
        # Aplicamos atención a out0 usando out4 como gating:
        attn_skip0 = self.attn_skip0(out0, out4)
        out4 = torch.cat([out4, attn_skip0], dim=1)
        out4 = self.dict_module["merge4"](out4)
        
        out5 = self.dict_module["conv5"](out4)
        final = self.dict_module["final"](out5)
        return final

class RoiNetResSkip(RoiNet):
    """
    Variante de RoiNet en la que las skip connections se fusionan de forma residual.
    Se utiliza una proyección (conv1x1) para adaptar las dimensiones y se refina la suma.
    """
    def __init__(self, ch_in, ch_out, ls_mid_ch=[32, 64, 128, 128, 64, 32],
                 k_size=9,
                 cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock):
        super().__init__(ch_in, ch_out, ls_mid_ch=ls_mid_ch,
                         k_size=k_size, cls_init_block=cls_init_block,
                         cls_conv_block=cls_conv_block)
        # Proyección para la skip connection de pool1 (skip1)
        self.res_skip1 = nn.Conv2d(ls_mid_ch[1] * 2, ls_mid_ch[3] // 2, kernel_size=1, bias=False)
        # Para la skip connection de conv0, se usa una identidad
        self.res_skip0 = nn.Identity()
        # Bloques de refinamiento tras la suma en cada fusión
        self.refine3 = nn.Sequential(
            nn.Conv2d(ls_mid_ch[3] // 2, ls_mid_ch[3] // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ls_mid_ch[3] // 2),
            nn.ReLU(inplace=True)
        )
        self.refine4 = nn.Sequential(
            nn.Conv2d(ls_mid_ch[4] // 2, ls_mid_ch[4] // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ls_mid_ch[4] // 2),
            nn.ReLU(inplace=True)
        )
        # Reemplazamos merge3 y merge4 para que se adapten a la fusión residual:
        # Originalmente, merge3 esperaba (ls_mid_ch[3]//2 + ls_mid_ch[1]*2)=192 canales.
        # En fusión residual, sumamos dos tensores con ls_mid_ch[3]//2 canales cada uno.
        self.dict_module["merge3"] = nn.Sequential(
            nn.Conv2d(ls_mid_ch[3] // 2, ls_mid_ch[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ls_mid_ch[1]),
            nn.ReLU(inplace=True)
        )
        # Para merge4, originalmente se concatenaban out4 (ls_mid_ch[4]//2) y out0 (ls_mid_ch[0]),
        # es decir, 32+32=64 canales; en la fusión residual, la suma da ls_mid_ch[4]//2 (32) canales.
        self.dict_module["merge4"] = nn.Sequential(
            nn.Conv2d(ls_mid_ch[4] // 2, ls_mid_ch[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ls_mid_ch[0]),
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

        # Bottleneck: se usan dos bloques
        bottle1 = self.dict_module["bottle1"](out2)
        bottle2 = self.dict_module["bottle2"](bottle1)
        bottle_cat = torch.cat([bottle2, skip2], dim=1)
        bottle_out = self.dict_module["merge2"](bottle_cat)

        # Decoder con fusión residual:
        out3 = self.dict_module["conv3"](bottle_out)
        out3 = self.dict_module["up3"](out3)  # Resultado con canales = ls_mid_ch[3]//2 (64)
        skip1_proj = self.res_skip1(skip1)      # Proyecta skip1 a 64 canales
        fusion3 = out3 + skip1_proj             # Suma residual: tensor con 64 canales
        fusion3 = self.refine3(fusion3)
        out3 = self.dict_module["merge3"](fusion3)  # Ahora merge3 espera 64 canales y produce ls_mid_ch[1] (64)

        out4 = self.dict_module["conv4"](out3)
        out4 = self.dict_module["up4"](out4)      # Resultado con canales = ls_mid_ch[4]//2 (32)
        skip0 = self.res_skip0(out0)              # out0 tiene 32 canales
        fusion4 = out4 + skip0                    # Suma residual: 32 canales
        fusion4 = self.refine4(fusion4)
        out4 = self.dict_module["merge4"](fusion4)  # merge4 espera 32 canales y produce ls_mid_ch[0] (32)

        out5 = self.dict_module["conv5"](out4)
        final = self.dict_module["final"](out5)
        return final


class RoiNetConcatPlus(RoiNet):
    """
    Variante de RoiNet que utiliza la concatenación de skip connections (como en el modelo original)
    y, a continuación, aplica un bloque extra de refinamiento en las fusiones (merge3 y merge4).
    """
    def __init__(self, ch_in, ch_out, ls_mid_ch=[32, 64, 128, 128, 64, 32],
                 k_size=9,
                 cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock):
        super().__init__(ch_in, ch_out, ls_mid_ch=ls_mid_ch,
                         k_size=k_size, cls_init_block=cls_init_block,
                         cls_conv_block=cls_conv_block)
        # Bloques de refinamiento extra después de merge3 y merge4
        self.refine_merge3 = nn.Sequential(
            nn.Conv2d(ls_mid_ch[1], ls_mid_ch[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ls_mid_ch[1]),
            nn.ReLU(inplace=True)
        )
        self.refine_merge4 = nn.Sequential(
            nn.Conv2d(ls_mid_ch[0], ls_mid_ch[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ls_mid_ch[0]),
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

        # Bottleneck (dos bloques)
        bottle1 = self.dict_module["bottle1"](out2)
        bottle2 = self.dict_module["bottle2"](bottle1)
        bottle_cat = torch.cat([bottle2, skip2], dim=1)
        bottle_out = self.dict_module["merge2"](bottle_cat)

        # Decoder con concatenación y refinamiento extra
        out3 = self.dict_module["conv3"](bottle_out)
        out3 = self.dict_module["up3"](out3)
        out3 = torch.cat([out3, skip1], dim=1)
        out3 = self.dict_module["merge3"](out3)
        out3 = self.refine_merge3(out3)

        out4 = self.dict_module["conv4"](out3)
        out4 = self.dict_module["up4"](out4)
        out4 = torch.cat([out4, out0], dim=1)
        out4 = self.dict_module["merge4"](out4)
        out4 = self.refine_merge4(out4)

        out5 = self.dict_module["conv5"](out4)
        final = self.dict_module["final"](out5)
        return final


class RoiNetMultiSkip(RoiNet):
    """
    Variante de RoiNet que utiliza múltiples skip connections en la fusión del decoder.
    En merge3 se concatenan la salida del upsampling, skip1 y una versión proyectada de out0.
    """
    def __init__(self, ch_in, ch_out, ls_mid_ch=[32, 64, 128, 128, 64, 32],
                 k_size=9,
                 cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock):
        super().__init__(ch_in, ch_out, ls_mid_ch=ls_mid_ch,
                         k_size=k_size, cls_init_block=cls_init_block,
                         cls_conv_block=cls_conv_block)
        # Redefinimos merge3 para que espere la concatenación de tres entradas:
        # up3 (canales: ls_mid_ch[3]//2), skip1 (canales: ls_mid_ch[1]*2) y out0 proyectado (canales: ls_mid_ch[0])
        in_channels_merge3 = (ls_mid_ch[3] // 2) + (ls_mid_ch[1] * 2) + ls_mid_ch[0]
        self.dict_module["merge3"] = nn.Sequential(
            nn.Conv2d(in_channels_merge3, ls_mid_ch[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ls_mid_ch[1]),
            nn.ReLU(inplace=True)
        )
        # Cambiamos la proyección para out0: aplicamos downsampling para obtener H/2 × W/2
        self.proj_out0 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        out0 = self.dict_module["conv0"](x)              # (B, 32, H, W)
        out1 = self.dict_module["conv1"](out0)             # (B, 64, H, W)
        out1 = self.dict_module["pool1"](out1)             # (B, 128, H/2, W/2)
        skip1 = out1
        out2 = self.dict_module["conv2"](out1)             # (B, 128, H/2, W/2)
        out2 = self.dict_module["pool2"](out2)             # (B, 256, H/4, W/4)
        skip2 = out2

        # Bottleneck: dos bloques
        bottle1 = self.dict_module["bottle1"](out2)
        bottle2 = self.dict_module["bottle2"](bottle1)
        bottle_cat = torch.cat([bottle2, skip2], dim=1)    # (B, 256+256=512, H/4, W/4)
        bottle_out = self.dict_module["merge2"](bottle_cat)  # (B, 256, H/4, W/4)

        # Decoder: upsampling y fusión multi-skip
        out3 = self.dict_module["conv3"](bottle_out)
        out3 = self.dict_module["up3"](out3)               # (B, ls_mid_ch[3]//2, H/2, W/2) e.g. (B, 64, H/2, W/2)
        proj_out0 = self.proj_out0(out0)                   # (B, 32, H/2, W/2)
        fusion3 = torch.cat([out3, skip1, proj_out0], dim=1)  # (B, 64+128+32 = 224, H/2, W/2)
        out3 = self.dict_module["merge3"](fusion3)

        out4 = self.dict_module["conv4"](out3)
        out4 = self.dict_module["up4"](out4)
        out4 = torch.cat([out4, out0], dim=1)
        out4 = self.dict_module["merge4"](out4)

        out5 = self.dict_module["conv5"](out4)
        final = self.dict_module["final"](out5)
        return final
