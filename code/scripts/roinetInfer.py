#!/usr/bin/env python
"""
Script de inferencia para RoiNet9 en la codebase de fivesegmentor.

Se asume:
  - El dataset FIVES tiene la siguiente estructura:
      <data_path>/test/image/
  - Los nombres de las imágenes tienen el formato: [número]_[A,D,N,G].png
    por ejemplo: 1_A.png
  - Se generarán outputs con el nombre:
      [número]_[A,D,N,G]_output_image.png
  - Los outputs se guardarán en:
      /mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/otrosfives/inferences/RoiNetInference/
  
Uso:
  python infer_roinet9_fives.py -model /ruta/al/modelo.pth --data_path /ruta/al/dataset
"""

import os
import cv2
import numpy as np
import torch
import argparse
import sys

# Asegurarse de que la raíz de la codebase esté en el path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

# Importar el modelo RoiNet desde la codebase (configuración RoiNet9)
from models.roinet import RoiNet


def process_image(image_path, device, model):
    """
    Carga y preprocesa la imagen, realiza la inferencia y retorna la imagen de output.
    """
    # Cargar imagen en color (BGR)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Error al cargar la imagen: {image_path}")
    # Normalizar a [0,1]
    image = image.astype(np.float32) / 255.0

    # Aplicar padding para que las dimensiones sean múltiplos de 32
    h, w, _ = image.shape
    pad_x = ((w // 32 + 1) * 32 - w) if (w % 32) != 0 else 0
    pad_y = ((h // 32 + 1) * 32 - h) if (h % 32) != 0 else 0
    image_padded = cv2.copyMakeBorder(
        image,
        pad_y // 2,
        pad_y - pad_y // 2,
        pad_x // 2,
        pad_x - pad_x // 2,
        cv2.BORDER_CONSTANT,
        value=0
    )

    # Reordenar dimensiones: (H, W, C) -> (C, H, W)
    image_transposed = np.transpose(image_padded, (2, 0, 1))
    # Convertir a tensor y agregar dimensión de batch: (1, C, H, W)
    image_tensor = torch.from_numpy(image_transposed).unsqueeze(0).to(device)

    # Inferencia sin cálculo de gradiente
    with torch.no_grad():
        output = model(image_tensor)
    # Se espera que la salida tenga forma (1, 1, H, W); quitamos dimensiones extra
    output_np = output.squeeze().cpu().numpy()  # (H, W)
    # Convertir a imagen uint8 en escala de grises (0-255)
    output_img = (output_np * 255).astype(np.uint8)
    return output_img


def main():
    parser = argparse.ArgumentParser(
        description="Inferencia de RoiNet9 en dataset FIVES (codebase fivesegmentor)"
    )
    parser.add_argument(
        '-model', type=str, required=True,
        help="Ruta al archivo .pth del modelo entrenado"
    )
    parser.add_argument(
        '--data_path', type=str, required=True,
        help="Ruta raíz del dataset FIVES (se espera que contenga test/image)"
    )
    parser.add_argument(
        '--output_dir', type=str,
        default="/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/otrosfives/inferences/RoiNetInference/",
        help="Directorio donde se guardarán las imágenes de output"
    )
    args = parser.parse_args()

    # Configurar dispositivo: usar GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Instanciar el modelo con la configuración RoiNet9 (k_size=9; ls_mid_ch por defecto)
    model = RoiNet(ch_in=3, ch_out=1, k_size=9)
    # Cargar el checkpoint
    checkpoint = torch.load(args.model, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Directorio de imágenes de test: <data_path>/test/image/
    input_dir = os.path.join(args.data_path, "test", "image")
    if not os.path.exists(input_dir):
        raise ValueError(f"El directorio de imágenes de test no existe: {input_dir}")
    # Crear directorio de output si no existe
    os.makedirs(args.output_dir, exist_ok=True)

    # Recorrer imágenes y procesarlas
    for filename in sorted(os.listdir(input_dir)):
        if filename.lower().endswith('.png'):
            image_path = os.path.join(input_dir, filename)
            try:
                output_img = process_image(image_path, device, model)
            except Exception as e:
                print(f"Error procesando {filename}: {e}")
                continue
            # Construir nombre del archivo de salida:
            # Ejemplo: "1_A.png" -> "1_A_output_image.png"
            base_name, ext = os.path.splitext(filename)
            output_filename = f"{base_name}_output_image.png"
            output_path = os.path.join(args.output_dir, output_filename)
            # Guardar la imagen de salida
            cv2.imwrite(output_path, output_img)
            print(f"Guardado: {output_path}")


if __name__ == "__main__":
    main()
