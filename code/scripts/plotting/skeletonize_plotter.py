#!/usr/bin/env python3
import argparse
import os
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import sys

# Agregar el directorio padre ("code") para poder importar los módulos de models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from training.soft_skeleton import SoftSkeletonize

def main(image_id, custom_input=None):
    # Directorios base
    input_dir = "/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fives-save/FIVESoriginal/test/image"
    label_dir = "/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fives-save/FIVESoriginal/test/label"
    inference_dir = "/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fork-roi/fivesegmentor/inference_results/RoiNetTest2bottleneck_residual"
    # Directorio para imágenes custom
    custom_dir = "/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fork-roi/fivesegmentor/inference_results/custom_images"
    
    # Construir nombres de archivos
    filename = f"{image_id}.png"
    input_path = os.path.join(input_dir, filename)
    label_path = os.path.join(label_dir, filename)
    
    if custom_input is not None:
        output_path = os.path.join(custom_dir, custom_input)
    else:
        output_filename = f"{image_id}_output_image.png"
        output_path = os.path.join(inference_dir, output_filename)
    
    # Comprobar si los archivos existen
    for path, desc in [(input_path, "input"), (label_path, "label"), (output_path, "inference")]:
        if not os.path.exists(path):
            print(f"Error: {desc} image not found at {path}")
            return
    
    # Cargar imágenes
    input_img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    label_img = cv2.imread(label_path, cv2.IMREAD_COLOR)
    output_img = cv2.imread(output_path, cv2.IMREAD_COLOR)
    
    if input_img is None:
        print(f"Error loading input image: {input_path}")
        return
    if label_img is None:
        print(f"Error loading label image: {label_path}")
        return
    if output_img is None:
        print(f"Error loading inference image: {output_path}")
        return
    
    # Convertir de BGR a RGB
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    
    # Inicializar soft skeletonizer
    skeletonizer = SoftSkeletonize(num_iter=20)
    
    # Convertir label y output a escala de grises
    label_gray = cv2.cvtColor(label_img, cv2.COLOR_RGB2GRAY)
    output_gray = cv2.cvtColor(output_img, cv2.COLOR_RGB2GRAY)
    
    # Normalizar y convertir a tensor de forma (1, 1, H, W)
    label_tensor = torch.from_numpy(label_gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    output_tensor = torch.from_numpy(output_gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    
    # Calcular la esqueletonización
    skeleton_label_tensor = skeletonizer(label_tensor)
    skeleton_output_tensor = skeletonizer(output_tensor)
    
    # Convertir las esqueletonizaciones de tensores a arrays de numpy escalados a [0, 255]
    skeleton_label = (skeleton_label_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    skeleton_output = (skeleton_output_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    
    # Crear la figura con 2 filas
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Fila 0: Imágenes completas (Input, Label, Inference)
    axes[0, 0].imshow(input_img)
    axes[0, 0].set_title("Full Input Image")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(label_img)
    axes[0, 1].set_title("Full Label Image")
    axes[0, 1].axis("off")
    
    axes[0, 2].imshow(output_img)
    axes[0, 2].set_title("Full Inference Image")
    axes[0, 2].axis("off")
    
    # Fila 1: Esqueletonización (Label, Inference)
    axes[1, 0].axis("off")
    axes[1, 0].text(0.5, 0.5, "No skeleton computed for input", 
                    horizontalalignment='center', verticalalignment='center', fontsize=12)
    
    axes[1, 1].imshow(skeleton_label, cmap="gray")
    axes[1, 1].set_title("Skeletonized Label")
    axes[1, 1].axis("off")
    
    axes[1, 2].imshow(skeleton_output, cmap="gray")
    axes[1, 2].set_title("Skeletonized Inference")
    axes[1, 2].axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to compare Input, Label, Inference images and their skeletonized outputs."
    )
    parser.add_argument("image_id", type=str, help="Image identifier (e.g., 105_G)")
    parser.add_argument("--customInput", type=str, default=None,
                        help="Nombre de la imagen custom (e.g., sliced_label_105_G.png) ubicada en inference_results/custom_images/")
    
    args = parser.parse_args()
    
    main(args.image_id, args.customInput)
