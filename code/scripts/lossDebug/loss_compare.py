#!/usr/bin/env python3
import argparse
import os
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import sys
import time

# Agregar el directorio padre ("code") para poder importar los módulos de models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importar las funciones de pérdida
from training.loss import DiceLoss, SoftCLDiceLoss, SoftDiceCLDiceLoss, ConexLoss

def create_difference_image(ground_truth, prediction):
    gt = ground_truth.astype(np.float32)
    pred = prediction.astype(np.float32)
    
    # Convertir a binario
    gt_bin = gt > 0.5
    pred_bin = pred > 0.5

    # Asegurarse de que ambas máscaras tengan la misma forma
    if gt_bin.shape != pred_bin.shape:
        pred_bin = cv2.resize(pred_bin.astype(np.uint8),
                              (gt_bin.shape[1], gt_bin.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)

    tp = pred_bin & gt_bin
    fp = pred_bin & (~gt_bin)

    # Crear imagen de diferencia con colores: 
    # - Rojo para areas de GT, 
    # - Blanco para TP, 
    # - Amarillo para FP.
    diff_img = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    diff_img[gt_bin] = [255, 0, 0]
    diff_img[tp] = [255, 255, 255]
    diff_img[fp] = [255, 255, 0]
    return diff_img

def main(image_id, true_pred, custom_input=None):
    # Directorios base
    input_dir = "/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fives-save/FIVESoriginal/test/image"
    label_dir = "/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fives-save/FIVESoriginal/test/label"
    inference_dir = "/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fork-roi/fivesegmentor/inference_results/RoiNetTest2bottleneck_residual"
    custom_dir = "/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fork-roi/fivesegmentor/inference_results/"
    
    # Construir nombres de archivos
    filename = f"{image_id}.png"
    input_path = os.path.join(input_dir, filename)
    label_path = os.path.join(label_dir, filename)
    
    if custom_input is not None:
        output_path = os.path.join(custom_dir, "custom_images", custom_input)
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
    
    # Si true_pred es True, igualamos la imagen de inferencia al label
    if true_pred:
        output_img = label_img.copy()
    
    # Convertir de BGR a RGB
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    
    # Convertir a escala de grises para la comparación
    label_gray = cv2.cvtColor(label_img, cv2.COLOR_RGB2GRAY)
    output_gray = cv2.cvtColor(output_img, cv2.COLOR_RGB2GRAY)
    
    # Normalizar y convertir a tensores con forma (1, 1, H, W)
    label_tensor = torch.from_numpy(label_gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    output_tensor = torch.from_numpy(output_gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    
    # Inicializar las funciones de pérdida
    dice_loss_fn = DiceLoss()
    softcldice_loss_fn = SoftCLDiceLoss(iter_=20, smooth=1e-12, exclude_background=True)
    softdicecldice_loss_fn = SoftDiceCLDiceLoss(iter_=3, alpha=0.5, smooth=1., exclude_background=True)
    conex_loss_fn = ConexLoss(reduction='mean')

    # Calcular pérdidas y tiempos
    start_time = time.time()
    dice_loss_value = dice_loss_fn(output_tensor, label_tensor)
    dice_loss_time = time.time() - start_time

    start_time = time.time()
    softcldice_loss_value = softcldice_loss_fn(label_tensor, output_tensor)
    softcldice_loss_time = time.time() - start_time

    start_time = time.time()
    softdicecldice_loss_value = softdicecldice_loss_fn(label_tensor, output_tensor)
    softdicecldice_loss_time = time.time() - start_time

    start_time = time.time()
    conex_loss_value = conex_loss_fn(output_tensor, label_tensor)
    conex_loss_time = time.time() - start_time

    dice_accuracy = (1 - dice_loss_value.item()) * 100
    softcldice_accuracy = (1 - softcldice_loss_value.item()) * 100
    softdicecldice_accuracy = (1 - softdicecldice_loss_value.item()) * 100
    conex_accuracy = (1 - conex_loss_value.item()) * 100

    print(f"Dice Loss: {dice_loss_value.item():.4f} | Accuracy: {dice_accuracy:.2f}% | Time: {dice_loss_time:.4f} s")
    print(f"SoftCLDice Loss: {softcldice_loss_value.item():.4f} | Accuracy: {softcldice_accuracy:.2f}% | Time: {softcldice_loss_time:.4f} s")
    print(f"SoftDiceCLDice Loss: {softdicecldice_loss_value.item():.4f} | Accuracy: {softdicecldice_accuracy:.2f}% | Time: {softdicecldice_loss_time:.4f} s")
    print(f"ConexLoss: {conex_loss_value.item():.4f} | Accuracy: {conex_accuracy:.2f}% | Time: {conex_loss_time:.4f} s")

    diff_img = create_difference_image(label_gray, output_gray)

    # Mostrar la imagen y las métricas
    fig, ax = plt.subplots(figsize=(14, 8))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 2, width_ratios=[3, 1])

    ax0 = plt.subplot(gs[0])
    ax0.imshow(diff_img)
    ax0.set_title("Comparison (GT, TP, FP)", fontsize=16)
    ax0.axis("off")

    metrics_text = (
        f"Dice Loss: {dice_loss_value.item():.4f}\nAccuracy: {dice_accuracy:.2f}%\nTime: {dice_loss_time:.4f} s\n\n"
        f"SoftCLDice Loss: {softcldice_loss_value.item():.4f}\nAccuracy: {softcldice_accuracy:.2f}%\nTime: {softcldice_loss_time:.4f} s\n\n"
        f"SoftDiceCLDice Loss: {softdicecldice_loss_value.item():.4f}\nAccuracy: {softdicecldice_accuracy:.2f}%\nTime: {softdicecldice_loss_time:.4f} s\n\n"
        f"ConexLoss: {conex_loss_value.item():.4f}\nAccuracy: {conex_accuracy:.2f}%\nTime: {conex_loss_time:.4f} s"
    )
    ax1 = plt.subplot(gs[1])
    ax1.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=14)
    ax1.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to compare Input, Label, Inference images and calculate loss values (Dice, SoftCLDice, SoftDiceCLDice y ConexLoss)."
    )
    parser.add_argument("image_id", type=str, help="Image identifier (e.g., 105_G)")
    parser.add_argument("--truePred", action="store_true", help="Make the inference image identical to the label image for testing loss functions.")
    parser.add_argument("--customInput", type=str, default=None,
                        help="Nombre de la imagen custom (e.g., sliced_label_105_G.png) ubicada en inference_results/custom_images")
    
    args = parser.parse_args()
    
    main(args.image_id, args.truePred, args.customInput)
