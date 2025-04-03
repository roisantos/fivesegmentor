import os
import sys
import time
import csv
import cv2
import numpy as np
import torch
from torchvision.utils import save_image
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score, jaccard_score, matthews_corrcoef, confusion_matrix

# Agregar el directorio padre ("code") para poder importar los módulos de models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importar los modelos y bloques comunes.
from models.roinet import *
from models.extraModels import * 
from models.common import ResidualBlock, SimpleResBlock  # Asegúrate de que estos bloques existan
from models.SantosNet import *

def compute_dice(pred, gt, eps=1e-6):
    """Calcula el coeficiente Dice a partir de arrays binarios (numpy) de predicción y ground truth."""
    intersection = np.sum(pred * gt)
    return (2.0 * intersection) / (np.sum(pred) + np.sum(gt) + eps)

def run_inference_on_directory(image_dir, label_dir, output_dir, model_config, model_path):
    """
    Ejecuta la inferencia en un directorio de imágenes usando el modelo configurado.
    
    Parámetros:
      - image_dir: Directorio de imágenes de entrada.
      - label_dir: Directorio de etiquetas correspondientes.
      - output_dir: Directorio donde se guardarán los resultados.
      - model_config: Diccionario con la configuración del modelo:
            {
              "type": "NombreClase",      # Ej. RoiNetTest1bottleneck
              "ch_in": 3,
              "ch_out": 1,
              "cls_init_block": "ResidualBlock" o "SimpleResBlock",
              "cls_conv_block": "ResidualBlock" o "SimpleResBlock"
            }
      - model_path: Ruta al archivo .pth con los pesos entrenados.
    """
    # Mapping de imagenes según la última letra del nombre
    type_map = {
        'N': 'Normal',
        'A': 'AMD',
        'D': 'DR',
        'G': 'Glaucoma'
    }
    
    # Seleccionar la clase del modelo a partir del parámetro "type"
    model_type = model_config["type"]
    if model_type == "RoiNetTest1bottleneck":
        ModelClass = RoiNetTest1bottleneck
    elif model_type == "RoiNetTest2bottleneck":
        ModelClass = RoiNetTest2bottleneck
    elif model_type == "RoiNetTest3bottleneck":
        ModelClass = RoiNetTest3bottleneck
    elif model_type == "RoiNetNoSkip":
        ModelClass = RoiNetNoSkip
    elif model_type == "RoiNetSumFusion":
        ModelClass = RoiNetSumFusion
    elif model_type == "RoiNetAttnSkip":
        ModelClass = RoiNetAttnSkip
    elif model_type == "RoiNetResSkip":
        ModelClass = RoiNetResSkip
    elif model_type == "RoiNetConcatPlus":
        ModelClass = RoiNetConcatPlus
    elif model_type == "RoiNetMultiSkip":
        ModelClass = RoiNetMultiSkip
    elif model_type == "SantosNet_GCh":
        ModelClass = SantosNet_GCh
    elif model_type == "SantosNet_PCh":
        ModelClass = SantosNet_PCh
    elif model_type == "SantosNet_CPCh":
        ModelClass = SantosNet_CPCh
    else:
        raise ValueError(f"Tipo de modelo desconocido: {model_type}")

    
    # Seleccionar los bloques a usar según la configuración
    block_name_init = model_config["cls_init_block"]
    block_name_conv = model_config["cls_conv_block"]
    if block_name_init == "ResidualBlock":
        BlockInit = ResidualBlock
    elif block_name_init == "SimpleResBlock":
        BlockInit = SimpleResBlock
    else:
        raise ValueError(f"Bloque de inicialización desconocido: {block_name_init}")
    
    if block_name_conv == "ResidualBlock":
        BlockConv = ResidualBlock
    elif block_name_conv == "SimpleResBlock":
        BlockConv = SimpleResBlock
    else:
        raise ValueError(f"Bloque convolucional desconocido: {block_name_conv}")
    
    # Instanciar el modelo con los parámetros: canales de entrada/salida, kernel=9 y bloques seleccionados.
    

    if "custom_weights" in model_config:
         model = ModelClass(ch_in=model_config["ch_in"],
                       ch_out=model_config["ch_out"],
                       k_size=9,
                       custom_weights=model_config["custom_weights"],
                       cls_init_block=BlockInit,
                       cls_conv_block=BlockConv)
    else:
        model = ModelClass(ch_in=model_config["ch_in"],
                       ch_out=model_config["ch_out"],
                       k_size=9,
                       cls_init_block=BlockInit,
                       cls_conv_block=BlockConv)

    # Cargar pesos entrenados
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Usar GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    os.makedirs(output_dir, exist_ok=True)

    inference_times = []
    dice_scores = []
    results_for_csv = []
    per_type = {}

    for filename in os.listdir(image_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename)

        base_name = os.path.splitext(filename)[0]
        image_type_letter = base_name[-1]
        image_type = type_map.get(image_type_letter, "Unknown")

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Warning: No se pudo cargar la imagen: {image_path}")
            continue

        image = image.astype("float32") / 255.0

        # Padding para que las dimensiones sean múltiplos de 32
        pad_x = (image.shape[1] // 32 + 1) * 32 - image.shape[1]
        pad_y = (image.shape[0] // 32 + 1) * 32 - image.shape[0]
        if pad_x == 32:
            pad_x = 0
        if pad_y == 32:
            pad_y = 0

        image_padded = cv2.copyMakeBorder(image, pad_y // 2, pad_y // 2,
                                          pad_x // 2, pad_x // 2,
                                          cv2.BORDER_CONSTANT, value=0)
        image_transposed = np.transpose(image_padded, (2, 0, 1))
        image_tensor = torch.from_numpy(image_transposed).unsqueeze(0).to(device)

        start_time = time.perf_counter()
        with torch.no_grad():
            output = model(image_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        inference_time = time.perf_counter() - start_time
        inference_times.append(inference_time)

        print(f"{filename} - Tiempo de inferencia: {inference_time*1000:.2f} ms | "
              f"min: {output.min().item():.4f}, max: {output.max().item():.4f}, mean: {output.mean().item():.4f}")

        base_name = os.path.splitext(filename)[0]
        tensor_save_path = os.path.join(output_dir, f"{base_name}_output_tensor.pth")
        image_save_path = os.path.join(output_dir, f"{base_name}_output_image.png")
        # Guardar la imagen de salida
        save_image(output, image_save_path)

        gt_label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if gt_label is None:
            print(f"Warning: No se pudo cargar la etiqueta para {filename}")
            continue

        _, gt_label = cv2.threshold(gt_label, 127, 1, cv2.THRESH_BINARY)

        pad_x_label = (gt_label.shape[1] // 32 + 1) * 32 - gt_label.shape[1]
        pad_y_label = (gt_label.shape[0] // 32 + 1) * 32 - gt_label.shape[0]
        if pad_x_label == 32:
            pad_x_label = 0
        if pad_y_label == 32:
            pad_y_label = 0

        gt_label_padded = cv2.copyMakeBorder(gt_label, pad_y_label // 2, pad_y_label // 2,
                                             pad_x_label // 2, pad_x_label // 2,
                                             cv2.BORDER_CONSTANT, value=0)

        pred_prob = output.cpu().numpy()[0, 0]
        pred_binary = (pred_prob > 0.5).astype(np.float32)
        dice = compute_dice(pred_binary, gt_label_padded.astype(np.float32))
        dice_scores.append(dice)

        gt_flat = gt_label_padded.flatten().astype(np.int32)
        pred_flat = pred_binary.flatten().astype(np.int32)
        pred_prob_flat = pred_prob.flatten()

        try:
            auc = roc_auc_score(gt_flat, pred_prob_flat)
        except Exception:
            auc = float('nan')
        f1 = f1_score(gt_flat, pred_flat)
        acc = accuracy_score(gt_flat, pred_flat)
        sen = recall_score(gt_flat, pred_flat)
        pre = precision_score(gt_flat, pred_flat)
        iou = jaccard_score(gt_flat, pred_flat)
        mcc = matthews_corrcoef(gt_flat, pred_flat)
        cm = confusion_matrix(gt_flat, pred_flat)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            spe = tn / (tn + fp) if (tn + fp) != 0 else 0.0
        else:
            spe = float('nan')

        print(f"{filename} - Dice: {dice:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, Acc: {acc:.4f}, "
              f"Sen: {sen:.4f}, Spe: {spe:.4f}, Pre: {pre:.4f}, IoU: {iou:.4f}, MCC: {mcc:.4f}")

        results_for_csv.append({
            "filename": filename,
            "image_type": image_type,
            "inference_time_ms": inference_time * 1000,
            "dice_score": dice,
            "auc": auc,
            "f1": f1,
            "acc": acc,
            "sen": sen,
            "spe": spe,
            "pre": pre,
            "iou": iou,
            "mcc": mcc
        })

        if image_type not in per_type:
            per_type[image_type] = {
                "times": [], "dice": [],
                "auc": [], "f1": [], "acc": [],
                "sen": [], "spe": [], "pre": [],
                "iou": [], "mcc": []
            }
        per_type[image_type]["times"].append(inference_time * 1000)
        per_type[image_type]["dice"].append(dice)
        per_type[image_type]["auc"].append(auc)
        per_type[image_type]["f1"].append(f1)
        per_type[image_type]["acc"].append(acc)
        per_type[image_type]["sen"].append(sen)
        per_type[image_type]["spe"].append(spe)
        per_type[image_type]["pre"].append(pre)
        per_type[image_type]["iou"].append(iou)
        per_type[image_type]["mcc"].append(mcc)

    avg_time = np.mean(inference_times) * 1000
    avg_dice = np.mean(dice_scores) if dice_scores else 0.0
    print(f"\nTiempo de inferencia promedio: {avg_time:.2f} ms")
    print(f"Dice promedio: {avg_dice:.4f}\n")

    for typ, stats in per_type.items():
        avg_time_type = np.mean(stats["times"])
        avg_dice_type = np.mean(stats["dice"])
        avg_auc = np.mean(stats["auc"])
        avg_f1 = np.mean(stats["f1"])
        avg_acc = np.mean(stats["acc"])
        avg_sen = np.mean(stats["sen"])
        avg_spe = np.mean(stats["spe"])
        avg_pre = np.mean(stats["pre"])
        avg_iou = np.mean(stats["iou"])
        avg_mcc = np.mean(stats["mcc"])
        print(f"Tipo: {typ} - Tiempo: {avg_time_type:.2f} ms, Dice: {avg_dice_type:.4f}, AUC: {avg_auc:.4f}, "
              f"F1: {avg_f1:.4f}, Acc: {avg_acc:.4f}, Sen: {avg_sen:.4f}, Spe: {avg_spe:.4f}, "
              f"Pre: {avg_pre:.4f}, IoU: {avg_iou:.4f}, MCC: {avg_mcc:.4f}")

    csv_file_path = os.path.join(output_dir, "results.csv")
    with open(csv_file_path, "w", newline="") as csvfile:
        fieldnames = ["filename", "image_type", "inference_time_ms", "dice_score",
                      "auc", "f1", "acc", "sen", "spe", "pre", "iou", "mcc"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_for_csv:
            writer.writerow(row)
    print(f"\nResultados por imagen escritos en {csv_file_path}")

    summary_csv_path = os.path.join(output_dir, "summary_by_type.csv")
    with open(summary_csv_path, "w", newline="") as csvfile:
        fieldnames = ["image_type", "avg_inference_time_ms", "avg_dice_score",
                      "avg_auc", "avg_f1", "avg_acc", "avg_sen", "avg_spe",
                      "avg_pre", "avg_iou", "avg_mcc"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for typ, stats in per_type.items():
            writer.writerow({
                "image_type": typ,
                "avg_inference_time_ms": np.mean(stats["times"]),
                "avg_dice_score": np.mean(stats["dice"]),
                "avg_auc": np.mean(stats["auc"]),
                "avg_f1": np.mean(stats["f1"]),
                "avg_acc": np.mean(stats["acc"]),
                "avg_sen": np.mean(stats["sen"]),
                "avg_spe": np.mean(stats["spe"]),
                "avg_pre": np.mean(stats["pre"]),
                "avg_iou": np.mean(stats["iou"]),
                "avg_mcc": np.mean(stats["mcc"])
            })
    print(f"Resumen por tipo escrito en {summary_csv_path}")

# ------------------ Configuración de Usuario ------------------
# Estas rutas se deben actualizar según el entorno.
image_dir = r"/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fork-roi/fivesegmentor/dataset/FIVES/test/image"
label_dir = r"/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fork-roi/fivesegmentor/dataset/FIVES/test/label"
