#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def create_difference_image(ground_truth, prediction):
    gt = ground_truth.astype(np.float32)
    pred = prediction.astype(np.float32)
    if gt.ndim == 3:
        gt = gt[..., 0]
    if pred.ndim == 3:
        pred = pred[..., 0]
    if gt.max() > 1:
        gt /= 255.0
    if pred.max() > 1:
        pred /= 255.0
    gt_bin = gt > 0.5
    pred_bin = pred > 0.5

    # Aseguramos que ambas máscaras tengan la misma forma
    if gt_bin.shape != pred_bin.shape:
        pred_bin = cv2.resize(pred_bin.astype(np.uint8),
                              (gt_bin.shape[1], gt_bin.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)

    tp = pred_bin & gt_bin
    fp = pred_bin & (~gt_bin)

    diff_img = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    # Rojo para GT
    diff_img[gt_bin] = [255, 0, 0]
    # Blanco para TP
    diff_img[tp] = [255, 255, 255]
    # Amarillo para FP
    diff_img[fp] = [255, 255, 0]
    return diff_img

def main():
    # Directorios de imágenes, etiquetas e inferencias
    input_dir = "/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fives-save/FIVESoriginal/test/image"
    gt_dir    = "/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fives-save/FIVESoriginal/test/label"
    inference_base = "/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fork-roi/fivesegmentor/inference_results/"

    # Directorio de comparación (puedes ajustar según tu organización)
    comparison_dir = "SantosNet_Various"

    # Comparativas: "Título a mostrar": "Nombre de la carpeta"
    comparativas = {
        "VesselView":    "RoiNetTest2bottleneck_residual",
        "SantosNet_CPCh": "SantosNet_CPCh",
        "SantosNet_GCh": "SantosNet_GCh"
    }

    if len(sys.argv) != 2:
        print("Uso: python3 script.py <image_id>")
        sys.exit(1)
    image_id = sys.argv[1]

    input_path = os.path.join(input_dir, f"{image_id}.png")
    gt_path    = os.path.join(gt_dir,   f"{image_id}.png")

    # Rutas de inferencia para cada comparativa
    inference_paths = {
        title: os.path.join(inference_base, folder, f"{image_id}_output_image.png")
        for title, folder in comparativas.items()
    }

    # Cargar imagen de input y ground truth
    try:
        input_img = mpimg.imread(input_path)
        gt_img    = mpimg.imread(gt_path)
    except Exception as e:
        print("Error al cargar la imagen de input o ground truth.")
        raise e

    # Cargar inferencias
    inf_imgs = {}
    for title, path in inference_paths.items():
        try:
            inf_imgs[title] = mpimg.imread(path)
        except Exception as e:
            print(f"Error al cargar la inferencia para {title} en {path}")
            raise e

    # Crear imágenes de diferencia (GT vs inferencia)
    diff_imgs = {
        title: create_difference_image(gt_img, inf_imgs[title])
        for title in comparativas.keys()
    }

    # Orden de las columnas: Input, Ground Truth y luego cada comparativa
    #full_order = ["Input", "Ground Truth"] + list(comparativas.keys())
    #full_order = ["Input", "Ground Truth"] + list(comparativas.keys())
    full_order = list(comparativas.keys())
    # Diccionario de imágenes a mostrar:
    # - Input y Ground Truth se muestran en su versión completa (a color)
    # - Para cada comparativa se muestra la imagen de diferencia
    images_display = {"Input": input_img, "Ground Truth": gt_img}
    images_display.update(diff_imgs)

    # Configuración de la figura: 1 fila con tantas columnas como imágenes
    n_cols = len(full_order)
    # Se define un ancho base por columna para que sean lo más grandes posible sin exceder 1080p
    fig, axs = plt.subplots(1, n_cols, figsize=(n_cols * 10, 10))
    plt.subplots_adjust(wspace=0.05)

    # Mostrar cada imagen con su título
    for i, key in enumerate(full_order):
        axs[i].imshow(images_display[key])
        axs[i].set_title(key, fontsize=10)
        axs[i].axis("off")

    # Guardar la figura y mostrarla
    save_dir = "inferCompareGraphs"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, comparison_dir), exist_ok=True)
    output_path = os.path.join(save_dir, comparison_dir, f"{image_id}_compare.png")
    fig.savefig(output_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
