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

def zoom_image(img, x, y, w, h):
    """Recorta la región (x, y, w, h) de la imagen."""
    return img[y:y+h, x:x+w]

def main():
    # Directorios de imágenes, etiquetas e inferencias
    input_dir = "/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fives-save/FIVESoriginal/test/image"
    gt_dir    = "/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fives-save/FIVESoriginal/test/label"
    inference_base = "/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fork-roi/fivesegmentor/inference_results/"

    comparison_dir = "SantosNetCPCh"

    # Comparativas: "Título a mostrar": "Nombre de la carpeta"
    comparativas = {
        "VesselView": "RoiNetTest2bottleneck_residual",
        "SantosNet_CPCh":  "SantosNet_CPCh"
        }

    if len(sys.argv) not in [2, 6]:
        print("Uso: python3 script.py <image_id> [zoom_x zoom_y zoom_width zoom_height]")
        sys.exit(1)
    image_id = sys.argv[1]
    if len(sys.argv) == 6:
        try:
            zoom_x = int(sys.argv[2])
            zoom_y = int(sys.argv[3])
            zoom_w = int(sys.argv[4])
            zoom_h = int(sys.argv[5])
        except ValueError:
            print("Las coordenadas de zoom deben ser números enteros.")
            sys.exit(1)
    else:
        zoom_x, zoom_y, zoom_w, zoom_h = 50, 50, 100, 100

    input_path = os.path.join(input_dir, f"{image_id}.png")
    gt_path    = os.path.join(gt_dir,   f"{image_id}.png")

    # Rutas de inferencia para cada comparativa
    inference_paths = {
        title: os.path.join(inference_base, folder, f"{image_id}_output_image.png")
        for title, folder in comparativas.items()
    }

    # Cargar input y ground truth
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

    # Crear versiones recortadas (zoom)
    zoom_input = zoom_image(input_img, zoom_x, zoom_y, zoom_w, zoom_h)
    zoom_gt    = zoom_image(gt_img,    zoom_x, zoom_y, zoom_w, zoom_h)
    zoom_inf   = {title: zoom_image(inf_imgs[title], zoom_x, zoom_y, zoom_w, zoom_h)
                  for title in comparativas.keys()}
    zoom_diff  = {title: zoom_image(diff_imgs[title], zoom_x, zoom_y, zoom_w, zoom_h)
                  for title in comparativas.keys()}

    # Orden de las columnas: Input y Ground Truth siempre presentes, seguidos de las comparativas
    full_order = ["Input", "Ground Truth"] + list(comparativas.keys())

    # Ajuste automático de la figura: se calcula el número de columnas y se define el ancho
    n_cols = len(full_order)
    base_col_width = 16 / 9  # Con 9 columnas la anchura es 16 pulgadas en el diseño original
    max_width = 16         # Máximo ancho (en pulgadas) para una pantalla 1080p
    fig_width = min(n_cols * base_col_width, max_width)
    fig_height = 5.5       # Altura fija para 3 filas
    fig, axs = plt.subplots(nrows=3, ncols=n_cols, figsize=(fig_width, fig_height))
    
    plt.subplots_adjust(hspace=0.00, wspace=0.05)

    # ============ Fila 0: Imágenes completas ============
    # Se muestran el input (a color), GT y las inferencias
    images_full = {"Input": input_img, "Ground Truth": gt_img}
    images_full.update(inf_imgs)
    for i, key in enumerate(full_order):
        axs[0, i].imshow(images_full[key])
        axs[0, i].set_title(key, fontsize=10)
        axs[0, i].axis("off")

    # ============ Fila 1: Imágenes en zoom ============
    images_zoom = {"Input": zoom_input, "Ground Truth": zoom_gt}
    images_zoom.update(zoom_inf)
    for i, key in enumerate(full_order):
        axs[1, i].imshow(images_zoom[key])
        axs[1, i].axis("off")

    # ============ Fila 2: Imágenes de diferencia (zoom) ============
    # Las dos primeras columnas se dejan para leyenda
    axs[2, 0].axis("off")
    axs[2, 1].text(0.5, 0.5, "Difference\n(Red: GT, Yellow: FP)\n(Zoom)",
                   fontsize=10, ha="center", va="center")
    axs[2, 1].axis("off")
    for i in range(2, n_cols):
        title = full_order[i]
        axs[2, i].imshow(zoom_diff[title])
        axs[2, i].axis("off")

    # Crear carpeta de salida si no existe
    save_dir = "inferCompareGraphs"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/{comparison_dir}", exist_ok=True)

    output_path = os.path.join(save_dir, f"{comparison_dir}/{image_id}_compare.png")
    fig.savefig(output_path, dpi=300)
    plt.show()  # Descomentar para visualizar en pantalla

if __name__ == "__main__":
    main()
