import os
import sys
import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

# Ensure project root ("code" folder) is in PYTHONPATH so we can import our modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

# Import the RoiNet model definition and any required blocks
from models.roinet import RoiNet  # k_size will be configured to 9 for RoiNet9
from models.common import *       # ResidualBlock etc.

def pad_to_multiple(img: np.ndarray, multiple: int = 32) -> np.ndarray:
    """Pad HxWxC (or HxW) array so both spatial dims are multiples of `multiple`."""
    h, w = img.shape[:2]
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    if img.ndim == 3:
        border_val = [0, 0, 0]
    else:
        border_val = 0
    padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                                borderType=cv2.BORDER_CONSTANT, value=border_val)
    return padded


def collect_predictions(model, device, image_dir: str, label_dir: str):
    """Traverse `image_dir`, run model, return concatenated GT and prediction arrays."""
    gt_all = []  # ground-truth (flattened)
    pred_all = []  # predicted probabilities (flattened)

    # Accept common image extensions
    valid_ext = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    file_names = sorted(os.listdir(image_dir))
    for fname in tqdm(file_names, desc="Processing images", unit="img"):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in valid_ext:
            continue

        img_path = os.path.join(image_dir, fname)
        lbl_path = os.path.join(label_dir, fname)

        # Load image (BGR) and convert to float32 [0,1]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Could not read image {img_path}. Skipping.")
            continue
        img = img.astype(np.float32) / 255.0

        # Pad to multiple of 32 (as used in training)
        img_pad = pad_to_multiple(img, multiple=32)

        # Prepare tensor (C,H,W) -> (1,C,H,W)
        img_tensor = torch.from_numpy(img_pad.transpose(2, 0, 1)).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
        prob_map = output.cpu().numpy()[0, 0]  # shape (H_pad, W_pad)

        # Load label as grayscale, threshold to binary 0/1
        lbl = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        if lbl is None:
            print(f"[WARN] Could not read label {lbl_path}. Skipping.")
            continue
        _, lbl = cv2.threshold(lbl, 127, 1, cv2.THRESH_BINARY)

        # Pad label to same shape as prob_map
        lbl_pad = pad_to_multiple(lbl, multiple=32)

        if lbl_pad.shape != prob_map.shape:
            # Rare mismatch due to rounding; resize label with nearest neighbor
            lbl_pad = cv2.resize(lbl_pad, (prob_map.shape[1], prob_map.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Flatten and collect
        gt_all.append(lbl_pad.flatten())
        pred_all.append(prob_map.flatten())

    if not gt_all:
        raise RuntimeError("No valid image/label pairs were processed.")

    gt_concat = np.concatenate(gt_all).astype(np.uint8)
    pred_concat = np.concatenate(pred_all).astype(np.float32)
    return gt_concat, pred_concat


def plot_and_save_roc(gt: np.ndarray, pred: np.ndarray, save_path: str):
    """Plot ROC curve given flattened GT and prediction arrays."""
    fpr, tpr, _ = roc_curve(gt, pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"ROC curve saved to {save_path} (AUC = {roc_auc:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Compute and plot ROC curve for RoiNet9 on a test dataset.")
    parser.add_argument('--image-dir', type=str, required=True, help='Path to directory with test images.')
    parser.add_argument('--label-dir', type=str, required=True, help='Path to directory with ground-truth labels.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model weights (.pth).')
    parser.add_argument('--output', type=str, default='roc_curve.png', help='Output path for ROC curve figure.')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model (RoiNet9 configuration)
    model = RoiNet(ch_in=3, ch_out=1, k_size=9)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # Run inference and collect predictions
    gt, pred = collect_predictions(model, device, args.image_dir, args.label_dir)

    # Plot ROC
    plot_and_save_roc(gt, pred, args.output)


if __name__ == '__main__':
    main() 