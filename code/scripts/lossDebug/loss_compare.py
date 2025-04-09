#!/usr/bin/env python3
import argparse
import os
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import sys
import time

# Add the parent directory ("code") to import modules from the codebase
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import loss functions including CompositeLoss
from training.loss import DiceLoss, SoftCLDiceLossStrict, ConexLoss, CompositeLoss

def create_difference_image(ground_truth, prediction):
    gt = ground_truth.astype(np.float32)
    pred = prediction.astype(np.float32)
    
    # Convert images to binary masks
    gt_bin = gt > 0.5
    pred_bin = pred > 0.5

    # Ensure that both masks have the same shape
    if gt_bin.shape != pred_bin.shape:
        pred_bin = cv2.resize(pred_bin.astype(np.uint8),
                              (gt_bin.shape[1], gt_bin.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)

    tp = pred_bin & gt_bin
    fp = pred_bin & (~gt_bin)

    # Create a difference image with colors:
    # - Red for GT areas,
    # - White for TP,
    # - Yellow for FP.
    diff_img = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    diff_img[gt_bin] = [255, 0, 0]
    diff_img[tp] = [255, 255, 255]
    diff_img[fp] = [255, 255, 0]
    return diff_img

def main(image_id, true_pred, custom_input=None):
    # Base directories
    input_dir = "/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fives-save/FIVESoriginal/test/image"
    label_dir = "/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fives-save/FIVESoriginal/test/label"
    inference_dir = "/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fork-roi/fivesegmentor/inference_results/RoiNetTest2bottleneck_residual"
    custom_dir = "/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fork-roi/fivesegmentor/inference_results/"
    
    # Construct file names
    filename = f"{image_id}.png"
    input_path = os.path.join(input_dir, filename)
    label_path = os.path.join(label_dir, filename)
    
    if custom_input is not None:
        output_path = os.path.join(custom_dir, "custom_images", custom_input)
    else:
        output_filename = f"{image_id}_output_image.png"
        output_path = os.path.join(inference_dir, output_filename)
    
    # Check that the files exist
    for path, desc in [(input_path, "input"), (label_path, "label"), (output_path, "inference")]:
        if not os.path.exists(path):
            print(f"Error: {desc} image not found at {path}")
            return
    
    # Load images
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
    
    # If true_pred is set, use the label image as the inference image
    if true_pred:
        output_img = label_img.copy()
    
    # Convert images from BGR to RGB
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    
    # Convert images to grayscale for comparison
    label_gray = cv2.cvtColor(label_img, cv2.COLOR_RGB2GRAY)
    output_gray = cv2.cvtColor(output_img, cv2.COLOR_RGB2GRAY)
    
    # Normalize and convert to tensors with shape (1, 1, H, W)
    label_tensor = torch.from_numpy(label_gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    output_tensor = torch.from_numpy(output_gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    
    # Define the available loss functions in a dictionary.
    # The "Composite" option mixes:
    #   - StrictCLDice (SoftCLDiceLossStrict) at 33%
    #   - DiceLoss at 33%
    #   - ConexLoss at 34%
    loss_options = {
        "Dice": DiceLoss(),
        "StrictCLDice": SoftCLDiceLossStrict(iter_=25, smooth=1e-6, penalty_power=5., exclude_background=False),
        "Conex": ConexLoss(reduction='mean'),
        "Composite": CompositeLoss([
            (SoftCLDiceLossStrict(iter_=25, smooth=1e-6, penalty_power=5., exclude_background=False), 0.5),
            (DiceLoss(), 0.5),
            (ConexLoss(reduction='mean'), 0.)
        ])
    }
    
    # For each loss function in the dictionary, compute its value and timing
    results = {}
    for key, loss_fn in loss_options.items():
        start_time = time.time()
        loss_val = loss_fn(output_tensor, label_tensor)
        elapsed_time = time.time() - start_time
        accuracy = (1 - loss_val.item()) * 100
        results[key] = {
            "loss": loss_val.item(),
            "accuracy": accuracy,
            "time": elapsed_time
        }
        print(f"{key} Loss: {loss_val.item():.4f} | Accuracy: {accuracy:.2f}% | Time: {elapsed_time:.4f} s")
    
    # Create a text block for all loss metrics for the display panel.
    metrics_text = ""
    for key in results:
        metrics_text += (
            f"{key} Loss: {results[key]['loss']:.4f}\n"
            f"Accuracy: {results[key]['accuracy']:.2f}%\n"
            f"Time: {results[key]['time']:.4f} s\n\n"
        )

    diff_img = create_difference_image(label_gray, output_gray)

    # Create figure with two columns: image on the left, loss metrics on the right
    fig, ax = plt.subplots(figsize=(14, 8))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 2, width_ratios=[3, 1])

    ax0 = plt.subplot(gs[0])
    ax0.imshow(diff_img)
    ax0.set_title("Comparison (GT, TP, FP)", fontsize=16)
    ax0.axis("off")

    ax1 = plt.subplot(gs[1])
    ax1.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=14)
    ax1.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to compare Input, Label, and Inference images and calculate all loss values "
                    "from a dictionary (including CompositeLoss)."
    )
    parser.add_argument("image_id", type=str, help="Image identifier (e.g., 105_G)")
    parser.add_argument("--truePred", action="store_true", help="Set inference image equal to the label image for testing loss functions.")
    parser.add_argument("--customInput", type=str, default=None,
                        help="Name of the custom image (e.g., sliced_label_105_G.png) in inference_results/custom_images")
    
    args = parser.parse_args()
    
    main(args.image_id, args.truePred, args.customInput)
