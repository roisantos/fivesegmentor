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

def zoom_image(img, x, y, w, h):
    """Crops the image to the rectangle defined by (x, y, w, h)."""
    return img[y:y+h, x:x+w]

def main(image_id, zoom_x, zoom_y, zoom_w, zoom_h):
    # Base directories
    input_dir = "/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fives-save/FIVESoriginal/test/image"
    label_dir = "/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fives-save/FIVESoriginal/test/label"
    inference_dir = "/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fork-roi/fivesegmentor/inference_results/RoiNetTest2bottleneck_residual"
    
    # Build filenames
    filename = f"{image_id}.png"
    input_path = os.path.join(input_dir, filename)
    label_path = os.path.join(label_dir, filename)
    output_filename = f"{image_id}_output_image.png"
    output_path = os.path.join(inference_dir, output_filename)
    
    # Check file existence
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
    
    # Convert from BGR to RGB
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    
    # Initialize soft skeletonizer
    skeletonizer = SoftSkeletonize(num_iter=100)
    
    # Convert label and inference to grayscale
    label_gray = cv2.cvtColor(label_img, cv2.COLOR_RGB2GRAY)
    output_gray = cv2.cvtColor(output_img, cv2.COLOR_RGB2GRAY)
    
    # Normalize and convert to tensor of shape (1,1,H,W)
    label_tensor = torch.from_numpy(label_gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    output_tensor = torch.from_numpy(output_gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    
    # Compute soft skeletons
    skeleton_label_tensor = skeletonizer(label_tensor)
    skeleton_output_tensor = skeletonizer(output_tensor)
    
    # Convert skeleton tensors back to numpy arrays scaled to [0,255]
    skeleton_label = (skeleton_label_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    skeleton_output = (skeleton_output_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    
    # Create zoomed versions (crop using provided zoom parameters)
    zoom_skel_label = zoom_image(skeleton_label, zoom_x, zoom_y, zoom_w, zoom_h)
    zoom_skel_output = zoom_image(skeleton_output, zoom_x, zoom_y, zoom_w, zoom_h)
    
    # Create a figure with 3 rows and 3 columns.
    # Row 0: Full images (Input, Label, Inference)
    # Row 1: Skeletonized outputs (first column shows a note)
    # Row 2: Zoomed skeletonized outputs (first column shows a note)
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    
    # Row 0
    axes[0, 0].imshow(input_img)
    axes[0, 0].set_title("Full Input Image")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(label_img)
    axes[0, 1].set_title("Full Label Image")
    axes[0, 1].axis("off")
    
    axes[0, 2].imshow(output_img)
    axes[0, 2].set_title("Full Inference Image")
    axes[0, 2].axis("off")
    
    # Row 1: Skeletonized outputs
    axes[1, 0].axis("off")
    axes[1, 0].text(0.5, 0.5, "No skeleton computed for input", 
                    horizontalalignment='center', verticalalignment='center', fontsize=12)
    
    axes[1, 1].imshow(skeleton_label, cmap="gray")
    axes[1, 1].set_title("Skeletonized Label")
    axes[1, 1].axis("off")
    
    axes[1, 2].imshow(skeleton_output, cmap="gray")
    axes[1, 2].set_title("Skeletonized Inference")
    axes[1, 2].axis("off")
    
    # Row 2: Zoomed skeletonized outputs
    axes[2, 0].axis("off")
    axes[2, 0].text(0.5, 0.5, "Zoom N/A", horizontalalignment='center',
                    verticalalignment='center', fontsize=12)
    
    axes[2, 1].imshow(zoom_skel_label, cmap="gray")
    axes[2, 1].set_title(f"Zoomed Skeletonized Label\n(x={zoom_x}, y={zoom_y}, w={zoom_w}, h={zoom_h})")
    axes[2, 1].axis("off")
    
    axes[2, 2].imshow(zoom_skel_output, cmap="gray")
    axes[2, 2].set_title(f"Zoomed Skeletonized Inference\n(x={zoom_x}, y={zoom_y}, w={zoom_w}, h={zoom_h})")
    axes[2, 2].axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to compare Input, Label, Inference images and their skeletonized outputs with a zoomed view."
    )
    parser.add_argument("image_id", type=str, help="Image identifier (e.g., 105_G)")
    # Four optional positional arguments for zoom; default values are provided.
    parser.add_argument("zoom_x", type=int, nargs="?", default=50, help="Zoom top-left x-coordinate (default: 50)")
    parser.add_argument("zoom_y", type=int, nargs="?", default=50, help="Zoom top-left y-coordinate (default: 50)")
    parser.add_argument("zoom_w", type=int, nargs="?", default=100, help="Zoom width (default: 100)")
    parser.add_argument("zoom_h", type=int, nargs="?", default=100, help="Zoom height (default: 100)")
    
    args = parser.parse_args()
    
    main(args.image_id, args.zoom_x, args.zoom_y, args.zoom_w, args.zoom_h)
