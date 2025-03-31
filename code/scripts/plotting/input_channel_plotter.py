#!/usr/bin/env python3
import argparse
import os
import cv2
import matplotlib.pyplot as plt

def main(image_id):
    # Define base directories
    input_dir = "/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fives-save/FIVESoriginal/test/image"
    label_dir = "/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fives-save/FIVESoriginal/test/label"

    # Build file paths (assuming file format is "[image_id].png")
    input_path = os.path.join(input_dir, f"{image_id}.png")
    label_path = os.path.join(label_dir, f"{image_id}.png")

    # Check if the files exist
    for path, desc in [(input_path, "Input"), (label_path, "Label")]:
        if not os.path.exists(path):
            print(f"Error: {desc} image not found at {path}")
            return

    # Read the input image (in color) and convert to RGB for proper display with matplotlib
    input_img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if input_img is None:
        print(f"Error loading input image: {input_path}")
        return
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

    # Read the label image; here we assume label is grayscale
    label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if label_img is None:
        print(f"Error loading label image: {label_path}")
        return

    # Create a grayscale version of the input image using cv2 (convert from RGB to grayscale)
    input_bw = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)

    # Extract individual color channels (input_img is already in RGB)
    red_channel = input_img[:, :, 0]
    green_channel = input_img[:, :, 1]
    blue_channel = input_img[:, :, 2]

    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Top row: Full images
    axes[0, 0].imshow(input_img)
    axes[0, 0].set_title("Full-Color Input")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(input_bw, cmap="gray")
    axes[0, 1].set_title("Grayscale Input")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(label_img, cmap="gray")
    axes[0, 2].set_title("Label")
    axes[0, 2].axis("off")

    # Bottom row: Individual color channels
    axes[1, 0].imshow(red_channel, cmap="gray")
    axes[1, 0].set_title("Red Channel")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(green_channel, cmap="gray")
    axes[1, 1].set_title("Green Channel")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(blue_channel, cmap="gray")
    axes[1, 2].set_title("Blue Channel")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot input image channels and label for a given image identifier."
    )
    parser.add_argument("image_id", type=str, help="Image identifier (e.g., 105_G)")
    args = parser.parse_args()
    main(args.image_id)
    #Example usage: python3 code/scripts/plotting/input_channel_plotter.py 105_G
