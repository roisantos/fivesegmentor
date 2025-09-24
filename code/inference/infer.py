import cv2
import torch
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/config')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/datasets')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/evaluation')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/inference')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/models')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/scripts')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/utils')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/training')))
from torchvision import transforms
from PIL import Image
from models.frnet import FRNet
from models.common import *
from torchvision.utils import save_image

def run_inference_on_directory(input_dir, output_dir, model_path):
    # Create an instance of the model
    model = FRNet(ch_in=1, ch_out=1, ls_mid_ch=[32]*6, out_k_size=11, k_size=3,
                  cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock)

    # Load the trained model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to inference mode

    # Check if a GPU is available and if not, use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each file in the input directory
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        # Load an image
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Skipping file due to load failure: {file_path}")
            continue

        # Normalize the image
        image = image.astype("float32") / 255

        # Apply padding to make dimensions multiples of 32
        pad_x = (image.shape[1] // 32 + 1) * 32 - image.shape[1]
        pad_y = (image.shape[0] // 32 + 1) * 32 - image.shape[0]
        image = cv2.copyMakeBorder(image, pad_y//2, pad_y//2, pad_x//2, pad_x//2, cv2.BORDER_CONSTANT, value=0)

        # Convert image to a tensor
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        image_tensor = image_tensor.to(device)

        # Perform the inference
        with torch.no_grad():
            output = model(image_tensor)

        # Save the tensor and image
        torch.save(output.cpu(), os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_output_tensor.pth"))
        save_image(output, os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_output_image.png"))

# Directory containing images
input_dir = r"dataset\FIVES512\test\scaffoldsCENTER1024masked\masked"
# Directory to save the results
output_dir = os.path.join('inference_results', '2025-06-28 after 280 epochs')
# Path to the trained model
model_path = 'C:\Users\roi\OneDrive - Universidade de Santiago de Compostela\investigacion\vesselview_pth\model_best.pth'

run_inference_on_directory(input_dir, output_dir, model_path)
