import os
import sys
import cv2
import json

# Set up paths and logging directories
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, ROOT_DIR)


import random
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import *
from models.common import *
from utils.utils import *
from training.loss import *
from tqdm import tqdm

writer = None

# Dataset path for FIVES512
#path_FIVES = "/mnt/netapp2/Home_FT2/home/usc/ec/rsm/tfg_codebase_cesga/code/dataset/FIVES512"
path_FIVES = os.path.join("dataset", "FIVES") #TODO:
assert os.path.exists(path_FIVES), f"Dataset path does not exist: {path_FIVES}"
path_FIVES512 = os.path.join("dataset", "FIVES512")
#assert os.path.exists(path_FIVES512), f"Dataset path does not exist: {path_FIVES512}"
path_FIVES_1024 = os.path.join("dataset", "FIVES1024")
#assert os.path.exists(path_FIVES_1024), f"Dataset path does not exist: {path_FIVES_1024}"



# --- Optional: Load Restormer Model for Preprocessing ---
try:
    from restormer import Restormer  # Adjust import as needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    restormer_model = Restormer(pretrained=True).to(device)
    restormer_model.eval()
except ImportError:
    restormer_model = None
    print("Restormer model not found. Restormer preprocessing will be disabled.")

def apply_restormer(image: np.ndarray) -> np.ndarray:
    
    #Applies the Restormer model to the input image for deblurring/sharpening.
    #Args:
    #    image (np.ndarray): Input image in BGR format (uint8).
    #Returns:
    #    np.ndarray: The restored image in BGR format (uint8).
    
    # Convert image to float32 and scale to [0,1]
    image_float = image.astype(np.float32) / 255.0
    # Convert BGR to RGB (if your Restormer expects RGB)
    image_rgb = cv2.cvtColor(image_float, cv2.COLOR_BGR2RGB)
    # Convert to tensor and add batch dimension: (1, 3, H, W)
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        restored_tensor = restormer_model(image_tensor)
    
    # Remove batch dimension and convert back to numpy (H, W, C)
    restored_img = restored_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # Convert from [0,1] to [0,255] and to uint8
    restored_img = np.clip(restored_img * 255.0, 0, 255).astype(np.uint8)
    # Convert RGB back to BGR
    restored_img_bgr = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)

    return restored_img_bgr


# Functions
# -----------------------------------------------------------------------------

def set_writer(new_writer):
    global writer
    writer = new_writer

def prepareDatasets():
    all_datasets = {}
    all_datasets['FIVES'] = {
        "train":SegmentationDataset(os.path.join(path_FIVES,"train"), ),
        "test":SegmentationDataset(os.path.join(path_FIVES, "test")),
        "val":SegmentationDataset(os.path.join(path_FIVES, "val"))
    }
    
    return all_datasets
    
def prepare_datasets_from_json(config_path, model_name, augmentation_config, restormer_config):
    
    #print(f" #### restormer config: {restormer_config} ########")

    with open(config_path, 'r') as f:
        config = json.load(f)

    
    if model_name is not None and model_name in config["models"]:
        num_channels = config["models"][model_name].get("ch_in", 3)
        print(f"USING CHANNELS = {num_channels}")
    else:
        print(f"ERROR: Can't get ch_in")
    
    

    datasets = {}
    for name, ds_config in config["datasets"].items():
        paths = ds_config["paths"]
        batch_size = ds_config.get("batch_size", 8)

        

        # Only create SegmentationDataset instances without DataLoader
        datasets[name] = {
            "train": SegmentationDataset(paths["train"],
                augmentation_config=augmentation_config,
                #**ds_config.get("preprocessing", {}),
                start=ds_config.get("preprocessing", {}).get("start", 0),
                end=ds_config.get("preprocessing", {}).get("end", 1),
                #restormer=ds_config.get("preprocessing", {}).get("restormer", False),
                restormer=restormer_config,
                num_channels=num_channels,),
            "val": SegmentationDataset(paths["val"],
                augmentation_config=augmentation_config,
                #**ds_config.get("preprocessing", {}),
                start=ds_config.get("preprocessing", {}).get("start", 0),
                end=ds_config.get("preprocessing", {}).get("end", 1),
                #restormer=ds_config.get("preprocessing", {}).get("restormer", False),
                restormer=restormer_config,
                num_channels=num_channels),
            "test": SegmentationDataset(paths["test"],
                augmentation_config=augmentation_config,
                #**ds_config.get("preprocessing", {}),
                start=ds_config.get("preprocessing", {}).get("start", 0),
                end=ds_config.get("preprocessing", {}).get("end", 1),
                #restormer=ds_config.get("preprocessing", {}).get("restormer", False),
                restormer=restormer_config,
                num_channels=num_channels)
        }

        # Debugging: Test a few samples from the training dataset
        """
        train_dataset = datasets[name]["train"]
        print(f"Testing dataset: {name} (train split)")
        for i in range(20):  # Adjust the range as needed
            try:
                name, image, label = train_dataset[i]
                print(f"Sample {i}: Name={name}, Image Shape={image.shape}, Label Shape={label.shape}")
                print(f"Image Dtype: {image.dtype}, Label Dtype: {label.dtype}")
            except Exception as e:
                print(f"Error at index {i}: {e}")
        """
    return datasets

# Dataset Class
# -----------------------------------------------------------------------------

class SegmentationDataset(Dataset):
    def __init__(self, dataset_paths: Union[str, List[str]], augmentation_config=None, start: float = 0, end: float = 1, restormer: bool = False, num_channels: int = 3) -> None:
    #def __init__(self, dataset_paths: Union[str, List[str]], augmentation_config=None, start: float = 0, end: float = 1, restormer: bool = False) -> None:

        """
        Dataset class for segmentation tasks. Loads images and labels from given paths.

        Args:
            dataset_paths (str or list): Path(s) to dataset directories.
            augmentation_config (dict): Augmentation configuration dictionary.
            start (float): Starting proportion of the dataset to include (default is 0).
            end (float): Ending proportion of the dataset to include (default is 1).
        """
        super().__init__()
        self.ls_item = []
        self.num_channels = num_channels #TODO: add augmentation config in script
        self.augmentation_config = augmentation_config if augmentation_config is not None else {} # Default to empty dict
        self.restormer = restormer

        # Ensure dataset_paths is a list
        if isinstance(dataset_paths, str):
            dataset_paths = [dataset_paths]

        # Load dataset items from paths
        for path_dataset in dataset_paths:
            path_dir_image = os.path.join(path_dataset, "image")
            path_dir_label = os.path.join(path_dataset, "label")
            #print(f"Absolute Dataset path: {os.path.abspath(path_dataset)}")
            #print(f"Image dir: {os.path.abspath(path_dir_image)}")
            #print(f"Label dir: {os.path.abspath(path_dir_label)}")

            # Verify image and label directories exist
            if not os.path.exists(path_dir_image) or not os.path.exists(path_dir_label):
                print(f"Error: Missing image or label directory in {path_dataset}")
                continue

            # Load images and labels with specific file extensions
            valid_extensions = ('.png', '.jpg', '.jpeg')
            ls_image_files = [f for f in os.listdir(path_dir_image) if f.endswith(valid_extensions)]
            ls_label_files = [f for f in os.listdir(path_dir_label) if f.endswith(valid_extensions)]

            #print(f"Found image files: {ls_image_files}")  # Debug print
            #print(f"Found label files: {ls_label_files}")  # Debug print

            # Match images with labels
            for name in ls_image_files:
                if name in ls_label_files:
                    path_image = os.path.join(path_dir_image, name)
                    path_label = os.path.join(path_dir_label, name)
                    assert os.path.exists(path_image), f"Image file does not exist: {path_image}"
                    assert os.path.exists(path_label), f"Label file does not exist: {path_label}"
                    self.ls_item.append({"name": name, "path_image": path_image, "path_label": path_label})

        # Check for valid images
        #print(f"Dataset items before ValueError check: {self.ls_item}")  # Debug print
        if not self.ls_item:
            raise ValueError("Error: No valid images found in dataset.")

        # Shuffle and slice the dataset
        random.seed(0)
        random.shuffle(self.ls_item)
        start_idx, end_idx = int(start * len(self.ls_item)), int(end * len(self.ls_item))
        self.ls_item = self.ls_item[start_idx:end_idx]

    def __len__(self) -> int:
        return len(self.ls_item)

    def __getitem__(self, index: int) -> Tuple[str, np.ndarray, np.ndarray]:
        try:
            index %= len(self)
            item = self.ls_item[index]

            # Load and preprocess image and label
            name = item['name']
            if self.num_channels == 1:
                image = cv2.imread(item['path_image'], cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(item['path_image'], cv2.IMREAD_COLOR)

            label = cv2.imread(item['path_label'], cv2.IMREAD_GRAYSCALE)

            # Debugging: Print information about the loaded data
            #print(f"Index: {index}, Name: {name}")
            #print(f"Image Path: {item['path_image']}, Label Path: {item['path_label']}")
            #print(f"Image Shape: {None if image is None else image.shape}, Label Shape: {None if label is None else label.shape}")

            # Validate that images loaded correctly
            if image is None:
                raise ValueError(f"Image failed to load at path: {item['path_image']}")
            if label is None:
                raise ValueError(f"Label failed to load at path: {item['path_label']}")

            
            # --- Optionally apply Restormer preprocessing ---
            if self.restormer and restormer_model is not None:
                try:
                    image = apply_restormer(image)
                except Exception as e:
                    print(f"Restormer preprocessing failed for {name}: {e}")
            
            


            # Apply data augmentations
            if self.augmentation_config.get("enabled", False):  # Check if augmentation is globally enabled
                image, label = self.augment(image, label)


            # Threshold label to ensure binary values
            _, label = cv2.threshold(label, 127, 1, cv2.THRESH_BINARY)

            # Preprocess image and label
            image, label = self.preprocess_image_label(image, label)

            # Debugging: Final preprocessed shapes
            #print(f"Preprocessed Image Shape: {image.shape}, Preprocessed Label Shape: {label.shape}")

            return name, image, label

        except Exception as e:
            print(f"Error processing index {index}: {e}")
            raise


    # Helper Methods
    # -----------------------------------------------------------------------------
    def augment(self, image, label):
        """Applies a combination of augmentations."""

        if self.augmentation_config.get("otrosfives", False): # Check config
            #print("####### applying otrosfives augment @@@@@@@")
            if random.random() < 0.5:  # 50% chance of applying geometric transformations
                # Aplicar transformaciones geométricas con 50% de probabilidad
                # --- Rotación fija aleatoria ---
                allowed_angles = [-180, -90, 0, 90]
                angle = random.choice(allowed_angles)
                center = (image.shape[1] // 2, image.shape[0] // 2)
                rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
                label = cv2.warpAffine(label, rot_mat, (label.shape[1], label.shape[0]), flags=cv2.INTER_NEAREST)
                
                # --- Flips ---
                # Flip horizontal con p=0.5
                if random.random() < 0.5:
                    image = cv2.flip(image, 1)
                    label = cv2.flip(label, 1)
                # Flip vertical con p=0.5
                if random.random() < 0.5:
                    image = cv2.flip(image, 0)
                    label = cv2.flip(label, 0)
        
        # Se eliminan los bloques de escalado, traslación, elastic, intensidad/color, gamma y noise.

        # --- Geometric Transformations ---
        if self.augmentation_config.get("geometric", False): # Check config
            #print("####### applying geometric augment @@@@@@@")
            if random.random() < 0.5:  # 50% chance of applying geometric transformations
                # Rotation
                angle = random.uniform(-15, 15)
                center = (image.shape[1] // 2, image.shape[0] // 2)
                rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
                label = cv2.warpAffine(label, rot_mat, (label.shape[1], label.shape[0]), flags=cv2.INTER_NEAREST)

                # Flipping
                if random.random() < 0.5:  # 50% chance of horizontal flip
                    image = cv2.flip(image, 1)
                    label = cv2.flip(label, 1)
                if random.random() < 0.5: # 50% chance of vertical flip
                    image = cv2.flip(image, 0)
                    label = cv2.flip(label, 0)

                # Scaling and Translation
                scale = random.uniform(0.9, 1.1)  # Scale between 90% and 110%
                tx = random.uniform(-0.1, 0.1) * image.shape[1]  # Translate by up to 10% of width
                ty = random.uniform(-0.1, 0.1) * image.shape[0]  # Translate by up to 10% of height
                trans_mat = np.float32([[scale, 0, tx], [0, scale, ty]])
                image = cv2.warpAffine(image, trans_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
                label = cv2.warpAffine(label, trans_mat, (label.shape[1], label.shape[0]), flags=cv2.INTER_NEAREST)

        # --- Elastic Deformations (Simplified) ---
        if self.augmentation_config.get("elastic", False): # Check config
            #print("####### applying elastic augment @@@@@@@")
            if random.random() < 0.3: # 30% chance
                alpha = image.shape[1] * random.uniform(0.1,0.5)  #Reduced range
                sigma = image.shape[1] * 0.05 #  sigma to 5% of image width

                #print(f"alpha: {alpha}, sigma: {sigma}")

                dx = cv2.GaussianBlur((np.random.rand(*image.shape[:2]) * 2 - 1), (0, 0), sigma) * alpha
                dy = cv2.GaussianBlur((np.random.rand(*image.shape[:2]) * 2 - 1), (0, 0), sigma) * alpha

                x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
                map_x = np.float32(x + dx)
                map_y = np.float32(y + dy)

                image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                label = cv2.remap(label, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

        # --- Intensity and Color Adjustments (Only on Image) ---
        if self.augmentation_config.get("intensity_and_color", False): # Check config
            #print("####### applying intensity and color augment @@@@@@@")
            if random.random() < 0.5:
                # Brightness and Contrast
                alpha = random.uniform(0.8, 1.2)  # Contrast
                beta = random.uniform(-30, 30)   # Brightness
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

                # Saturation (for color images)
                if image.ndim == 3:
                    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)
                    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # --- Gamma Correction (Only on Image) ---
        if self.augmentation_config.get("gamma", False):# Check config
            #print("####### applying gamma augment @@@@@@@")
            if random.random() < 0.5:
                gamma = random.uniform(0.9, 1.1)
                invGamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                image = cv2.LUT(image, table)

        # --- Noise Addition (Only on Image) ---
        if self.augmentation_config.get("noise", False): # Check config
            #print("####### applying noise augment @@@@@@@")
            if random.random() < 0.3:
                sigma = random.uniform(0, 5)  # Gaussian noise standard deviation
                gauss = np.random.normal(0, sigma, image.size)
                gauss = gauss.reshape(image.shape).astype('uint8')
                image = cv2.add(image, gauss)

        return image, label

    def preprocess_image_label(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Processes image and label for model input, including padding and normalization."""
        # Normalize image to [0, 1] range
        image = image.astype("float32") / 255.0

        # Ensure dimensions are multiples of 32 for model compatibility
        pad_x = (image.shape[1] // 32 + 1) * 32 - image.shape[1]
        pad_y = (image.shape[0] // 32 + 1) * 32 - image.shape[0]
        if (pad_x == 32): pad_x = 0 #Evitar padding innecesario
        if (pad_y == 32): pad_y = 0
        image = cv2.copyMakeBorder(image, pad_y // 2, pad_y // 2, pad_x // 2, pad_x // 2, cv2.BORDER_CONSTANT, value=0)
        label = cv2.copyMakeBorder(label, pad_y // 2, pad_y // 2, pad_x // 2, pad_x // 2, cv2.BORDER_CONSTANT, value=0)

        # Adjust shape for model compatibility
        if image.ndim == 3:  # For RGB images
            image = np.transpose(image, (2, 0, 1))  # From (H, W, C) to (C, H, W)
        else:  # For grayscale images
            image = image[np.newaxis, ...]  # Add channel dimension

        label = label[np.newaxis, ...]  # Add channel dimension for label

        return image, label
