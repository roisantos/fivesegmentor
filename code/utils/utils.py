import os
import sys
import torch
import torch.nn as nn
import json
import cv2
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time

# Set up ROOT_DIR
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, ROOT_DIR)

# Import custom modules
from evaluation.evaluation import *
from training.loss import *

# Globals
activations = defaultdict(list)
gradients = defaultdict(list)

# Function to debug GPU memory
def print_gpu_memory_info(step_desc=""):
    """Imprime el estado de memoria de la GPU."""
    print(f"\n==== Memoria GPU - {step_desc} ====")
    for i in range(torch.cuda.device_count()):
        total_mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)
        allocated_mem = torch.cuda.memory_allocated(i) / (1024 ** 2)
        cached_mem = torch.cuda.memory_reserved(i) / (1024 ** 2)
        print(f"GPU {i} - Total: {total_mem:.1f} MB | Allocated: {allocated_mem:.1f} MB | Cached: {cached_mem:.1f} MB\n")



# Hook functions for saving activations and gradients
def save_activation(name):
    def hook(model, input, output):
        if name not in activations:
            activations[name] = {'mean': [], 'std': [], 'max': [], 'min': []}
        activations[name]['mean'].append(output.mean().item())
        activations[name]['std'].append(output.std().item())
        activations[name]['max'].append(output.max().item())
        activations[name]['min'].append(output.min().item())

        #print(f"\n==== Memory after activation in layer {name} ====")
        #print_gpu_memory_info(f"Activation in {name}")

        if len(activations[name]['mean']) > 10:  # Keep the last 10 measurements
            for key in activations[name]:
                activations[name][key] = activations[name][key][-10:]
    return hook

def save_gradient(name):
    def hook(model, input, output):
        if name not in gradients:
            gradients[name] = {'mean': [], 'std': [], 'max': [], 'min': []}
        gradients[name]['mean'].append(output[0].mean().item())
        gradients[name]['std'].append(output[0].std().item())
        gradients[name]['max'].append(output[0].max().item())
        gradients[name]['min'].append(output[0].min().item())

        #print(f"\n==== Memory after gradient in layer {name} ====")
        #print_gpu_memory_info(f"Gradient in {name}")

        if len(gradients[name]['mean']) > 10:  # Keep the last 10 measurements
            for key in gradients[name]:
                gradients[name][key] = gradients[name][key][-10:]
    return hook



# Function to register hooks on model layers
def register_hooks(model):
    layer_count = 0  # Initialize counter for Conv2d layers
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            layer_name = f"{name}_Conv2d_{layer_count}"
            layer_count += 1
            layer.register_forward_hook(save_activation(f"{layer_name}_forward"))
            layer.register_full_backward_hook(save_gradient(f"{layer_name}_backward"))
            print(f"Registering hooks on layer: {layer_name}")



# Function for logging hook data
def log_hook_data(epoch, activations, gradients, writer, lr, log_section):
    # Log activation statistics
    for layer_name, stats in activations.items():
        for stat_name, values in stats.items():
            avg_stat = torch.mean(torch.tensor(values)).item()
            writer.add_scalar(tag=f"{log_section}/activations/{layer_name}/{stat_name}",
                              scalar_value=avg_stat, global_step=epoch)
    # Log gradient statistics
    for layer_name, stats in gradients.items():
        for stat_name, values in stats.items():
            avg_stat = torch.mean(torch.tensor(values)).item()
            writer.add_scalar(tag=f"{log_section}/gradients/{layer_name}/{stat_name}",
                              scalar_value=avg_stat, global_step=epoch)
    # Log learning rate
    writer.add_scalar(tag=f"{log_section}/learning_rate", scalar_value=lr, global_step=epoch)



# Function for traversing dataset during training and evaluation
def traverseDataset(model: nn.Module, loader: DataLoader, epoch: int,
                   thresh_value: float,
                   log_section: str = None, log_writer: SummaryWriter = None,
                   description: str = None, device=None,
                   funcLoss=None, optimizer=None,
                   tb_logger=None, global_step=0):  # Add new parameters for our logger
    """
    Traverse a dataset for training or evaluation.

    Args:
        model: The model to use
        loader: DataLoader for the dataset
        epoch: Current epoch number
        thresh_value: Threshold value for metrics
        log_section: Section name for logs
        log_writer: SummaryWriter for TensorBoard (legacy)
        description: Description for progress bar
        device: Device to use (CPU/GPU)
        funcLoss: Loss function to use
        optimizer: Optimizer for training (None for evaluation)
        tb_logger: Comprehensive TensorBoard logger (new)
        global_step: Current global step for step-level logging (new)
    """
    
    # Set model to appropriate mode
    is_training = optimizer is not None
    if is_training:
        model.train()
    else:
        model.eval()

    # Check if we're using our comprehensive logger
    using_comprehensive_logger = tb_logger is not None

    # Initialize variables to track metrics
    sum_loss = 0
    sum_dice = 0
    sum_sens = 0
    sum_spec = 0
    sum_iou = 0
    sum_samples = 0

    # Create a progress bar
    tepoch = tqdm(loader, desc=description) if description else loader

    # Traverse the dataset batch by batch
    for i, (name, data, label) in enumerate(tepoch):
        # Skip empty batches (could happen with custom collate)
        if data is None or label is None:
            continue

        # Start step timer for performance logging
        if using_comprehensive_logger and is_training:
            tb_logger.start_step_timer()
        
        # Move data to device
        data, label = data.to(device), label.to(device)

        # Debugging: Print memory info for large batches
        if data.shape[0] > 16:
            print(f"- Tamaño del lote: {data.size()} elementos")
            print(f"- Memoria ocupada por `data`: {data.element_size() * data.nelement() / (1024 ** 2):.2f} MB")
            print_gpu_memory_info(f"{description} - Después de cargar datos del lote {i} en GPU")

        # Forward pass
        if is_training:
            # Training mode
            optimizer.zero_grad()
            out = model(data)
            loss = funcLoss(out, label)
            loss.backward()
            optimizer.step()
            
            # Log step metrics with our comprehensive logger
            if using_comprehensive_logger:
                current_step = global_step + i
                tb_logger.log_step(optimizer, loss.item(), current_step)
        else:
            # Evaluation mode
            with torch.no_grad():
                out = model(data)
                loss = funcLoss(out, label)

        # Calculate metrics
        out_binary = (out > 0.5).float()
        dice_score = calculate_dice(out_binary, label)
        sensitivity = calculate_sensitivity(out_binary, label)
        specificity = calculate_specificity(out_binary, label)
        iou_score = calculate_iou(out_binary, label)

        # Accumulate metrics
        batch_size = data.size(0)
        sum_loss += loss.item() * batch_size
        sum_dice += dice_score * batch_size
        sum_sens += sensitivity * batch_size
        sum_spec += specificity * batch_size
        sum_iou += iou_score * batch_size
        sum_samples += batch_size

        # Update progress bar
        if tepoch is not loader:
            gpu_usage_str = ""
            if torch.cuda.is_available():
                gpu_usage_str = f"{torch.cuda.memory_allocated() / (1024 ** 3):.1f}/{torch.cuda.max_memory_allocated() / (1024 ** 3):.1f} GB"
            
            avg_loss = sum_loss / sum_samples if sum_samples > 0 else 0
            tepoch.set_postfix(avg_loss=f'{avg_loss:.3f}', curr_loss=f'{loss.item():.3f}', gpu_usage=gpu_usage_str)

    # Calculate final metrics
    metrics = {}
    if sum_samples > 0:
        metrics = {
            "loss": sum_loss / sum_samples,
            "dice": sum_dice / sum_samples,
            "sensitivity": sum_sens / sum_samples,
            "specificity": sum_spec / sum_samples,
            "iou": sum_iou / sum_samples
        }

    return metrics

def calculate_dice(pred, target, smooth=1e-6):
    """
    Calculate Dice coefficient between prediction and target.
    
    Args:
        pred: Prediction tensor (binary)
        target: Target tensor (binary)
        smooth: Smoothing constant to avoid division by zero
        
    Returns:
        Dice coefficient value
    """
    # Flatten prediction and target
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    # Calculate Dice coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return dice.item()

def calculate_sensitivity(pred, target, smooth=1e-6):
    """
    Calculate sensitivity (recall) between prediction and target.
    
    Args:
        pred: Prediction tensor (binary)
        target: Target tensor (binary)
        smooth: Smoothing constant to avoid division by zero
        
    Returns:
        Sensitivity value
    """
    # Flatten prediction and target
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # True Positives (TP) and False Negatives (FN)
    tp = (pred_flat * target_flat).sum()
    fn = ((1 - pred_flat) * target_flat).sum()
    
    # Calculate sensitivity
    sensitivity = (tp + smooth) / (tp + fn + smooth)
    
    return sensitivity.item()

def calculate_specificity(pred, target, smooth=1e-6):
    """
    Calculate specificity between prediction and target.
    
    Args:
        pred: Prediction tensor (binary)
        target: Target tensor (binary)
        smooth: Smoothing constant to avoid division by zero
        
    Returns:
        Specificity value
    """
    # Flatten prediction and target
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # True Negatives (TN) and False Positives (FP)
    tn = ((1 - pred_flat) * (1 - target_flat)).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    
    # Calculate specificity
    specificity = (tn + smooth) / (tn + fp + smooth)
    
    return specificity.item()

def calculate_iou(pred, target, smooth=1e-6):
    """
    Calculate Intersection over Union (IoU) between prediction and target.
    Args:
        pred: Prediction tensor (binary)
        target: Target tensor (binary)
        smooth: Smoothing constant to avoid division by zero
    Returns:
        IoU value
    """
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()
