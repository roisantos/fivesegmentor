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



# Dataset traversal function for training and evaluation
def traverseDataset(model: nn.Module, loader: DataLoader, epoch: int,
                    description, device, funcLoss, 
                    log_writer: SummaryWriter, log_section, optimizer=None, scheduler=None,
                    show_result=False, thresh_value=None):
    is_training = (optimizer is not None)
    model.train(is_training)
    total_loss = 0
    ls_eval_result = []
    start_time = time.time()
    
    print_gpu_memory_info("Inicio de época")
    
    with tqdm(loader, unit="batch", mininterval=1.0) as tepoch:
        for i, (name, data, label) in enumerate(tepoch):
            tepoch.set_description(description)
            data, label = data.to(device), label.to(device)

            if is_training:
                optimizer.zero_grad()
                out = model(data)
                loss = sum(funcLoss(x, label) for x in (out if isinstance(out, list) else [out]))
                loss.backward()
                
                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                if scheduler: scheduler.step()
                
                # Clear some memory
                del out
                torch.cuda.empty_cache()
            else:
                with torch.no_grad():
                    out = model(data)
                    loss = funcLoss(out, label)
                    for index in range(loader.batch_size):
                        pred, gt = out[index][0].cpu().numpy(), label[index][0].cpu().numpy()
                        eval_result = calc_result(pred, gt, thresh_value)
                        ls_eval_result.append(eval_result)
                    # Clear some memory
                    del out
                    torch.cuda.empty_cache()

            avg_loss = (total_loss + loss.item()) / (i + 1)
            total_loss += loss.item()
            
            # Collect GPU memory usage per device
            gpu_usage = {f"GPU {i}": torch.cuda.memory_allocated(i) / (1024 ** 2) for i in range(torch.cuda.device_count())}
            gpu_usage_str = " | ".join([f"{k}: {v:.1f}MB" for k, v in gpu_usage.items()])

            tepoch.set_postfix(avg_loss=f'{avg_loss:.3f}', curr_loss=f'{loss.item():.3f}', gpu_usage=gpu_usage_str)
            
            # Clear some memory
            del data, label, loss
            torch.cuda.empty_cache()
    
    avg_ms = (time.time() - start_time) * 1000 / len(loader) / loader.batch_size
    result = avg_result(ls_eval_result)
    result.update({
        'avg_ms': avg_ms,
        'num_params': sum(p.numel() for p in model.parameters())
    })
    
    # Log hook data if training and hooks are registered
    if log_writer and (activations or gradients) and (optimizer is not None):
        lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
        log_hook_data(epoch, activations, gradients, log_writer, lr, log_section)
        activations.clear()
        gradients.clear()

    return result
