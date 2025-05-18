# utils/tb_logging.py
import torch
import torchvision
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils.epoch_stats import EpochActivationStats

class TensorboardLogger:
    """
    Comprehensive TensorBoard logging utility that handles:
    - Core metrics (loss, dice, etc.)
    - Optimizer stats (LR, etc.)
    - Activation health (mean, std, zero rates)
    - Gradients (norms, update ratios)
    - Weights (norms, histograms)
    - Sample visualizations
    - System performance
    """
    def __init__(self, writer, model, log_freq={"step": 10, "epoch": 1, "heavy": 5}):
        """
        Initialize the logger.
        
        Args:
            writer: SummaryWriter instance
            model: The model to track
            log_freq: Dictionary with logging frequencies:
                - "step": How often to log per-step metrics
                - "epoch": How often to log per-epoch metrics
                - "heavy": How often to log heavy metrics (histograms, images)
        """
        self.writer = writer
        self.model = model
        self.log_freq = log_freq
        self.step_time_start = None
        self.tracked_layers = {}
        self.activation_trackers = {}
        self.zero_rate_trackers = {}
        
        # Register activation trackers for key layers
        self._register_activation_trackers()
        
    def _register_activation_trackers(self):
        """Register hooks on all conv+ReLU layers to track activations and zero/dead neuron rates"""
        if hasattr(self.model, 'dict_module'):
            modules = self.model.dict_module
            # Ensure tracked_layers is initialized
            if not hasattr(self, 'tracked_layers'):
                self.tracked_layers = {}

            for layer_name, layer in modules.items():
                if not hasattr(layer, 'forward'):
                    continue

                # Register activation stats tracker (histograms)
                tracker = EpochActivationStats(self.writer, tag=f"Activations/{layer_name}")
                tracker.register(layer)
                self.activation_trackers[layer_name] = tracker

                # Register zero rate tracker
                zero_tracker = ZeroRateTracker(self.writer, tag=f"ZeroRate/{layer_name}")
                handle_zero = layer.register_forward_hook(zero_tracker.hook)
                self.zero_rate_trackers[layer_name] = (zero_tracker, handle_zero)

                # Register dead neuron rate tracker
                if DeadNeuronRateTracker is not None: # Ensure class is available
                    dead_tracker = DeadNeuronRateTracker(self.writer, tag=f"DeadNeurons/{layer_name}")
                    handle_dead = layer.register_forward_hook(dead_tracker.hook)
                    self.dead_neuron_trackers[layer_name] = (dead_tracker, handle_dead)
                
                # Store reference to tracked layers for _log_weight_stats
                self.tracked_layers[layer_name] = layer
        else:
            print("Model does not have 'dict_module', skipping activation tracker registration.")
    
    def start_step_timer(self):
        """Start timing a training step"""
        self.step_time_start = time.time()
    
    def log_step(self, optimizer, loss, global_step):
        """Log per-step metrics"""
        if global_step % self.log_freq["step"] != 0:
            return
            
        # 1. Step execution time
        if self.step_time_start is not None:
            step_time = (time.time() - self.step_time_start) * 1000  # in ms
            self.writer.add_scalar("System/step_time_ms", step_time, global_step)
            self.step_time_start = None  # Reset timer
        
        # 2. Learning rate
        for i, param_group in enumerate(optimizer.param_groups):
            self.writer.add_scalar(f"LR/group_{i}", param_group['lr'], global_step)
        
        # 3. Current loss
        self.writer.add_scalar("Train/step_loss", loss, global_step)
        
        # 4. Global gradient norm (if we have gradients)
        params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        if params:
            # Compute without clipping, just for logging
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in params]), 2)
            self.writer.add_scalar("Grad/global_norm", total_norm.item(), global_step)
        
        # 5. Log gradient stats for selected layers (every 10 steps)
        if global_step % (self.log_freq["step"] * 10) == 0:
            self._log_gradient_stats(global_step)
    
    def _log_gradient_stats(self, global_step):
        """Log detailed gradient statistics for selected layers"""
        for name, layer in self.tracked_layers.items():
            # Only log if the layer has parameters with gradients
            for param_name, param in layer.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Gradient norm for this parameter
                    grad_norm = param.grad.norm().item()
                    self.writer.add_scalar(f"Grad/{name}/{param_name}_norm", grad_norm, global_step)
                    
                    # Update-to-weight ratio
                    weight_norm = param.data.norm().item()
                    update_ratio = grad_norm / (weight_norm + 1e-12)
                    self.writer.add_scalar(f"Grad/{name}/{param_name}_update_ratio", update_ratio, global_step)
    
    def log_epoch(self, epoch, train_metrics, val_metrics, sample_inputs=None, sample_targets=None, sample_outputs=None):
        """
        Log epoch-level metrics.
        
        Args:
            epoch: The current epoch
            train_metrics: Dictionary of training metrics
            val_metrics: Dictionary of validation metrics
            sample_inputs: Optional batch of input images for visualization
            sample_targets: Optional batch of target masks for visualization
            sample_outputs: Optional batch of model predictions for visualization
        """
        if epoch % self.log_freq["epoch"] != 0:
            return
            
        # 1. Training metrics
        for name, value in train_metrics.items():
            self.writer.add_scalar(f"Train/{name}", value, epoch)
            
        # 2. Validation metrics
        for name, value in val_metrics.items():
            self.writer.add_scalar(f"Val/{name}", value, epoch)
        
        # Log activation stats (histograms) through our trackers
        for layer_name, tracker in self.activation_trackers.items():
            tracker.log_epoch(epoch)
            
        # Log zero rate stats through our trackers
        for layer_name, (tracker, _) in self.zero_rate_trackers.items(): # Unpack tuple
            tracker.log_epoch(epoch)
        
        # Log dead neuron rate stats through our trackers
        if hasattr(self, 'dead_neuron_trackers'):
            for layer_name, (tracker, _) in self.dead_neuron_trackers.items(): # Unpack tuple
                tracker.log_epoch(epoch)
        
        # Heavy logging (less frequent)
        if epoch % self.log_freq["heavy"] == 0:
            # 3. Weight histograms and norms
            self._log_weight_stats(epoch)
            # Activation histograms are now logged above via EpochActivationStats
            # 5. Sample visualizations
            if all(x is not None for x in [sample_inputs, sample_targets, sample_outputs]):
                self._log_sample_visualizations(epoch, sample_inputs, sample_targets, sample_outputs)
    
    def _log_weight_stats(self, epoch):
        """Log weight statistics and histograms"""
        # Ensure tracked_layers exists and is populated
        if hasattr(self, 'tracked_layers'):
            for name, layer in self.tracked_layers.items():
                for param_name, param in layer.named_parameters():
                    if param.requires_grad: # Check if param has data and requires_grad
                        if param.data is not None:
                             # Weight norm
                            weight_norm = param.data.norm().item()
                            self.writer.add_scalar(f"Weights/{name}/{param_name}_norm", weight_norm, epoch)
                            
                            # Weight histogram
                            self.writer.add_histogram(f"Weights/{name}/{param_name}_hist", param.data.cpu(), epoch) # Ensure CPU
                        if param.grad is not None:
                            # Gradient norm for this parameter
                            grad_norm = param.grad.norm().item()
                            self.writer.add_scalar(f"GradStats/{name}/{param_name}_norm", grad_norm, epoch)
                            self.writer.add_histogram(f"GradStats/{name}/{param_name}_hist", param.grad.cpu(), epoch) # Ensure CPU

                            # Update-to-weight ratio (only if weight_norm is available and > 0)
                            if param.data is not None:
                                weight_norm_val = param.data.norm().item()
                                if weight_norm_val > 1e-12:
                                    update_ratio = grad_norm / weight_norm_val
                                    self.writer.add_scalar(f"GradStats/{name}/{param_name}_update_ratio", update_ratio, epoch)
        else:
            print("'_tracked_layers' not found in TensorboardLogger. Skipping weight stats.")
    
    def _log_sample_visualizations(self, epoch, inputs, targets, outputs):
        """Log sample visualizations"""
        # Take up to 4 samples from the batch
        n_samples = min(4, inputs.shape[0])
        
        # Process each sample
        for i in range(n_samples):
            # Get sample data
            input_img = inputs[i].detach().cpu()
            target_mask = targets[i].detach().cpu()
            output_mask = outputs[i].detach().cpu()
            
            # Normalize to [0,1] for visualization
            if input_img.shape[0] == 3:  # RGB image
                input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min() + 1e-8)
            elif input_img.shape[0] == 1:  # Grayscale image
                input_img = input_img.repeat(3, 1, 1)  # Convert to RGB
                input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min() + 1e-8)
            
            # Create a 3-channel visualization for the target and output masks
            target_vis = torch.zeros((3, target_mask.shape[1], target_mask.shape[2]))
            target_vis[0] = target_mask[0]  # Put in red channel
            
            output_vis = torch.zeros((3, output_mask.shape[1], output_mask.shape[2]))
            output_vis[1] = output_mask[0]  # Put in green channel
            
            # Create composite visualization (input with masks overlaid)
            alpha = 0.6
            composite = input_img.clone()
            mask = (target_mask[0] > 0.5) | (output_mask[0] > 0.5)
            composite[:, mask] = alpha * composite[:, mask] + (1-alpha) * torch.stack([
                target_mask[0, mask],  # Red: target
                output_mask[0, mask],  # Green: prediction
                torch.zeros_like(target_mask[0, mask])  # Blue: unused
            ])
            
            # Log individual images
            self.writer.add_image(f"Images/sample_{i}/input", input_img, epoch)
            self.writer.add_image(f"Images/sample_{i}/target", target_vis, epoch)
            self.writer.add_image(f"Images/sample_{i}/output", output_vis, epoch)
            self.writer.add_image(f"Images/sample_{i}/composite", composite, epoch)
            
    def close(self):
        """Clean up resources"""
        for _layer_name, tracker in self.activation_trackers.items():
            tracker.remove()
        for _layer_name, (_tracker, handle) in self.zero_rate_trackers.items():
            handle.remove()
        if hasattr(self, 'dead_neuron_trackers'):
            for _layer_name, (_tracker, handle) in self.dead_neuron_trackers.items():
                handle.remove()
        
        self.activation_trackers.clear()
        self.zero_rate_trackers.clear()
        if hasattr(self, 'dead_neuron_trackers'):
            self.dead_neuron_trackers.clear()

        if self.writer:
            self.writer.close()


class ZeroRateTracker:
    """Tracks the fraction of zeros in activations (ReLU dead neurons)"""
    def __init__(self, writer, tag):
        self.writer = writer
        self.tag = tag
        self.reset()
        
    def reset(self):
        self.total_zeros = 0
        self.total_elements = 0
        
    @torch.no_grad()
    def hook(self, _module, _inp, out):
        zeros = (out == 0).float().sum().item()
        elements = out.numel()
        self.total_zeros += zeros
        self.total_elements += elements
        
    def log_epoch(self, epoch):
        if self.total_elements > 0:
            zero_rate = self.total_zeros / self.total_elements
            self.writer.add_scalar(f"{self.tag}", zero_rate, epoch)
        self.reset()


class DeadNeuronRateTracker:
    """Tracks the fraction of channels that are always zero (dead neurons) in activations"""
    def __init__(self, writer, tag):
        self.writer = writer
        self.tag = tag
        self.reset()
    def reset(self):
        self.channel_zero_counts = None
        self.total_batches = 0
    @torch.no_grad()
    def hook(self, _module, _inp, out):
        # out: (B, C, H, W)
        if out.dim() < 2:
            return
        zeros_per_channel = (out == 0).float().view(out.size(0), out.size(1), -1).sum(-1)
        total_per_channel = out[0,0].numel()
        dead = (zeros_per_channel == total_per_channel).float().sum(-1)  # per batch
        if self.channel_zero_counts is None:
            self.channel_zero_counts = dead.clone()
        else:
            self.channel_zero_counts += dead
        self.total_batches += out.size(0)
    def log_epoch(self, epoch):
        if self.channel_zero_counts is not None and self.total_batches > 0:
            dead_rate = self.channel_zero_counts.sum().item() / (self.total_batches * self.channel_zero_counts.numel())
            self.writer.add_scalar(f"{self.tag}", dead_rate, epoch)
        self.reset() 

class EpochActivationStats:
    def __init__(self, writer, tag):
        self.writer = writer
        self.tag = tag
        self.activations = []
        self.hook_handle = None

    @torch.no_grad()
    def _save_activations_hook(self, module, input, output):
        # Detach and move to CPU to save memory, especially if accumulating over an epoch
        self.activations.append(output.detach().cpu()) # Ensure it's on CPU

    def log_epoch(self, epoch):
        if len(self.activations) > 0:
            all_activations = torch.cat(self.activations) # Already on CPU
            # Log mean and std if desired
            # self.writer.add_scalar(f"{self.tag}/mean", all_activations.mean(), epoch)
            # self.writer.add_scalar(f"{self.tag}/std", all_activations.std(), epoch)
            self.writer.add_histogram(f"{self.tag}/hist", all_activations, epoch)
        self.activations = [] # Clear for next epoch

    def register(self, module):
        if self.hook_handle is not None: # Avoid double registration
            self.remove()
        self.hook_handle = module.register_forward_hook(self._save_activations_hook)

    def remove(self):
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None 