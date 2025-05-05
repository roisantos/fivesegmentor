#!/usr/bin/env python3
"""
Quick test script to generate TensorBoard logs.
This script runs a minimal training session to test that TensorBoard logging is working correctly.
"""
import os
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime as dt

# Set up paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, ROOT_DIR)

from run_benchmark import train_and_evaluate, load_models_from_json
from ds.dataset import prepare_datasets_from_json



def main():
    print("Setting up test TensorBoard logging run...")
    
    # Set up log directory with timestamp
    timestamp = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join(ROOT_DIR, "runs", f"tensorboard_test_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)
    
    # Load a model and dataset
    config_path = os.path.join(ROOT_DIR, "code/config/config.json")
    models = load_models_from_json(config_path)
    
    import run_benchmark
    run_benchmark.models = models      
    
    # Set up a simple augmentation config
    augmentation_config = {
        "enabled": True,
        "geometric": False,
        "elastic": False,
        "intensity_and_color": False,
        "gamma": False,
        "noise": False,
        "otrosfives": True
    }
    
    # Load dataset (limit to a small number of epochs for testing)
    all_datasets = prepare_datasets_from_json(config_path, "SantosNet_PCh", augmentation_config, restormer_config=False)
    
    # Check available models and datasets
    available_models = list(models.keys())
    available_datasets = list(all_datasets.keys())
    
    print(f"Available models: {available_models}")
    print(f"Available datasets: {available_datasets}")
    
    # Choose a model and dataset
    model_name = "SantosNet_PCh" if "SantosNet_PCh" in available_models else available_models[0]
    dataset_name = "FIVES" if "FIVES" in available_datasets else available_datasets[0]
    
    print(f"Testing with model: {model_name}")
    print(f"Testing with dataset: {dataset_name}")
    
    # Run a brief training session (3 epochs)
    dataset = all_datasets[dataset_name]
    
    # Configure args (emulating command line arguments)
    class Args:
        def __init__(self):
            self.epochs = 3
            self.early_stopping = 10
            self.batch_size = 1
            self.num_workers = 4
            self.lr = 1e-4
            self.weight_decay = 0.001
            self.loss = "DirectionalSanLoss"
            self.logging = True
            self.output_prefix = f"test_{timestamp}"
            self.thresh_value = 10
            self.alpha = 0.2
            self.beta = 0.8
            self.gamma = 1.5
            self.entropy_weight = 0.5
            self.direction_weight = 0.7
            self.kernel_size = 5
    
    # Set global args for run_benchmark.py
    global args
    args = Args()
    
    # Run training with logging enabled
    train_and_evaluate(model_name, dataset, logging_enabled=True)
    
    print(f"Test complete. TensorBoard logs written to: {log_dir}")
    print("To view logs, run: tensorboard --logdir=runs/")
    
    # Close SummaryWriter
    writer.close()

if __name__ == "__main__":
    main() 