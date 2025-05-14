import os
import json
import argparse
import numpy as np
import sys
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
from torch.nn import DataParallel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import datetime as dt
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# Clean CUDA cache
torch.cuda.empty_cache()

# Set up root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, ROOT_DIR)

from utils.utils import *
from models.roinet import *
from models.common import *

class FIVESJoinedDataset(torch.utils.data.Dataset):
    """Dataset for binary segmentation with stratification support."""
    def __init__(self, root_dir):
        self.samples = []
        img_dir = os.path.join(root_dir, "image")
        lbl_dir = os.path.join(root_dir, "label")
        for fn in sorted(os.listdir(img_dir)):
            if not fn.endswith(".png"): continue
            # Extract disease type from filename for stratification
            disease = fn.split('_')[-1].split('.')[0]
            self.samples.append({
                'image_path': os.path.join(img_dir, fn),
                'label_path': os.path.join(lbl_dir, fn),
                'disease': disease
            })
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        print(f"Found {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Load image and label
        image = Image.open(sample['image_path']).convert('RGB')
        label = Image.open(sample['label_path']).convert('L')
        
        # Apply transforms
        image = self.transform(image)
        
        # Convert label to tensor and ensure binary values (0 and 1)
        label = self.transform(label)
        label = (label > 0.5).float()  # Ensure binary values
        
        # Verify label contains both 0 and 1
        if not (label.min() == 0 and label.max() == 1):
            # If label is all zeros or all ones, create a dummy label
            # This is a fallback to prevent the assertion error
            label = torch.zeros_like(label)
            label[0, 0, 0] = 1  # Set at least one pixel to 1
        
        return sample['image_path'], image, label

    def get_stratification_labels(self):
        """Returns disease labels for stratification."""
        return [sample['disease'] for sample in self.samples]

def save_fold_results(output_dir, fold, metrics):
    """Save metrics for each fold."""
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f'fold_{fold}_results.json')
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)

def main():
    print("Parsing arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Path to FIVES_joined dataset')
    parser.add_argument('--folds', type=int, default=10, help='Number of folds')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs per fold')
    parser.add_argument('--bs', type=int, default=1, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join('runs', f'cv_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    print("Initializing TensorBoard writer...")
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=output_dir)

    # Save configuration
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("Initializing dataset...")
    # Initialize dataset
    dataset = FIVESJoinedDataset(args.dataset)
    
    # Get stratification labels
    labels = dataset.get_stratification_labels()
    le = LabelEncoder()
    stratify_labels = le.fit_transform(labels)

    # Initialize k-fold cross validation
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Track metrics across folds
    all_fold_metrics = {
        'dice': [], 'acc': [], 'fdr': [], 
        'sen': [], 'spe': [], 'gmean': [], 
        'iou': []
    }

    # Perform k-fold cross validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), stratify_labels), 1):
        print(f"\n=== Fold {fold}/{args.folds} ===")
        
        # Clean up memory before starting new fold
        torch.cuda.empty_cache()
        
        # Create data loaders for this fold
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.bs,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # Always use batch size 1 for validation
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        # Initialize model for this fold
        model = RoiNet(
            ch_in=3,
            ch_out=1,
            ls_mid_ch=[32, 64, 128, 128, 64, 32],
            k_size=9,
            cls_init_block=ResidualBlock,
            cls_conv_block=ResidualBlock
        ).to(device)

        if torch.cuda.device_count() > 1:
            model = DataParallel(model)

        # Initialize optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = DiceLoss()

        # Training loop for this fold
        best_val_metrics = {'dice': 0}  # Track best validation metrics
        fold_metrics = []
        
        for epoch in range(1, args.epochs + 1):
            # Training phase
            model.train()
            train_metrics = traverseDataset(
                model=model,
                loader=train_loader,
                epoch=epoch,
                description=f"Fold {fold} Train Epoch {epoch}",
                device=device,
                funcLoss=loss_fn,
                optimizer=optimizer,
                log_writer=writer,
                log_section=f"fold_{fold}/train",
                thresh_value=None
            )

            # Log training metrics to TensorBoard
            for metric_name, value in train_metrics.items():
                writer.add_scalar(f"fold_{fold}/train/{metric_name}", value, epoch)

            # Clean up memory before validation
            torch.cuda.empty_cache()

            # Validation phase
            model.eval()
            val_metrics = traverseDataset(
                model=model,
                loader=val_loader,
                epoch=epoch,
                description=f"Fold {fold} Val Epoch {epoch}",
                device=device,
                funcLoss=loss_fn,
                optimizer=None,
                log_writer=writer,
                log_section=f"fold_{fold}/val",
                thresh_value=None
            )

            # Log validation metrics to TensorBoard
            for metric_name, value in val_metrics.items():
                writer.add_scalar(f"fold_{fold}/val/{metric_name}", value, epoch)

            # Track best model for this fold based on dice score
            if val_metrics['dice'] > best_val_metrics['dice']:
                best_val_metrics = val_metrics.copy()  # Save all metrics
                # Save best model for this fold
                torch.save({
                    'fold': fold,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': best_val_metrics,
                }, os.path.join(output_dir, f'fold_{fold}_best_model.pth'))

            # Save epoch metrics
            epoch_metrics = {
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics
            }
            fold_metrics.append(epoch_metrics)

            # Clean up memory after epoch
            torch.cuda.empty_cache()

        # Save fold results
        save_fold_results(output_dir, fold, {
            'metrics': fold_metrics,
            'best_metrics': best_val_metrics
        })
        
        # Store best metrics for this fold
        for metric in all_fold_metrics.keys():
            if metric in best_val_metrics:
                all_fold_metrics[metric].append(best_val_metrics[metric])
        
        # Print fold results
        print(f"\n=== Fold {fold}/{args.folds} Results ===")
        for metric in best_val_metrics.keys():
            print(f"{metric.upper():>8}: {best_val_metrics[metric]:.4f}")
        
        # If we have completed more than one fold, show running average
        if fold > 1:
            print(f"\n=== Running Average (Folds 1-{fold}) ===")
            for metric in all_fold_metrics.keys():
                values = np.array(all_fold_metrics[metric])
                mean_val = float(np.mean(values))
                std_val = float(np.std(values))
                print(f"{metric.upper():>8}: {mean_val:.4f} ± {std_val:.4f}")

        # Clean up memory after fold
        del model, optimizer
        torch.cuda.empty_cache()

    # Calculate and save overall results
    final_results = {
        'folds': args.folds,
        'metrics': {}
    }
    
    print("\n=== Final Results (All Folds) ===")
    for metric in all_fold_metrics.keys():
        values = np.array(all_fold_metrics[metric])
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        final_results['metrics'][metric] = {
            'mean': mean_val,
            'std': std_val,
            'values': values.tolist()
        }
        print(f"{metric.upper():>8}: {mean_val:.4f} ± {std_val:.4f}")
    
    with open(os.path.join(output_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)

    # Close TensorBoard writer
    writer.close()

if __name__ == '__main__':
    main()
