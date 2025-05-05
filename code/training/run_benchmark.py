import os
import sys
import shutil
import torch
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, default_collate
from torch.utils.tensorboard import SummaryWriter
from torch.nn import DataParallel
import datetime as dt
import json
import argparse


# Clear CUDA cache
torch.cuda.empty_cache()

# Set up root directory and custom module imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, ROOT_DIR)

from ds.dataset import prepare_datasets_from_json, set_writer
from utils.utils import *
#from config.settings_benchmark import models  # Assuming `models` is a dictionary with available models
from models.common import *
from models.frnet import *
from models.roinet import *
from models.extraModels import * 
from models.SantosNet import *
from utils.epoch_stats import EpochActivationStats  # ⬅️ NEW
from utils.tb_logging import TensorboardLogger  # ⬅️ NEW

# Initialize SummaryWriter for TensorBoard
#writer = SummaryWriter()

# ---------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def select_device():
    """Selects CUDA device if available, otherwise CPU."""
    count_card = torch.cuda.device_count()
    id_card = 0
    if count_card > 1:
        print(f"Using {count_card} GPUs")  # Print the number of available GPUs
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def custom_collate(batch):
    """Custom collate function to skip empty batches."""
    # Debugging: Log raw batch contents
    # print(f"Raw batch contents: {batch}")

    # Filter out None values
    batch = [item for item in batch if item is not None]

    # Debugging: Check if batch is empty
    # if len(batch) == 0:
    #     print("##### BATCH IS EMPTY #####")

    return None if len(batch) == 0 else default_collate(batch)

import json

def load_models_from_json(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    models = {}
    # Get the loss function specified in the training section
    #loss_function = config.get("training", {}).get("loss_function", "Dice")

    for name, model_config in config["models"].items():
        # Append the loss function to the model name
        #new_name = f"{name}_{loss_function}"
        new_name = name
        m_type = model_config["type"]
        if "RoiNet_3bottleneck" == model_config["type"]:
            print("##DEBUG: Found RoiNet_3bottleneck")
            models[new_name] = lambda mc=model_config: RoiNet_3bottleneck(
                ch_in=mc.get("ch_in", 3),
                ch_out=mc.get("ch_out", 1),
                ls_mid_ch=mc.get("ls_mid_ch", [32, 64, 128, 128, 64, 32]),
                k_size=mc.get("k_size", 9),
                cls_init_block=eval(mc.get("cls_init_block", "ResidualBlock")),
                cls_conv_block=eval(mc.get("cls_conv_block", "ResidualBlock"))
            )
        elif "RoiNet" == model_config["type"]:
            # Capture model_config in the lambda using a default argument
            print("##DEBUG: Found RoiNet")
            models[new_name] = lambda mc=model_config: RoiNet(
                ch_in=mc.get("ch_in", 3),
                ch_out=mc.get("ch_out", 1),
                ls_mid_ch=mc.get("ls_mid_ch", [32, 64, 128, 128, 64, 32]),
                k_size=mc.get("k_size", 3),
                cls_init_block=eval(mc.get("cls_init_block", "ResidualBlock")),
                cls_conv_block=eval(mc.get("cls_conv_block", "ResidualBlock"))
            )
        elif "FRNet" == model_config["type"]:
            print("##DEBUG: Found FRNet")
            models[new_name] = lambda mc=model_config: FRNet(
                ch_in=mc.get("ch_in", 3),
                ch_out=mc.get("ch_out", 1),
                cls_init_block=eval(mc.get("cls_init_block", "ResidualBlock")),
                cls_conv_block=eval(mc.get("cls_conv_block", "ResidualBlock"))
            )
        elif "RoiNet_1bottleneck" == model_config["type"]:
            print("##DEBUG: Found RoiNet_1bottleneck")
            models[new_name] = lambda mc=model_config: RoiNet_1bottleneck(
                ch_in=mc.get("ch_in", 3),
                ch_out=mc.get("ch_out", 1),
                ls_mid_ch=mc.get("ls_mid_ch", [32, 64, 128, 128, 64, 32]),
                k_size=mc.get("k_size", 3),
                cls_init_block=eval(mc.get("cls_init_block", "ResidualBlock")),
                cls_conv_block=eval(mc.get("cls_conv_block", "ResidualBlock"))
            )
        elif "RoiNetTest1bottleneck" == model_config["type"]:
            print("##DEBUG: Found RoiNetTest1bottleneck")
            models[new_name] = lambda mc=model_config: RoiNetTest1bottleneck(
                ch_in=mc.get("ch_in", 3),
                ch_out=mc.get("ch_out", 1),
                ls_mid_ch=mc.get("ls_mid_ch", [32, 64, 128, 128, 64, 32]),
                k_size=mc.get("k_size", 9),
                cls_init_block=eval(mc.get("cls_init_block", "ResidualBlock")),
                cls_conv_block=eval(mc.get("cls_conv_block", "ResidualBlock"))
            )
        elif "RoiNetTest3bottleneck" == model_config["type"]:
            print("##DEBUG: Found RoiNetTest3bottleneck")
            models[new_name] = lambda mc=model_config: RoiNetTest3bottleneck(
                ch_in=mc.get("ch_in", 3),
                ch_out=mc.get("ch_out", 1),
                ls_mid_ch=mc.get("ls_mid_ch", [32, 64, 128, 128, 64, 32]),
                k_size=mc.get("k_size", 9),
                cls_init_block=eval(mc.get("cls_init_block", "ResidualBlock")),
                cls_conv_block=eval(mc.get("cls_conv_block", "ResidualBlock"))
            )
        elif "RoiNetTest2bottleneck" == model_config["type"]:
            print("##DEBUG: Found RoiNetTest2bottleneck")
            models[new_name] = lambda mc=model_config: RoiNetTest2bottleneck(
                ch_in=mc.get("ch_in", 3),
                ch_out=mc.get("ch_out", 1),
                ls_mid_ch=mc.get("ls_mid_ch", [32, 64, 128, 128, 64, 32]),
                k_size=mc.get("k_size", 9),
                cls_init_block=eval(mc.get("cls_init_block", "ResidualBlock")),
                cls_conv_block=eval(mc.get("cls_conv_block", "ResidualBlock"))
            )
        elif m_type == "RoiNetNoSkip":
            print("##DEBUG: Found RoiNetNoSkip")
            models[new_name] = lambda mc=model_config: RoiNetNoSkip(
                ch_in=mc.get("ch_in", 3),
                ch_out=mc.get("ch_out", 1),
                ls_mid_ch=mc.get("ls_mid_ch", [32, 64, 128, 128, 64, 32]),
                k_size=mc.get("k_size", 9),
                cls_init_block=eval(mc.get("cls_init_block", "ResidualBlock")),
                cls_conv_block=eval(mc.get("cls_conv_block", "ResidualBlock"))
            )
        elif m_type == "RoiNetSumFusion":
            print("##DEBUG: Found RoiNetSumFusion")
            models[new_name] = lambda mc=model_config: RoiNetSumFusion(
                ch_in=mc.get("ch_in", 3),
                ch_out=mc.get("ch_out", 1),
                ls_mid_ch=mc.get("ls_mid_ch", [32, 64, 128, 128, 64, 32]),
                k_size=mc.get("k_size", 9),
                cls_init_block=eval(mc.get("cls_init_block", "ResidualBlock")),
                cls_conv_block=eval(mc.get("cls_conv_block", "ResidualBlock"))
            )
        elif m_type == "RoiNetAttnSkip":
            print("##DEBUG: Found RoiNetAttnSkip")
            models[new_name] = lambda mc=model_config: RoiNetAttnSkip(
                ch_in=mc.get("ch_in", 3),
                ch_out=mc.get("ch_out", 1),
                ls_mid_ch=mc.get("ls_mid_ch", [32, 64, 128, 128, 64, 32]),
                k_size=mc.get("k_size", 9),
                cls_init_block=eval(mc.get("cls_init_block", "ResidualBlock")),
                cls_conv_block=eval(mc.get("cls_conv_block", "ResidualBlock"))
            )
        elif m_type == "RoiNetResSkip":
            print("##DEBUG: Found RoiNetResSkip")
            models[new_name] = lambda mc=model_config: RoiNetResSkip(
                ch_in=mc.get("ch_in", 3),
                ch_out=mc.get("ch_out", 1),
                ls_mid_ch=mc.get("ls_mid_ch", [32, 64, 128, 128, 64, 32]),
                k_size=mc.get("k_size", 9),
                cls_init_block=eval(mc.get("cls_init_block", "ResidualBlock")),
                cls_conv_block=eval(mc.get("cls_conv_block", "ResidualBlock"))
            )
        elif m_type == "RoiNetConcatPlus":
            print("##DEBUG: Found RoiNetConcatPlus")
            models[new_name] = lambda mc=model_config: RoiNetConcatPlus(
                ch_in=mc.get("ch_in", 3),
                ch_out=mc.get("ch_out", 1),
                ls_mid_ch=mc.get("ls_mid_ch", [32, 64, 128, 128, 64, 32]),
                k_size=mc.get("k_size", 9),
                cls_init_block=eval(mc.get("cls_init_block", "ResidualBlock")),
                cls_conv_block=eval(mc.get("cls_conv_block", "ResidualBlock"))
            )
        elif m_type == "RoiNetMultiSkip":
            print("##DEBUG: Found RoiNetMultiSkip")
            models[new_name] = lambda mc=model_config: RoiNetMultiSkip(
                ch_in=mc.get("ch_in", 3),
                ch_out=mc.get("ch_out", 1),
                ls_mid_ch=mc.get("ls_mid_ch", [32, 64, 128, 128, 64, 32]),
                k_size=mc.get("k_size", 9),
                cls_init_block=eval(mc.get("cls_init_block", "ResidualBlock")),
                cls_conv_block=eval(mc.get("cls_conv_block", "ResidualBlock"))
            )
        elif model_config["type"] == "SantosNet_GCh":
            models[new_name] = lambda mc=model_config: SantosNet_GCh(
                ch_in=mc.get("ch_in", 3),
                ch_out=mc.get("ch_out", 1),
                ls_mid_ch=mc.get("ls_mid_ch", [32, 64, 128, 128, 64, 32]),
                k_size=mc.get("k_size", 9),
                cls_init_block=eval(mc.get("cls_init_block", "ResidualBlock")),
                cls_conv_block=eval(mc.get("cls_conv_block", "ResidualBlock"))
            )
        elif model_config["type"] == "SantosNet_PCh":
            models[new_name] = lambda mc=model_config: SantosNet_PCh(
                ch_in=mc.get("ch_in", 3),
                ch_out=mc.get("ch_out", 1),
                ls_mid_ch=mc.get("ls_mid_ch", [32, 64, 128, 128, 64, 32]),
                k_size=mc.get("k_size", 9),
                cls_init_block=eval(mc.get("cls_init_block", "ResidualBlock")),
                cls_conv_block=eval(mc.get("cls_conv_block", "ResidualBlock"))
            )
        elif model_config["type"] == "SantosNet_CPCh":
            models[new_name] = lambda mc=model_config: SantosNet_CPCh(
                ch_in=mc.get("ch_in", 3),
                ch_out=mc.get("ch_out", 1),
                ls_mid_ch=mc.get("ls_mid_ch", [32, 64, 128, 128, 64, 32]),
                k_size=mc.get("k_size", 9),
                cls_init_block=eval(mc.get("cls_init_block", "ResidualBlock")),
                cls_conv_block=eval(mc.get("cls_conv_block", "ResidualBlock")),
                custom_weights=mc.get("custom_weights", [0.1, 0.8, 0.1])
            )



            
    return models

def log_parameters(args, config, dataset_name, model_name, augmentation_config, restormer_config, output_dir):
    """Logs parameters to a file in a human-readable format."""
    log_file_path = os.path.join(output_dir, f"parameters_{dataset_name}_{model_name}_{args.loss}_{augmentation_config['enabled']}_{restormer_config}.log")
    with open(log_file_path, "w") as f:
        f.write("---------- Training Parameters ----------\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n\n")

        f.write("---------- Command-line Arguments ----------\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write("\n")

        f.write("---------- Configuration File ----------\n")
        for section, section_config in config.items():
            f.write(f"[{section}]\n")
            for key, value in section_config.items():
                if isinstance(value, dict):
                    f.write(f"  {key}:\n")
                    for sub_key, sub_value in value.items():
                        f.write(f"    {sub_key}: {sub_value}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")

        f.write("---------- Augmentation Configuration ----------\n")
        for key, value in augmentation_config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        f.write(f"---------- Restormer Configuration ----------\n")
        f.write(f"Restormer Enabled: {restormer_config}\n")
        f.write("\n")

    print(f"Parameters logged to: {log_file_path}")


# ---------------------------------------
# TRAINING AND EVALUATION FUNCTION
# ---------------------------------------

#def train_and_evaluate(model_name, dataset, config, logging_enabled=False):
def train_and_evaluate(model_name, dataset, logging_enabled=False):
    """
    Trains and evaluates a specific model on a dataset.
    Saves the best model and results based on dice score.
    """
    device = select_device()
    model: torch.nn.Module = models[model_name]().to(device)

    # Determine the logging name using the loss function from config.
    # If model_name doesn't already contain the loss function, append it.
    """
    loss_function = config["training"].get("loss_function", "Dice")
    if loss_function not in model_name:
        model_log_name = f"{model_name}_{loss_function}"
    else:
        model_log_name = model_name
    """


    print(f"\nModelo cargado en GPU: {model}")
    print(f"- Parámetros totales: {sum(p.numel() for p in model.parameters())}")
    print(f"- Memoria ocupada por modelo en GPU: {sum(p.element_size() * p.nelement() for p in model.parameters()) / (1024 ** 2):.2f} MB")
    print_gpu_memory_info("Después de cargar el modelo en GPU")

    # Set up TensorBoard logging with different frequencies based on dataset size
    log_frequencies = {
        "step": max(1, len(dataset['train']) // 20),  # Log ~20 times per epoch
        "epoch": 1,                                   # Log every epoch
        "heavy": 5                                    # Log heavy data every 5 epochs
    }
    
    # Initialize our comprehensive logger if logging is enabled
    tb_logger = TensorboardLogger(writer, model, log_freq=log_frequencies) if logging_enabled else None

    # Remove old activation trackers since we now use the comprehensive logger
    # ------------------------------------------------------------------
    # (A) Create the epoch‑level stat tracker **once**
    # ------------------------------------------------------------------
    # act_tracker = EpochActivationStats(
    #     writer,
    #     tag=f"{model_log_name}/activations/conv0_forward"
    # )
    #
    # # Find the appropriate layer - adapt this to your model structure
    # if hasattr(model, 'module'):  # For DataParallel models
    #     if hasattr(model.module, 'dict_module') and 'conv0' in model.module.dict_module:
    #         target_layer = model.module.dict_module['conv0']
    #     else:
    #         # Fallback for other model structures
    #         print("Warning: Using fallback layer for activation tracking")
    #         target_layer = list(model.module.children())[0]
    # else:  # For non-DataParallel models
    #     if hasattr(model, 'dict_module') and 'conv0' in model.dict_module:
    #         target_layer = model.dict_module['conv0']
    #     else:
    #         # Fallback for other model structures
    #         print("Warning: Using fallback layer for activation tracking")
    #         target_layer = list(model.children())[0]
    #         
    # handle = target_layer.register_forward_hook(act_tracker.hook)
    # ------------------------------------------------------------------

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)

    # Get training parameters from config
    """
    training_config = config.get("training", {})
    epochs = training_config.get("epochs", 300)
    thresh_value = training_config.get("early_stopping_threshold", 100)
    batch_size = training_config.get("batch_size", 8)
    num_workers = training_config.get("num_workers", 32)
    learning_rate = training_config.get("learning_rate", 1e-4)
    weight_decay = training_config.get("weight_decay", 0.001)
    """

    #training_config = config.get("training", {})
    epochs = args.epochs
    thresh_value = args.thresh_value
    batch_size = args.batch_size
    num_workers = args.num_workers
    learning_rate = args.lr
    weight_decay = args.weight_decay


    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=learning_rate, weight_decay=weight_decay
    )

    # Select the loss function based on config
    loss_function = args.loss
    if loss_function not in model_name:
        model_log_name = f"{model_name}_{loss_function}"
    else:
        model_log_name = model_name

    if loss_function == "Dice":
        funcLoss = DiceLoss()
        print("\nUSING: Dice")
    elif loss_function == "clDice":
        funcLoss = SoftCLDiceLoss(iter_=50, smooth=1e-12, exclude_background=False)
        print("\nUSING: soft_cldice")
    elif loss_function == "soft_dice_cldice":
        funcLoss = SoftDiceCLDiceLoss(iter_=3, alpha=0.5, smooth=1e-6, exclude_background=False)
        print("USING: soft_dice_cldice")
    elif loss_function == "FocalTversky":
        #funcLoss = FocalTverskyLoss(alpha=0.2, beta=0.8, gamma=0.3, smooth=1e-6)
        #print("USING: FocalTverskyLoss")
        funcLoss = FocalTverskyLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma, smooth=1e-6)
        print(f"USANDO: FocalTverskyLoss con alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}")
    elif loss_function == "SanLoss":
        funcLoss = SanLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma, entropy_weight=args.entropy_weight, smooth=1e-6)
        print(f"USANDO: SanLoss con alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}, entropy_weight={args.entropy_weight}")
    elif loss_function == "DirectionalSanLoss":
        funcLoss = DirectionalSanLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma, 
                                      direction_weight=args.direction_weight, kernel_size=args.kernel_size, smooth=1e-6)
        print(f"USANDO: DirectionalSanLoss con alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}, direction_weight={args.direction_weight}, kernel_size={args.kernel_size}")
    else:
        raise ValueError(f"Loss function '{loss_function}' no reconocida")

    # Configure DataLoaders
    trainLoader = DataLoader(
        dataset=dataset['train'],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=num_workers,
        pin_memory=True
    )
    if batch_size % 2 == 0:
        valLoader = DataLoader(
            dataset=dataset['val'],
            batch_size=batch_size // 2,
            shuffle=True,
            collate_fn=custom_collate,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        valLoader = DataLoader(
            dataset=dataset['val'],
            batch_size=1,
            shuffle=True,
            collate_fn=custom_collate,
            num_workers=num_workers,
            pin_memory=True
        )

    testLoader = DataLoader(dataset=dataset['test'])

    bestResult = {"epoch": -1, "dice": -1}
    ls_best_result = []
    
    # Track total training steps
    global_step = 0

    for epoch in range(epochs):
        torch.cuda.empty_cache()

        # Training
        # Start timing if logging is enabled
        if logging_enabled:
            tb_logger.start_step_timer()
            
        result_train = traverseDataset(
            model=model,
            loader=trainLoader,
            epoch=epoch,
            thresh_value=thresh_value,
            log_section=f"{model_log_name}_{epoch}_train",
            log_writer=None,  # We'll handle logging in our custom logger
            description=f"Train Epoch {epoch}",
            device=device,
            funcLoss=funcLoss,
            optimizer=optimizer,
            tb_logger=tb_logger if logging_enabled else None,
            global_step=global_step
        )
        
        # Update global step counter
        global_step += len(trainLoader)

        # Validation
        result_val = traverseDataset(
            model=model,
            loader=valLoader,
            epoch=epoch,
            thresh_value=thresh_value,
            log_section=f"{model_log_name}_{epoch}_val",
            log_writer=None,  # We'll handle logging in our custom logger
            description=f"Val Epoch {epoch}",
            device=device,
            funcLoss=funcLoss
        )

        # Use our comprehensive logger for epoch-level metrics
        if logging_enabled:
            # Get sample data for visualization (use first batch from validation set)
            sample_data = next(iter(valLoader)) if len(valLoader) > 0 else None
            sample_inputs = sample_data[1].to(device) if sample_data is not None else None
            sample_targets = sample_data[2].to(device) if sample_data is not None else None
            
            # Get model predictions
            sample_outputs = None
            if sample_inputs is not None:
                with torch.no_grad():
                    sample_outputs = model(sample_inputs)
            
            # Log all epoch-level metrics
            tb_logger.log_epoch(
                epoch=epoch,
                train_metrics=result_train,
                val_metrics=result_val,
                sample_inputs=sample_inputs,
                sample_targets=sample_targets,
                sample_outputs=sample_outputs
            )

        # Evaluate dice score and update if it's the best model so far
        dice = result_val['dice']
        print(f"Validation Dice: {dice} for Model: {model_log_name}")

        if dice > bestResult['dice']:
            bestResult.update({"epoch": epoch, "dice": dice})
            ls_best_result.append({"epoch": epoch, "val_dice": dice})
            print("New best dice found, evaluating on test set...")

            result_test = traverseDataset(
                model=model,
                loader=testLoader,
                epoch=epoch,
                thresh_value=thresh_value,
                log_section=None,
                log_writer=None,
                description=f"Test Epoch {epoch}",
                device=device,
                funcLoss=funcLoss
            )
            ls_best_result.append(result_test)
            # Use the new logging name for saving results
            save_best_results(model, ls_best_result, model_log_name)

        # Early stopping
        if epoch - bestResult['epoch'] >= thresh_value:
            print(f"Stopping training: no improvement in last {thresh_value} epochs.")
            break

    # Clean up our logger if used
    if logging_enabled and tb_logger is not None:
        tb_logger.close()

    # Remove the old hook (comment out as we're no longer using it)
    # handle.remove()
    
    # ... rest of the function ...


# ---------------------------------------
# SAVE RESULTS FUNCTION
# ---------------------------------------

"""
def save_best_results(model, results, model_name):
    #Saves the best model and results in a unique timestamped directory.
    root_result = f"run_{model_name}/result_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(root_result, exist_ok=True)

    # Save results and model
    with open(os.path.join(root_result, "best_result.json"), "w") as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), os.path.join(root_result, "model_best.pth"))
    with open(os.path.join(root_result, "finished.flag"), "w") as f:
        f.write("training and testing finished.")
"""
def save_best_results(model, results, model_name):
    root_result = f"{global_output_dir}/{model_name}"
    os.makedirs(root_result, exist_ok=True)

    # Guardar resultados y modelo
    with open(os.path.join(root_result, "best_result.json"), "w") as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), os.path.join(root_result, "model_best.pth"))
    with open(os.path.join(root_result, "finished.flag"), "w") as f:
        f.write("training and testing finished.")



# ---------------------------------------
# MAIN EXECUTION
# ---------------------------------------
if __name__ == "__main__":
    torch.cuda.empty_cache()

    """
    parser = argparse.ArgumentParser(description="Run benchmark for a specific model")
    parser.add_argument("-model", type=str, required=True, help="Name of the model to train")
    args = parser.parse_args()

    # Load configuration from JSON file
    config_path = 'code/config/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

   # Print configuration before starting
    print("Loaded configuration:")
    print(json.dumps(config, indent=4))

    logging_enabled = config["training"].get("logging_enabled", True)

    models = load_models_from_json(config_path)
    all_datasets = prepare_datasets_from_json(config_path)
    #all_datasets = prepareDatasets()

    print(f"Available Models: {[name for name in models]}")
    print(f"Available Datasets: {[name for name in all_datasets]}")
    # Check if the requested model exists
    model_name = args.model
    if model_name not in models:
        print(f"Error: Model '{model_name}' not found in the configuration.")
        sys.exit(1)

    print(f"\n\nTraining Model: {model_name}")
    for dataset_name, dataset in all_datasets.items():
        print(f"Using Dataset: {dataset_name}")
        train_and_evaluate(model_name, dataset, config, logging_enabled=logging_enabled)


    """


     # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Benchmark Training Script")
    parser.add_argument("-model", type=str, required=True, help="Name of the model to train")
    parser.add_argument("-dataset", type=str, required=True, help="Name of the dataset to use")
    parser.add_argument("--config", type=str, default="code/config/config.json",
                        help="Path to the config file (library of models and datasets)")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--early_stopping", type=int, default=100, help="Epochs with no improvement for early stopping")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of DataLoader workers")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay")
    parser.add_argument("--loss", type=str, default="Dice", choices=["Dice", "clDice", "soft_dice_cldice", "FocalTversky", "SanLoss", "DirectionalSanLoss"],
                        help="Loss function to use")
    parser.add_argument("--logging", type=str2bool, default=True, help="Enable TensorBoard logging")
    parser.add_argument("--output_prefix", type=str, default="", help="Prefix for the output folder")
    #parser.add_argument("--augmentation", type=bool, default=False, help="Apply data augmentation")
    parser.add_argument("--thresh_value", type=int, default=100, help="Thresh value")

    parser.add_argument("--augment_geometric", type=str2bool, default=False, help="Enable geometric augmentation")
    parser.add_argument("--augment_elastic", type=str2bool, default=False, help="Enable elastic augmentation")
    parser.add_argument("--augment_intensity", type=str2bool, default=False, help="Enable intensity and color augmentation")
    parser.add_argument("--augment_gamma", type=str2bool, default=False, help="Enable gamma correction augmentation")
    parser.add_argument("--augment_noise", type=str2bool, default=False, help="Enable noise addition augmentation")
    parser.add_argument("--augment_otrosfives", type=str2bool, default=False, help="Enable otrosfives augmentation")

    parser.add_argument("--alpha", type=float, default=0.2,help="Valor de alpha (TP) para Focal Tversky Loss. Sugerido entre 0.2 y 0.3")
    parser.add_argument("--beta", type=float, default=0.8, help="Valor de beta (FN) para Focal Tversky Loss. Sugerido entre 0.7 y 0.8")
    parser.add_argument("--gamma", type=float, default=0.5, help="Valor de gamma (parámetro de foco) para Focal Tversky Loss. Sugerido comenzar en 0.5")
    parser.add_argument("--entropy_weight", type=float, default=0.5, help="Weight for entropy map in SanLoss. Higher values increase focus on uncertain regions.")
    parser.add_argument("--direction_weight", type=float, default=0.7, help="Weight for directional component in DirectionalSanLoss. Higher values increase focus on vessel continuity.")
    parser.add_argument("--kernel_size", type=int, default=5, choices=[3, 5, 7], help="Kernel size for directional calculations in DirectionalSanLoss.")

    parser.add_argument("--restormer", type=str2bool, default=False, help="Enable restormer")


    args = parser.parse_args()

    #Almacenaje de la augmentation
    augmentation_config = {
        "enabled": (args.augment_geometric or args.augment_elastic or args.augment_intensity or args.augment_gamma or args.augment_noise or args.augment_otrosfives),
        "geometric": args.augment_geometric,
        "elastic": args.augment_elastic,
        "intensity_and_color": args.augment_intensity,
        "gamma": args.augment_gamma,
        "noise": args.augment_noise,
        "otrosfives":args.augment_otrosfives
    }
    augmentation_enabled = augmentation_config["enabled"]


    # Load configuration from JSON file
    with open(args.config, 'r') as f:
        config = json.load(f)

    #Load models and dataset TODO: This is not efficient. Only the selected model and dataset should be loaded
    models = load_models_from_json(args.config)
    print(f"Available Models: {[name for name in models]}")
    if args.model not in models:
        print(f"Error: Model '{args.model}' not found in the config library.")
        sys.exit(1)
    model_name = args.model

    datasets_library = config.get("datasets", {})
    if args.dataset not in datasets_library:
        print(f"Error: Dataset '{args.dataset}' not found in the config library.")
        sys.exit(1)
    print(f"Datasets_library: {datasets_library}")



    all_datasets = prepare_datasets_from_json(args.config, args.model, augmentation_config, restormer_config=args.restormer)
    print(f"Available Datasets: {[dataset for dataset in all_datasets]}\n")
    dataset = all_datasets[args.dataset]
    print(f"All datasets: {all_datasets}")
    print(f"Dataset a usar: {dataset}")


    #Setting up the writer for the logs
    timestamp = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    global_output_dir = os.path.join("runs", f"{args.output_prefix}")
    os.makedirs(global_output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=global_output_dir)
    set_writer(writer)

    # Print complete training configuration with usage comments
    print("Configuración de Entrenamiento:")
    print(f"  Model: {args.model}")  
    #   -> Used to select the model from the model library (load_models_from_json)
    print(f"  Dataset: {args.dataset}")  
    #   -> Path to the JSON configuration file, used to load models and datasets
    print(f"  Config File: {args.config}")  
    #   -> Number of epochs; used in the training loop in train_and_evaluate
    print(f"  Epochs: {args.epochs}")  
    #   -> Early stopping value (although it's printed, the early stopping logic uses thresh_value)
    print(f"  Thresh Value: {args.thresh_value}")  
    #   -> Batch size; passed to DataLoaders for training and validation
    print(f"  Batch Size: {args.batch_size}")  
    #   -> Number of threads for loading data; used in the creation of DataLoader
    print(f"  Num Workers: {args.num_workers}")  
    #   -> Selection of the loss function (e.g., Dice, clDice, etc.); used in train_and_evaluate
    print(f"  Learning Rate: {args.lr}")  
    #   -> Dictionary that gathers augmentation flags; passed to prepare_datasets_from_json to configure the dataset
    print(f"  Weight Decay: {args.weight_decay}")  
    #   -> Directory where results, logs, and the best performing model will be saved (used in save_best_results)
    print(f"  Loss Function: {args.loss}")  
    print(f"  Loss Alpha: {args.alpha}")  
    print(f"  Loss Beta: {args.beta}")  
    print(f"  Loss Gamma: {args.gamma}")
    #   -> Flag for enabling TensorBoard logging (used to condition writes in the SummaryWriter)
    print(f"  Logging Enabled: {args.logging}")  
    print(f"  Output Prefix: {args.output_prefix}")  
    print(f"  Augmentation Config: {augmentation_config}")  
    print(f"  Restormer Enabled: {args.restormer}")  
    print(f"  Output Directory: {global_output_dir}")  



    # Log parameters before training
    log_parameters(args, config, args.dataset, model_name, augmentation_config, args.restormer, global_output_dir)

    #Training start
    #train_and_evaluate(model_name, dataset, config, logging_enabled=args.logging)
    train_and_evaluate(model_name, dataset, logging_enabled=args.logging)

