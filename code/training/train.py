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

# Limpia la caché de CUDA
torch.cuda.empty_cache()

# Establecer directorio raíz y rutas para importar módulos personalizados
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, ROOT_DIR)

from ds.dataset import prepare_datasets_from_json, set_writer
from utils.utils import *
from models.common import *
from models.frnet import *
from models.roinet import *
from models.extraModels import *
from models.SantosNet import *

# Importamos las clases de pérdidas desde loss.py (el nuevo módulo de pérdida)
#from loss import CompositeLoss, DiceLoss, SoftCLDiceLoss, SoftDiceCLDiceLoss, FocalTverskyLoss

# ---------------------------------------
# NUEVA FUNCIÓN: Parsear la cadena de composite loss
# ---------------------------------------
def build_composite_loss(loss_string):
    """
    Parsea la cadena que define la composite loss y devuelve
    una lista de tuplas (loss_fn, weight) para instanciar CompositeLoss.
    
    Formato esperado:
       "LossName:param1=value1,param2=value2,weight=VAL;AnotherLoss:...,...,weight=VAL"
       
    Ejemplo:
       "Dice:weight=0.5000;FocalTversky:alpha=0.2,beta=0.8,gamma=0.5,weight=0.5000"
    """
    components = []
    for comp in loss_string.split(";"):
        comp = comp.strip()
        if not comp:
            continue
        # Separar el nombre de la pérdida y el string de parámetros
        parts = comp.split(":", 1)
        loss_name = parts[0]
        params_str = parts[1] if len(parts) > 1 else ""
        params = {}
        if params_str:
            # Los parámetros vienen separados por comas
            for param in params_str.split(","):
                param = param.strip()
                if param:
                    if "=" in param:
                        key, value = param.split("=", 1)
                        params[key.strip()] = value.strip()
                    else:
                        # Si es una bandera sin valor
                        params[param] = True
        if "weight" not in params:
            raise ValueError("Cada componente de pérdida debe incluir un parámetro 'weight'.")
        weight = float(params.pop("weight"))
        
        # Instanciar la función de pérdida según loss_name
        if loss_name == "Dice":
            loss_fn = DiceLoss()

        elif loss_name == "SoftCLDiceLoss":
            iter_ = int(params.get("iter", 20))
            smooth = float(params.get("smooth", 1e-12))
            exclude_background = params.get("exclude_background", "False") == "True"
            loss_fn = SoftCLDiceLoss(iter_=iter_, smooth=smooth, exclude_background=exclude_background)

        elif loss_name == "soft_dice_cldice":
            iter_ = int(params.get("iter", 3))
            alpha = float(params.get("alpha", 0.5))
            smooth = float(params.get("smooth", 1.0))
            exclude_background = params.get("exclude_background", "False") == "True"
            loss_fn = SoftDiceCLDiceLoss(iter_=iter_, alpha=alpha, smooth=smooth, exclude_background=exclude_background)

        elif loss_name == "FocalTversky":
            alpha = float(params.get("alpha", 0.2))
            beta = float(params.get("beta", 0.8))
            gamma = float(params.get("gamma", 0.5))
            smooth = float(params.get("smooth", 1e-6))
            loss_fn = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma, smooth=smooth)

        elif loss_name == "SoftCLDiceLossStrict":
            penalty_power = float(params.get("penalty_power", 5.))
            smooth = float(params.get("smooth", 1e-6))
            SoftCLDiceLossStrict(iter_=25, smooth=1e-6, penalty_power=penalty_power, exclude_background=False)
            
        elif loss_name in ("DistanceWeightedBCE", "dw_bce"):
            loss_fn = DistanceWeightedBCELoss(
                sigma=float(params.get("sigma", 5.0))
            )

        elif loss_name in ("VesselHaloLoss", "halo"):
            loss_fn = VesselHaloLoss(
                band_width=int(params.get("band_width", 5)),
                alpha=float(params.get("alpha", 1.0))
            )

        elif loss_name in ("HaloCLDiceLoss", "halo_cldice"):
            loss_fn = HaloCLDiceLoss(
                band_width=int(params.get("band_width", 5)),
                alpha=float(params.get("alpha", 0.5)),
                beta=float(params.get("beta", 0.5)),
                iter=int(params.get("iter", 3))
            )
        else:
            raise ValueError(f"Función de pérdida '{loss_name}' no reconocida en composite loss.")
        
        components.append((loss_fn, weight))
    return components

# ---------------------------------------
# Funciones Auxiliares
# ---------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Se esperaba un valor booleano.')

def select_device():
    """Selecciona dispositivo CUDA si está disponible, de lo contrario CPU."""
    count_card = torch.cuda.device_count()
    if count_card > 1:
        print(f"Usando {count_card} GPUs")
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def custom_collate(batch):
    """Función custom para el collate que filtra elementos nulos."""
    batch = [item for item in batch if item is not None]
    return None if len(batch) == 0 else default_collate(batch)

def load_models_from_json(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    models = {}
    for name, model_config in config["models"].items():
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
        elif model_config["type"] == "SantosNet_GCh_lite":
            print("##DEBUG: Found SantosNet_GCh_lite")
            models[new_name] = lambda mc=model_config: SantosNet_GCh_lite(
                ch_out=mc.get("ch_out", 1),
                ls_mid_ch=mc.get("ls_mid_ch", [16, 32, 64, 64, 32, 16]),
                k_size=mc.get("k_size", 3)
            )
        elif model_config["type"] == "SantosNet_GCh_lite_v2":
            print("##DEBUG: Found SantosNet_GCh_lite_v2")
            models[new_name] = lambda mc=model_config: SantosNet_GCh_lite_v2(
                ch_out=mc.get("ch_out", 1),
                ls_mid_ch=mc.get("ls_mid_ch", [16, 32, 64, 64, 32, 16]),
                k_size=mc.get("k_size", 3)
            )
    return models

def log_parameters(args, config, dataset_name, model_name, augmentation_config, restormer_config, output_dir):
    """Guarda los parámetros de entrenamiento en un archivo legible."""
    log_file_path = os.path.join(output_dir, f"parameters_{dataset_name}_{model_name}_{augmentation_config['enabled']}_{restormer_config}.log")
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
# Función de entrenamiento y evaluación
# ---------------------------------------
def train_and_evaluate(model_name, dataset, logging_enabled=False):
    """
    Entrena y evalúa un modelo en un dataset dado, guardando el mejor modelo según la métrica dice.
    """
    device = select_device()
    model: torch.nn.Module = models[model_name]().to(device)

    print(f"\nModelo cargado en GPU: {model}")
    print(f"- Parámetros totales: {sum(p.numel() for p in model.parameters())}")
    print(f"- Memoria ocupada por modelo en GPU: {sum(p.element_size() * p.nelement() for p in model.parameters()) / (1024 ** 2):.2f} MB")
    print_gpu_memory_info("Después de cargar el modelo en GPU")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)

    # Parámetros de entrenamiento (usando los valores provenientes de args)
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

    # Seleccionar la función de pérdida:
    # Si se especifica un composite loss, se parsea y se crea CompositeLoss
    if hasattr(args, "composite_loss_components") and args.composite_loss_components:
        components = build_composite_loss(args.composite_loss_components)
        funcLoss = CompositeLoss(components)
        print(f"\nUSANDO: Composite Loss con componentes: {args.composite_loss_components}")
        model_log_name = f"{model_name}_Composite"
    else:
        raise ValueError(f"Loss function no encontrada")
    
    diceLoss = DiceLoss()

    # Configurar DataLoaders
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

    for epoch in range(epochs):
        torch.cuda.empty_cache()

        # Entrenamiento
        result_train = traverseDataset(
            model=model,
            loader=trainLoader,
            epoch=epoch,
            thresh_value=thresh_value,
            log_section=f"{model_log_name}_{epoch}_train",
            log_writer=writer if (epoch % 1 == 0 and logging_enabled) else None,
            description=f"Train Epoch {epoch}",
            device=device,
            funcLoss=funcLoss,
            optimizer=optimizer
        )

        for key, value in result_train.items():
            writer.add_scalar(f"{model_log_name}/{key}_train", value, epoch)

        # Validación
        result_val = traverseDataset(
            model=model,
            loader=valLoader,
            epoch=epoch,
            thresh_value=thresh_value,
            log_section=f"{model_log_name}_{epoch}_val",
            log_writer=writer if (epoch % 1 == 0 and logging_enabled) else None,
            description=f"Val Epoch {epoch}",
            device=device,
            funcLoss=diceLoss
        )

        for key, value in result_val.items():
            writer.add_scalar(f"{model_log_name}/{key}_val", value, epoch)

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
                funcLoss=diceLoss
            )
            ls_best_result.append(result_test)
            save_best_results(model, ls_best_result, model_log_name)

        if epoch - bestResult['epoch'] >= thresh_value:
            print(f"Stopping training: no improvement in last {thresh_value} epochs.")
            break

# ---------------------------------------
# Función para guardar los mejores resultados
# ---------------------------------------
def save_best_results(model, results, model_name):
    root_result = os.path.join(global_output_dir, model_name)
    os.makedirs(root_result, exist_ok=True)
    with open(os.path.join(root_result, "best_result.json"), "w") as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), os.path.join(root_result, "model_best.pth"))
    with open(os.path.join(root_result, "finished.flag"), "w") as f:
        f.write("training and testing finished.")

# ---------------------------------------
# Bloque principal
# ---------------------------------------
if __name__ == "__main__":
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description="Benchmark Training Script")
    parser.add_argument("-model", type=str, required=True, help="Nombre del modelo a entrenar")
    parser.add_argument("-dataset", type=str, required=True, help="Nombre del dataset a usar")
    parser.add_argument("--config", type=str, default="code/config/config.json",
                        help="Ruta del archivo de configuración (librería de modelos y datasets)")
    parser.add_argument("--epochs", type=int, default=300, help="Número de épocas de entrenamiento")
    parser.add_argument("--early_stopping", type=int, default=100, help="Épocas sin mejora para early stopping")
    parser.add_argument("--batch_size", type=int, default=8, help="Tamaño del batch de entrenamiento")
    parser.add_argument("--num_workers", type=int, default=32, help="Número de trabajadores para DataLoader")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay")
    #parser.add_argument("--loss", type=str, default="Dice", choices=["Dice", "clDice", "soft_dice_cldice", "FocalTversky"], help="Función de pérdida a usar")
    parser.add_argument("--logging", type=str2bool, default=True, help="Habilitar logs en TensorBoard")
    parser.add_argument("--output_prefix", type=str, default="", help="Prefijo para la carpeta de salida")
    parser.add_argument("--thresh_value", type=int, default=100, help="Valor threshold para early stopping")
    parser.add_argument("--augment_geometric", type=str2bool, default=False, help="Habilitar augmentación geométrica")
    parser.add_argument("--augment_elastic", type=str2bool, default=False, help="Habilitar augmentación elástica")
    parser.add_argument("--augment_intensity", type=str2bool, default=False, help="Habilitar augmentación de intensidad y color")
    parser.add_argument("--augment_gamma", type=str2bool, default=False, help="Habilitar augmentación de corrección gamma")
    parser.add_argument("--augment_noise", type=str2bool, default=False, help="Habilitar augmentación de ruido")
    parser.add_argument("--augment_otrosfives", type=str2bool, default=False, help="Habilitar augmentación otrosfives")
    parser.add_argument("--alpha", type=float, default=0.2, help="Valor alpha para Focal Tversky Loss")
    parser.add_argument("--beta", type=float, default=0.8, help="Valor beta para Focal Tversky Loss")
    parser.add_argument("--gamma", type=float, default=0.5, help="Valor gamma para Focal Tversky Loss")
    parser.add_argument("--restormer", type=str2bool, default=False, help="Habilitar restormer")
    # Nuevo argumento para composite loss
    parser.add_argument("--composite_loss_components", type=str, default=None,
                        help="Cadena que define los componentes de la composite loss")

    args = parser.parse_args()

    # Configuración de augmentación
    augmentation_config = {
        "enabled": (args.augment_geometric or args.augment_elastic or args.augment_intensity or args.augment_gamma or args.augment_noise or args.augment_otrosfives),
        "geometric": args.augment_geometric,
        "elastic": args.augment_elastic,
        "intensity_and_color": args.augment_intensity,
        "gamma": args.augment_gamma,
        "noise": args.augment_noise,
        "otrosfives": args.augment_otrosfives
    }

    # Cargar configuración desde archivo JSON
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Cargar modelos y comprobar la existencia del modelo solicitado
    models = load_models_from_json(args.config)
    print(f"Available Models: {[name for name in models]}")
    if args.model not in models:
        print(f"Error: Modelo '{args.model}' no encontrado en la librería de configuración.")
        sys.exit(1)
    model_name = args.model

    datasets_library = config.get("datasets", {})
    if args.dataset not in datasets_library:
        print(f"Error: Dataset '{args.dataset}' no encontrado en la librería de configuración.")
        sys.exit(1)
    print(f"Datasets_library: {datasets_library}")

    all_datasets = prepare_datasets_from_json(args.config, args.model, augmentation_config, restormer_config=args.restormer)
    print(f"Available Datasets: {[dataset for dataset in all_datasets]}\n")
    dataset = all_datasets[args.dataset]
    print(f"Dataset a usar: {dataset}")

    # Configurar el directorio y el SummaryWriter para TensorBoard
    timestamp = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    global_output_dir = os.path.join("runs", f"{args.output_prefix}")
    os.makedirs(global_output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=global_output_dir)
    set_writer(writer)

    # Imprimir la configuración completa del entrenamiento
    print("Configuración de Entrenamiento:")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Config File: {args.config}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Thresh Value: {args.thresh_value}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Num Workers: {args.num_workers}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  Loss Function Components: {args.composite_loss_components}")
    #print(f"  Loss Function: {args.loss}")
    #print(f"  Loss Alpha: {args.alpha}")
    #print(f"  Loss Beta: {args.beta}")
    #print(f"  Loss Gamma: {args.gamma}")
    print(f"  Logging Enabled: {args.logging}")
    print(f"  Output Prefix: {args.output_prefix}")
    print(f"  Augmentation Config: {augmentation_config}")
    print(f"  Restormer Enabled: {args.restormer}")
    print(f"  Output Directory: {global_output_dir}")

    # Registrar parámetros en archivo
    log_parameters(args, config, args.dataset, model_name, augmentation_config, args.restormer, global_output_dir)

    # Iniciar entrenamiento y evaluación
    train_and_evaluate(model_name, dataset, logging_enabled=args.logging)
