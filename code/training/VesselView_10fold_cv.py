import os
import json
import argparse
import numpy as np
import sys


from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.nn import DataParallel

# Limpia la caché de CUDA
torch.cuda.empty_cache()

# Establecer directorio raíz y rutas para importar módulos personalizados
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, ROOT_DIR)


from ds.dataset import prepare_datasets_from_json
from utils.utils import traverseDataset, DiceLoss
from models.roinet import *
from models.common import * 

import os
import numpy as np
import argparse
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn import DataParallel
from PIL import Image
from torchvision import transforms

# ------------------------- Dataset -------------------------
class FIVESJoinedDataset(Dataset):
    """
    Dataset para imágenes y máscaras binarias.
    """
    def __init__(self, root_dir):
        self.samples = []
        img_dir = os.path.join(root_dir, "image")
        lbl_dir = os.path.join(root_dir, "label")
        print("Transformando imagenes a tensores...")
        for fn in sorted(os.listdir(img_dir)):
            if not fn.endswith(".png"): continue
            self.samples.append((os.path.join(img_dir, fn),
                                  os.path.join(lbl_dir, fn)))
        self.img_tf = transforms.ToTensor()
        self.lbl_tf = transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, lbl_p = self.samples[idx]
        img = Image.open(img_p).convert("RGB")
        lbl = Image.open(lbl_p).convert("L")
        img_t = self.img_tf(img)
        lbl_t = (self.lbl_tf(lbl) > 0.5).float()
        return img_t, lbl_t


# ------------------------- Evaluation -------------------------
def calc_result(np_pred, np_label, thresh=None):
    print("Calculando resultado...")
    import cv2
    temp = cv2.normalize(np_pred, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    if thresh is None:
        _, bin_pred = cv2.threshold(temp, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, bin_pred = cv2.threshold(temp, thresh, 1, cv2.THRESH_BINARY)
    pred_flat = bin_pred.flatten()
    lbl_flat = np_label.flatten()
    assert set(lbl_flat.tolist()) <= {0,1}, "Valores de etiqueta fuera de {0,1}"
    TP = np.sum((pred_flat==1)&(lbl_flat==1))
    TN = np.sum((pred_flat==0)&(lbl_flat==0))
    FP = np.sum((pred_flat==1)&(lbl_flat==0))
    FN = np.sum((pred_flat==0)&(lbl_flat==1))
    smooth=1e-12
    dice = (2*TP + smooth)/(2*TP + FP + FN + smooth)
    iou  = (TP + smooth)/(TP + FP + FN + smooth)
    acc  = (TP+TN)/(TP+TN+FP+FN)
    return {'dice': float(dice), 'iou':float(iou), 'acc':float(acc)}

def avg_result(results):
    agg = {}
    for r in results:
        for k,v in r.items():
            agg.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k,v in agg.items()}

# ------------------------- Traversal -------------------------
def traverse_dataset(model, loader, device, loss_fn, optimizer=None):
    print("Traverse dataset llamado...")
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    evals = []
    with torch.set_grad_enabled(training):
        for data, label in tqdm(loader, desc="Train" if training else "Val"):
            data, label = data.to(device), label.to(device)
            out = model(data)
            loss = loss_fn(out, label)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                pred_np = out.detach().cpu().numpy()[0,0]
                lbl_np  = label.cpu().numpy()[0,0]
                evals.append(calc_result(pred_np, lbl_np))
            total_loss += loss.item()
    avg_loss = total_loss/len(loader)
    metrics = {'loss':avg_loss}
    if not training:
        metrics.update(avg_result(evals))
    return metrics

# ------------------------- Main -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Ruta al dataset FIVES_joined')
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    root = args.dataset
    ds = FIVESJoinedDataset(root)
    # Extraer etiqueta final del nombre de archivo (_<disease>.png)
    # 1) Extraigo la parte tras el último '_', sin convertir aún:
    raw_labels = [
        os.path.basename(lbl_path).split('_')[-1].split('.')[0]
        for _, lbl_path in ds.samples
    ]

    # 2) Codifico cadenas únicas a enteros 0,1,2,...
    le = LabelEncoder()
    labels = le.fit_transform(raw_labels)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = DiceLoss()
    all_metrics = []

    for fold, (tr,va) in enumerate(skf.split(np.zeros(len(labels)), labels),1):
        print(f"\n=== Fold {fold} ===")
        tr_ds = Subset(ds, tr); va_ds = Subset(ds, va)
        tr_ld = DataLoader(tr_ds, batch_size=args.bs, shuffle=True)
        va_ld = DataLoader(va_ds, batch_size=1)

        # Modelo RoiNet
        model = RoiNet(
            ch_in=3, ch_out=1,
            ls_mid_ch=[32,64,128,128,64,32],
            k_size=9,
            cls_init_block=ResidualBlock,
            cls_conv_block=ResidualBlock
        ).to(device)
        if torch.cuda.device_count()>1:
            model = DataParallel(model)

        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        best = {'dice':0}
        for ep in range(1, args.epochs+1):
            traverse_dataset(model, tr_ld, device, loss_fn, optimizer=opt)
            mva = traverse_dataset(model, va_ld, device, loss_fn, optimizer=None)
            print(f"Fold {fold} Epoch {ep}: Val Dice={mva['dice']:.4f}")
            if mva['dice'] > best['dice']:
                best = mva.copy()
        print(f"Fold {fold} Mejor: {best}")
        all_metrics.append(best)

    # Estadísticas finales
    avg_all = avg_result(all_metrics)
    print("\n=== Resultados Finales ===")
    for k,v in avg_all.items(): print(f"{k}: {v:.4f}")

if __name__=='__main__':
    main()
