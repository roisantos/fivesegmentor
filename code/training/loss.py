import torch.nn as nn
import torch
import torch.nn.functional as F
from .soft_skeleton import SoftSkeletonize

"""
try:
    import kornia as K
except ImportError:
    K = None



"""
K = None




class CompositeLoss(nn.Module):
    """
    CompositeLoss combines multiple loss functions into a single loss value via weighting.
    
    The user should pass a list (or tuple) of pairs:
    
         [(loss_fn1, weight1), (loss_fn2, weight2), ...]
         
    where each "loss_fn" is an instantiated loss function (e.g., DiceLoss(), SoftCLDiceLoss(), etc.)
    and "weight" is the corresponding weighting factor for that loss.
    
    In the forward() method, each loss is computed and added up according to its weight.
    """
    def __init__(self, losses):
        """
        Args:
            losses (list or tuple): List of pairs (loss_function, weight).
                                     Example: [(DiceLoss(), 0.4),
                                               (SoftCLDiceLoss(iter_=20, smooth=1e-12, exclude_background=True), 0.3),
                                               (FocalTverskyLoss(alpha=0.2, beta=0.8, gamma=0.5), 0.3)]
        """
        super(CompositeLoss, self).__init__()
        if not isinstance(losses, (list, tuple)):
            raise ValueError("Expected 'losses' to be a list or tuple of (loss_function, weight) pairs.")
        self.losses = losses

    def forward(self, predict, target):
        total_loss = 0.0
        for loss_fn, weight in self.losses:
            loss_val = loss_fn(predict, target)
            total_loss += weight * loss_val
        return total_loss


        

class DiceLoss(nn.Module):

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        self.epsilon = 1e-12

        pre = predict.flatten()
        tar = target.flatten()

        
        intersection = (pre * tar).sum(-1).sum()  # Multiplies the predicted value by the label as intersection

        union = (pre + tar).sum(-1).sum()
        

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
        
        return score

class SoftCLDiceLoss(nn.Module):
    """
    Implementa la función de pérdida soft clDice para tareas de segmentación.
    
    Entrada:
      - y_true: Tensor de ground truth de tamaño (B, C, H, W) with values in [0, 1].
      - y_pred: Tensor de predicción de tamaño (B, C, H, W) with values in [0, 1].
        En el caso de segmentación binaria, C suele ser 1.
    
    Salida:
      - Un tensor escalar que representa el valor de la pérdida clDice.
    
    Funcionamiento:
      1. Si exclude_background is True and the number of channels is greater than 1,
         se excluye el canal de fondo (se asume que es el canal 0).
      2. Se calcula la imagen esquelética suave (soft skeleton) tanto de y_pred como de y_true.
      3. Se calculan dos términos:
           " tprec: Precisión, que es la suma del producto elemento a elemento entre el esqueleto de la predicción y y_true, normalizada.
           " tsens: Sensibilidad, que es la suma del producto entre el esqueleto de la ground truth y y_pred, normalizada.
      4. Se computa clDice como 1  2*(tprec*tsens)/(tprec+tsens).
    """
    def __init__(self, iter_=20, smooth=0.000001, exclude_background=False):
        super(SoftCLDiceLoss, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=self.iter)
        self.exclude_background = exclude_background

    def forward(self, y_true, y_pred):
        # Convertir a float en caso de que no lo sean
        y_true = y_true.float()
        y_pred = y_pred.float()
        
        # Si se excluye el background, se opera solo si hay más de un canal
        if self.exclude_background and y_true.shape[1] > 1:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]
        
        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)
        
        # Se calcula la precisión y sensibilidad sobre los esqueletonizados
        tprec = (torch.sum(skel_pred * y_true) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(skel_true * y_pred) + self.smooth) / (torch.sum(skel_true) + self.smooth)
        
        # Se añade un epsilon en el denominador para evitar división por cero
        epsilon = 1e-8
        cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens + epsilon)
        return cl_dice

def soft_dice(y_true, y_pred, smooth=1.):
    """
    Calcula la pérdida soft Dice.
    
    Entrada:
      - y_true: Tensor ground truth de tamaño (B, C, H, W).
      - y_pred: Tensor de predicción de tamaño (B, C, H, W).
    
    Salida:
      - Un valor escalar que representa la pérdida Dice.
    """
    intersection = torch.sum(y_true * y_pred)
    dice_coeff = (2. * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
    return 1. - dice_coeff

class SoftDiceCLDiceLoss(nn.Module):
    """
    Combina la pérdida soft Dice y la soft clDice en una única función de pérdida.
    
    Entrada:
      - y_true: Tensor ground truth de tamaño (B, C, H, W).
      - y_pred: Tensor de predicción de tamaño (B, C, H, W).
    
    Salida:
      - Un tensor escalar que representa la combinación ponderada de ambas pérdidas.
    
    Funcionamiento:
      1. Se calcula la pérdida soft Dice.
      2. Se calcula la pérdida soft clDice (similar to SoftCLDiceLoss).
      3. Se retorna una combinación lineal de ambas, donde alpha pondera la contribución de clDice.
    """
    def __init__(self, iter_=3, alpha=0.5, smooth=1., exclude_background=False):
        super(SoftDiceCLDiceLoss, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha
        self.soft_skeletonize = SoftSkeletonize(num_iter=self.iter)
        self.exclude_background = exclude_background

    def forward(self, y_true, y_pred):
        y_true = y_true.float()
        y_pred = y_pred.float()
        
        if self.exclude_background and y_true.shape[1] > 1:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]
        
        dice_loss = soft_dice(y_true, y_pred, smooth=self.smooth)
        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)
        tprec = (torch.sum(skel_pred * y_true) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(skel_true * y_pred) + self.smooth) / (torch.sum(skel_true) + self.smooth)
        epsilon = 1e-8
        cl_dice_loss = 1. - 2.0 * (tprec * tsens) / (tprec + tsens + epsilon)
        combined_loss = (1.0 - self.alpha) * dice_loss + self.alpha * cl_dice_loss
        return combined_loss
    


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.2, beta=0.8, gamma=0.5, smooth=1e-6):
        """
        Parameters:
            alpha: Weight for false positives.
            beta: Weight for false negatives.
                   Typically, for vessel segmentation, you may set beta > alpha.
            gamma: Focusing parameter to emphasize misclassified pixels.
            smooth: Smoothing constant to avoid division by zero.
        """
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        """
        y_pred: Predicted probabilities (after sigmoid/softmax) of shape (B, C, H, W).
        y_true: Ground truth mask of the same shape.
        """
        # Flatten the tensors (you can also do it per batch element)
        y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.contiguous().view(-1)

        # Compute true positives, false positives and false negatives
        TP = (y_true * y_pred).sum()
        FP = ((1 - y_true) * y_pred).sum()
        FN = (y_true * (1 - y_pred)).sum()

        # Compute the Tversky index
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        # Compute the Focal Tversky loss
        focal_tversky_loss = (1 - tversky) ** self.gamma

        return focal_tversky_loss



class ConexLoss(nn.Module):
    """
    Pérdida para penalizar desconexiones, especialmente sensibles a pequeñas discontinuidades 
    en la segmentación de vasos sanguíneos de fondo de ojo.
    
    En esta versión se calcula el gradiente (diferencia entre píxeles vecinos) de la predicción.
    Se espera que en zonas donde el ground truth indique la presencia de un vaso (valor 1) la
    predicción sea suave; transiciones bruscas se traducen en gradientes altos y se penalizan.
    
    Parámetros:
      - reduction: Método para reducir la pérdida ('mean' o 'sum').
    """
    def __init__(self, reduction='mean'):
        super(ConexLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, pred, label):
        """
        Calcula la pérdida de conectividad penalizando gradientes altos en las regiones donde se esperan vasos.
        
        Entradas:
          pred: Tensor predicho de forma (B, 1, H, W), con valores en [0, 1].
          label: Tensor ground truth de forma (B, 1, H, W), idealmente binario.
        
        Salida:
          Un escalar que representa la pérdida. Los gradientes altos en regiones de vasos (label==1)
          se penalizan.
        """
        # Asegurarse de que pred y label tengan la misma forma
        assert pred.shape == label.shape, "La forma de pred y label debe ser la misma."
        
        # Calcular el gradiente horizontal (diferencia entre píxeles adyacentes en la dirección X)
        grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        # Calcular el gradiente vertical (diferencia entre píxeles adyacentes en la dirección Y)
        grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]

        # Se aplica padding para mantener el mismo tamaño en ambas direcciones
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
        
        # Calcular la magnitud del gradiente
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-12)
        
        # Penalizamos los gradientes altos en las regiones donde se espera vaso (label==1)
        loss = grad_mag * label
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss



class SoftCLDiceLossStrict(nn.Module):
    """
    Versión más estricta de SoftCLDiceLoss.
    Penaliza con mayor fuerza los errores estructurales usando una potencia sobre el harmonic mean.
    """
    def __init__(self, iter_=25, smooth=1e-6, penalty_power=5., exclude_background=False):
        super(SoftCLDiceLossStrict, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.penalty_power = penalty_power
        self.soft_skeletonize = SoftSkeletonize(num_iter=self.iter)
        self.exclude_background = exclude_background

    def forward(self, y_true, y_pred):
        y_true = y_true.float()
        y_pred = y_pred.float()

        if self.exclude_background and y_true.shape[1] > 1:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]

        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)

        tprec = (torch.sum(skel_pred * y_true) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(skel_true * y_pred) + self.smooth) / (torch.sum(skel_true) + self.smooth)

        harmonic = 2.0 * (tprec * tsens) / (tprec + tsens + self.smooth)

        # Penalizar errores estructurales con más severidad
        cl_dice = 1. - harmonic.pow(self.penalty_power)

        return cl_dice






###############################################################
# 1. DistanceWeighted BinaryCrossEntropy (DWBCE)
###############################################################
class DistanceWeightedBCELoss(nn.Module):
    """
    Binary CrossEntropy ponderada por la distancia euclídea al píxel de vaso
    más cercano (target == 1).

    Para cada píxel se calcula:
        w = exp( -d / sigma )

    de modo que los píxeles sobre el vaso reciben el máximo peso y la
    importancia decae exponencialmente con la distancia.

    Parámetros
    ----------
    sigma : float, opcional
        Controla la rapidez de la caída (en píxeles). 5.0 por defecto.
    eps : float, opcional
        Pequeña constante para evitar divisiones por cero. 1e7 por defecto.
    """

    def __init__(self, sigma: float = 5.0, eps: float = 1e-7):
        super().__init__()
        self.sigma = float(sigma)
        self.eps = float(eps)

    # --------------------------------------------------------------------- #
    # Distancia euclídea                                                        #
    # --------------------------------------------------------------------- #
    def _edt(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Calcula la Euclidean Distance Transform sobre un tensor binario.

        mask : (B, 1, H, W) bool/float
            1 en los píxeles de vaso.

        Devuelve
        --------
        dist : (B, 1, H, W) float
            Distancia (en píxeles) al píxel de vaso más cercano.
        """
        if K is not None and mask.is_cuda:
            # Kornia espera tensor float en [0,1] en GPU
            return K.morphology.distance_transform(mask)
        else:
            # Fallback CPU using SciPy
            import numpy as np
            from scipy.ndimage import distance_transform_edt

            mask_np = mask.detach().cpu().numpy()          # (B, 1, H, W)
            dist_np = np.stack(
                [distance_transform_edt(1.0 - m[0]) for m in mask_np]
            )
            dist = torch.from_numpy(dist_np).to(mask.device).unsqueeze(1)
            return dist.float()

    # --------------------------------------------------------------------- #
    # Forward                                                                 #
    # --------------------------------------------------------------------- #
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, 1, H, W) Logits sin sigmoide.
        targets : (B, 1, H, W) Máscara GT (0/1 o 0/255). Se castea a float.
        """
        # Asegurar forma BCHW
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        if logits.ndim == 3:
            logits = logits.unsqueeze(1)

        # Mismo dtype/dispositivo que los logits  (evita RuntimeError)
        targets = targets.to(dtype=logits.dtype, device=logits.device)
        # Binarizar en caso de llegar como probabilidad o 0255
        targets = (targets > 0.5).float()

        # ---- mapa de pesos dependiente de la distancia -------------------
        with torch.no_grad():
            dist = self._edt(targets)                              # (B,1,H,W)
            weights = torch.exp(-dist / (self.sigma + self.eps))
            weights = weights / (weights.mean() + self.eps)        # normaliza

        # ---- BCE ponderada ----------------------------------------------
        bce = F.binary_cross_entropy_with_logits(logits,
                                                 targets,
                                                 reduction="none")
        loss = (bce * weights).mean()
        return loss



# ===============================================================
# 2. VesselHalo Loss  (BCE + énfasis en el borde)
# ===============================================================
class VesselHaloLoss(nn.Module):
    """
    Combina BCE estándar con un término adicional que penaliza un *halo*
    (una banda estrecha alrededor de los bordes del vaso) para conseguir
    contornos más nítidos.

    Parámetros
    ----------
    band_width : int
        Radio (en px) del halo. 35 px funciona bien en imágenes 2 k×2 k.
    alpha : float
        Peso extra que reciben los píxeles del halo (halo_weight = 1+alpha).
    """

    def __init__(self, band_width: int = 5, alpha: float = 1.0):
        super().__init__()
        self.band_width = int(band_width)
        self.alpha = float(alpha)
        # Dilatación rápida con maxpool
        self.pool = nn.MaxPool2d(2 * band_width + 1,
                                 stride=1,
                                 padding=band_width)

    # ----------------------------------------------------------- #
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # --- asegurar BCHW y tipos adecuados --------------------
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        if logits.ndim == 3:
            logits = logits.unsqueeze(1)

        targets = targets.to(dtype=logits.dtype, device=logits.device)
        targets = (targets > 0.5).float()         # binarizar

        # --- BCE base ------------------------------------------
        base_bce = F.binary_cross_entropy_with_logits(logits,
                                                      targets,
                                                      reduction="none")

        # --- halo: dilatación menos máscara original -----------
        with torch.no_grad():
            dilated = (self.pool(targets) > 0.5).float()
            halo = (dilated - targets).clamp_(min=0.0)
            halo_weight = 1.0 + self.alpha * halo  # 1 fuera, 1+± en halo

        loss = (base_bce * halo_weight).mean()
        return loss


# ===============================================================
# 3. HaloCLDice Loss (estructura + contorno)
# ===============================================================
class HaloCLDiceLoss(nn.Module):
    """
    Combina *VesselHaloLoss* con un término SoftCLDice que alinea los
    esqueletos (centrelines) para promover conectividad y grosor correcto.

    Parámetros
    ----------
    band_width : int
        Radio para el halo (se pasa a `VesselHaloLoss`).
    alpha : float
        Peso dentro del halo en el término halo.
    beta  : float
        Peso relativo del término CLDice frente al halo.
    """

    def __init__(self,
                 band_width: int = 5,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 iter: int = 30):
        super().__init__()
        self.halo_loss = VesselHaloLoss(band_width=band_width, alpha=alpha)
        self.beta = float(beta)
        self.iter = iter

    # ----------------- soft skeleton helpers ------------------ #
    @staticmethod
    def _soft_erode(img: torch.Tensor) -> torch.Tensor:
        p1 = -F.max_pool2d(-img, (3, 1), stride=1, padding=(1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), stride=1, padding=(0, 1))
        return torch.minimum(p1, p2)

    def _skeletonize_soft(self, x: torch.Tensor, iters: int = 30) -> torch.Tensor:
        skel = x.clone()
        for _ in range(iters):
            skel_new = self._soft_erode(skel)
            contour = F.relu(skel - skel_new)
            if contour.sum() == 0:
                break
            skel = skel_new
        return skel

    def _cl_dice(self,
                 preds: torch.Tensor,
                 targets: torch.Tensor,
                 eps: float = 1e-7) -> torch.Tensor:
        """
        Soft CLDice de Shit et al. (2021). 0 = perfecto, 1 = fallo total.
        """
        P_soft = torch.sigmoid(preds)
        P_skel = self._skeletonize_soft(P_soft, iters = self.iter)
        T_skel = self._skeletonize_soft(targets, iters = self.iter)

        tprec = (P_skel * targets).sum(dim=[2, 3]) / (P_skel.sum(dim=[2, 3]) + eps)
        tsens = (T_skel * P_soft).sum(dim=[2, 3]) / (T_skel.sum(dim=[2, 3]) + eps)
        cl_dice = 1.0 - 2.0 * tprec * tsens / (tprec + tsens + eps)
        return cl_dice.mean()

    # ----------------------------- forward --------------------- #
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        if logits.ndim == 3:
            logits = logits.unsqueeze(1)

        targets = targets.to(dtype=logits.dtype, device=logits.device)
        targets = (targets > 0.5).float()

        loss_halo = self.halo_loss(logits, targets)           # BCE + halo
        loss_cl   = self._cl_dice(logits, targets)            # CLDice

        return loss_halo + self.beta * loss_cl