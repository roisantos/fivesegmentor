import torch.nn as nn
import torch
import torch.nn.functional as F
from .soft_skeleton import SoftSkeletonize



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