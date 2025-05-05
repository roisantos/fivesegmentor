# utils/epoch_stats.py
import torch

class EpochActivationStats:
    """
    Accumulates ∑x and ∑x² over every forward pass in one epoch and
    writes mean & std to TensorBoard exactly once per epoch.
    """
    def __init__(self, writer, tag):
        self.writer = writer      # SummaryWriter
        self.tag    = tag         # e.g. "Train/activations/ResBlock0/std"
        self.reset()

    def reset(self):
        self.sum   = 0.0
        self.sum2  = 0.0
        self.count = 0

    @torch.no_grad()
    def hook(self, _module, _inp, out):
        self.sum   += out.sum().item()
        self.sum2  += (out ** 2).sum().item()
        self.count += out.numel()

    def log_epoch(self, epoch: int):
        mean = self.sum / self.count
        var  = self.sum2 / self.count - mean ** 2
        std  = var ** 0.5
        self.writer.add_scalar(f"{self.tag}/epoch_mean", mean, epoch)
        self.writer.add_scalar(f"{self.tag}/epoch_std",  std,  epoch)
        self.reset() 