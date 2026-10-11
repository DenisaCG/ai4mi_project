"""Boundary loss (Kervadec et al., arXiv:1812.07032) added to Dice + CE with a decreasing regional weight.

Per foreground class the loss is mean(softmax probability * signed distance map of the GT), the discrete form of the paper's
integral over the level-set function (negative inside the organ, positive outside). Combined as

    alpha * (Dice + CE) + (1 - alpha) * boundary,    alpha: alpha_start -> alpha_end, linear in the epoch.

The paper writes (1 - a) * regional + a * boundary with a raised by 0.01 per epoch, so alpha here is 1 - a. Its run is hundreds of
epochs; the schedule is rescaled to the configured epoch count. The boundary term sees the full-resolution output only, the
deep-supervision scales keep the plain Dice + CE. Distance maps come from the dataset (src.data), on the training grid, in pixels.
"""
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch import Tensor

from src.losses.diceCE import DiceCE
from src.registry import register


def one_hot2dist(onehot: np.ndarray) -> np.ndarray:
    """Signed distance map per class of a (K, H, W) one-hot GT, float32, in pixels.

    Outside a class: distance to it (> 0). Inside: minus the distance to the outside, shifted by one so the organ's innermost
    boundary layer is 0 and deeper pixels are < 0. A class with no pixel in the slice stays all zeros, as in the reference
    implementation (LIVIAETS/boundary-loss), so the term neither rewards nor penalises anything for it on that slice.
    """
    out = np.zeros(onehot.shape, dtype=np.float32)
    for k, mask in enumerate(onehot.astype(bool)):
        if mask.any():
            outside = ~mask
            out[k] = distance_transform_edt(outside) * outside - (distance_transform_edt(mask) - 1) * mask
    return out


def foreground_dist_maps(gt: Tensor) -> Tensor:
    """(K, H, W) one-hot GT -> (K, H, W) float distance maps; background (class 0) is left at zero, the loss never uses it."""
    maps = torch.zeros(gt.shape, dtype=torch.float32)
    maps[1:] = torch.from_numpy(one_hot2dist(gt[1:].numpy()))
    return maps


class BoundaryDiceCE:
    """`__call__` is the regional Dice + CE (what every deep-supervision scale gets); `combine` adds the boundary term."""

    def __init__(self, regional: DiceCE, idk: list[int], alpha_start: float, alpha_end: float, epochs: int):
        self.regional, self.idk = regional, idk
        self.alpha_start, self.alpha_end, self.epochs = alpha_start, alpha_end, epochs
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def alpha_at(self, epoch: int) -> float:
        return self.alpha_start + (self.alpha_end - self.alpha_start) * epoch / max(self.epochs - 1, 1)

    @property
    def alpha(self) -> float:
        return self.alpha_at(self.epoch)

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        return self.regional(probs, target)

    def boundary(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        return (probs[:, self.idk].float() * dist_maps[:, self.idk].float()).mean()

    def combine(self, regional: Tensor, probs: Tensor, dist_maps: Tensor) -> Tensor:
        """`regional` is the (deep-supervision weighted) Dice + CE; `probs` the full-resolution softmax."""
        return self.alpha * regional + (1 - self.alpha) * self.boundary(probs, dist_maps)


@register("loss", "boundary_dice_ce")
def build_boundary_dice_ce(num_classes: int, epochs: int, alpha_start: float = 1.0, alpha_end: float = 0.1,
                           ce_idk: list[int] | None = None, dice_idk: list[int] | None = None,
                           boundary_idk: list[int] | None = None, ce_weight: float = 1.0, dice_weight: float = 1.0,
                           eps: float = 1e-6) -> BoundaryDiceCE:
    """`epochs` is filled in by the training loop from train.epochs; the boundary term covers the foreground classes."""
    regional = DiceCE(list(range(num_classes)) if ce_idk is None else ce_idk,
                      list(range(1, num_classes)) if dice_idk is None else dice_idk, ce_weight, dice_weight, eps)
    return BoundaryDiceCE(regional, list(range(1, num_classes)) if boundary_idk is None else boundary_idk,
                          alpha_start, alpha_end, epochs)
