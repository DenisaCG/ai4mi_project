"""Distance-weighted error loss aimed at Hausdorff outliers: CE + Dice + hd.

Per foreground class, hd = sum((p - g)^2 * w) / (sum(g) + sum(p) + eps), summed over the batch, where w = min(|sdf|, dmax) / dmax
is the GT boundary distance clipped and scaled to [0, 1] (the GT-only form of the Hausdorff-DT loss of Karimi & Salcudean,
arXiv:1904.10030). A false positive far from the organ costs up to 1, one at the boundary almost 0, so stray blobs that set
HD95 are punished and boundary jitter is not. A class absent from a sample has no GT distance, so every pixel of it gets
w = 1: a hallucinated organ on a slice without it pays the full price (the plain boundary loss gives such a class zero).
"""
import torch
from torch import Tensor

from src.distance_maps import one_hot_to_sdf
from src.losses.cross_entropy import CrossEntropy
from src.losses.dice import SoftDice
from src.registry import register


class DiceCEHD:
    requires_distance_maps = True

    def __init__(self, ce_idk, dice_idk, hd_idk, ce_weight=1.0, dice_weight=1.0, hd_weight=1.0, dmax=20.0, eps=1e-6):
        self.ce, self.dice = CrossEntropy(idk=ce_idk), SoftDice(dice_idk, eps)
        self.hd_idk, self.dmax, self.eps = hd_idk, dmax, eps
        self.ce_weight, self.dice_weight, self.hd_weight = ce_weight, dice_weight, hd_weight

    def hd(self, probs: Tensor, target: Tensor, dist_maps: Tensor | None) -> Tensor:
        if dist_maps is None:
            dist_maps = one_hot_to_sdf(target, self.hd_idk)
        p, g = probs[:, self.hd_idk], target[:, self.hd_idk].float()
        w = (dist_maps[:, self.hd_idk].abs() / self.dmax).clamp(max=1.0).detach()
        absent = g.sum((2, 3), keepdim=True) == 0
        w = torch.where(absent, torch.ones_like(w), w)
        err = ((p - g) ** 2 * w).sum((0, 2, 3))
        return (err / (g.sum((0, 2, 3)) + p.sum((0, 2, 3)) + self.eps)).mean()

    def __call__(self, probs: Tensor, target: Tensor, dist_maps: Tensor | None = None) -> Tensor:
        return (self.ce_weight * self.ce(probs, target) + self.dice_weight * self.dice(probs, target)
                + self.hd_weight * self.hd(probs, target, dist_maps))


@register("loss", "ce_dice_hd")
def build_ce_dice_hd(num_classes: int, ce_idk=None, dice_idk=None, boundary_idk=None, ce_weight=1.0, dice_weight=1.0,
                     hd_weight=1.0, dmax=20.0, eps=1e-6) -> DiceCEHD:
    """`boundary_idk` names the distance-map classes (the data loader reads that key); default the foreground."""
    fg = list(range(1, num_classes))
    return DiceCEHD(list(range(num_classes)) if ce_idk is None else ce_idk, fg if dice_idk is None else dice_idk,
                    fg if boundary_idk is None else boundary_idk, ce_weight, dice_weight, hd_weight, dmax, eps)
