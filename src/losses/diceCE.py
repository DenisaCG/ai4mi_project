"""Losses take (pred_probs, gt_onehot), both (B, K, H, W), like the course CrossEntropy."""
from torch import Tensor

from losses import CrossEntropy
from src.losses.dice import SoftDice
from src.registry import register


class DiceCE:
    """ce_weight * cross-entropy + dice_weight * soft Dice loss."""

    def __init__(self, ce_idk: list[int], dice_idk: list[int], ce_weight: float = 1.0,
                 dice_weight: float = 1.0, eps: float = 1e-6):
        self.ce, self.dice = CrossEntropy(idk=ce_idk), SoftDice(dice_idk, eps)
        self.ce_weight, self.dice_weight = ce_weight, dice_weight

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        return self.ce_weight * self.ce(probs, target) + self.dice_weight * self.dice(probs, target)


@register("loss", "dice_ce")
def build_dice_ce(num_classes: int, ce_idk: list[int] | None = None, dice_idk: list[int] | None = None,
                  ce_weight: float = 1.0, dice_weight: float = 1.0, eps: float = 1e-6) -> DiceCE:
    """CE over all classes by default, Dice over the foreground classes only."""
    return DiceCE(list(range(num_classes)) if ce_idk is None else ce_idk,
                  list(range(1, num_classes)) if dice_idk is None else dice_idk, ce_weight, dice_weight, eps)
