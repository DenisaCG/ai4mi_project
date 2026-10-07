"""Fold-weighted cross entropy combined with foreground soft Dice."""
from torch import Tensor

from src.losses.dice import SoftDice
from src.losses.weighted_cross_entropy import WeightedCrossEntropy
from src.registry import register


class WeightedCEDice:
    """ce_weight * class-weighted CE + dice_weight * unweighted soft Dice."""

    def __init__(self, weights: list[float], ce_idk: list[int], dice_idk: list[int],
                 ce_weight: float = 1.0, dice_weight: float = 1.0, eps: float = 1e-6):
        self.ce = WeightedCrossEntropy(idk=ce_idk, weights=weights)
        self.dice = SoftDice(dice_idk, eps)
        self.ce_weight, self.dice_weight = ce_weight, dice_weight

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        return self.ce_weight * self.ce(probs, target) + self.dice_weight * self.dice(probs, target)


@register("loss", "weighted_ce_dice")
def build_weighted_dice_ce(num_classes: int, weights: list[float],
                           ce_idk: list[int] | None = None,
                           dice_idk: list[int] | None = None,
                           ce_weight: float = 1.0, dice_weight: float = 1.0,
                           eps: float = 1e-6) -> WeightedCEDice:
    return WeightedCEDice(
        weights,
        list(range(num_classes)) if ce_idk is None else ce_idk,
        list(range(1, num_classes)) if dice_idk is None else dice_idk,
        ce_weight, dice_weight, eps,
    )