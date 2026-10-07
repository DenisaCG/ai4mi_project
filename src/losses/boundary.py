"""Signed-distance boundary loss, CE + Dice + boundary and weighted CE + Dice + boundary."""
from src.distance_maps import one_hot_to_sdf
from src.losses.dice import SoftDice
from src.losses.cross_entropy import CrossEntropy
from src.losses.weighted_cross_entropy import WeightedCrossEntropy
from src.registry import register



class BoundaryLoss:
    requires_distance_maps = True

    def __init__(self, boundary_idk):
        self.boundary_idk = boundary_idk

    def __call__(self, probs, target, dist_maps=None):
        if dist_maps is None:
            dist_maps = one_hot_to_sdf(target, self.boundary_idk)
        if probs.shape != dist_maps.shape:
            raise ValueError("Probabilities and distance maps must have matching shapes")
        return (probs[:, self.boundary_idk] * dist_maps[:, self.boundary_idk].detach()).mean()

class DiceCEBoundary(BoundaryLoss):
    def __init__(
        self,
        ce_idk,
        dice_idk,
        boundary_idk,
        ce_weight=1.0,
        dice_weight=1.0,
        boundary_weight=0.01,
        eps=1e-6,
    ):
        super().__init__(boundary_idk)

        self.ce = CrossEntropy(idk=ce_idk)
        self.dice = SoftDice(dice_idk, eps)

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight

    def __call__(self, probs, target, dist_maps=None):
        return (
            self.ce_weight * self.ce(probs, target)
            + self.dice_weight * self.dice(probs, target)
            + self.boundary_weight * super().__call__(probs, target, dist_maps)
        )


class WeightedCEDiceBoundary(BoundaryLoss):
    def __init__(self, weights, ce_idk, dice_idk, boundary_idk,
                 ce_weight=1.0, dice_weight=1.0, boundary_weight=0.01, eps=1e-6):
        super().__init__(boundary_idk)
        self.ce = WeightedCrossEntropy(idk=ce_idk, weights=weights)
        self.dice = SoftDice(dice_idk, eps)
        self.ce_weight, self.dice_weight = ce_weight, dice_weight
        self.boundary_weight = boundary_weight

    def __call__(self, probs, target, dist_maps=None):
        return (self.ce_weight * self.ce(probs, target)
                + self.dice_weight * self.dice(probs, target)
                + self.boundary_weight * super().__call__(probs, target, dist_maps))


@register("loss", "boundary")
def build_boundary(num_classes, boundary_idk=None):
    classes = list(range(1, num_classes)) if boundary_idk is None else boundary_idk
    if not classes or any(c < 0 or c >= num_classes for c in classes):
        raise ValueError("Invalid boundary_idk")
    return BoundaryLoss(classes)


@register("loss", "weighted_ce_dice_boundary")
def build_weighted_combined(num_classes, weights, ce_idk=None, dice_idk=None, boundary_idk=None,
                   ce_weight=1.0, dice_weight=1.0, boundary_weight=0.01, eps=1e-6):
    return WeightedCEDiceBoundary(
        weights, list(range(num_classes)) if ce_idk is None else ce_idk,
        list(range(1, num_classes)) if dice_idk is None else dice_idk,
        build_boundary(num_classes, boundary_idk).boundary_idk,
        ce_weight, dice_weight, boundary_weight, eps)

@register("loss", "ce_dice_boundary")
def build_ce_dice_boundary(
    num_classes,
    ce_idk=None,
    dice_idk=None,
    boundary_idk=None,
    ce_weight=1.0,
    dice_weight=1.0,
    boundary_weight=0.01,
    eps=1e-6,
):
    return DiceCEBoundary(
        list(range(num_classes)) if ce_idk is None else ce_idk,
        list(range(1, num_classes)) if dice_idk is None else dice_idk,
        build_boundary(num_classes, boundary_idk).boundary_idk,
        ce_weight,
        dice_weight,
        boundary_weight,
        eps,
    )