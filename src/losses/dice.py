"""Losses take (pred_probs, gt_onehot), both (B, K, H, W), like the course CrossEntropy."""
from torch import Tensor

from src.registry import register


class SoftDice:
    """1 - mean over `idk` classes of the soft Dice, with intersection/union summed over the whole batch.

    A class absent from the batch (e.g. aorta in the original SEGTHOR labels) scores ~0 Dice with a
    vanishing gradient, so it adds a constant to the loss value instead of a training signal."""

    def __init__(self, idk: list[int], eps: float = 1e-6):
        self.idk, self.eps = idk, eps

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert probs.shape == target.shape
        p, t = probs[:, self.idk], target[:, self.idk].float()
        inter = (p * t).sum((0, 2, 3))
        return 1 - ((2 * inter + self.eps) / (p.sum((0, 2, 3)) + t.sum((0, 2, 3)) + self.eps)).mean()


@register("loss", "soft_dice")
def build_soft_dice(num_classes: int, idk: list[int] | None = None, eps: float = 1e-6) -> SoftDice:
    return SoftDice(list(range(1, num_classes)) if idk is None else idk, eps)
