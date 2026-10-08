"""Losses take (pred_probs, gt_onehot), both (B, K, H, W), like the course CrossEntropy."""

from torch import Tensor

from losses import CrossEntropy
from src.registry import register


class SoftTversky:
    """1 - mean over `idk` classes of TP / (TP + alpha * FP + beta * FN), summed over the whole batch.

    With alpha = beta = 0.5 this is the soft Dice loss of `SoftDice`; beta > alpha penalises missed
    foreground more than extra foreground."""

    def __init__(self, idk: list[int], alpha: float, beta: float, eps: float = 1e-6):
        self.idk, self.alpha, self.beta, self.eps = idk, alpha, beta, eps

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert probs.shape == target.shape
        p, t = probs[:, self.idk], target[:, self.idk].float()
        tp = (p * t).sum((0, 2, 3))
        fp = (p * (1 - t)).sum((0, 2, 3))
        fn = ((1 - p) * t).sum((0, 2, 3))
        return (
            1
            - (
                (tp + self.eps / 2)
                / (tp + self.alpha * fp + self.beta * fn + self.eps / 2)
            ).mean()
        )


class TverskyCE:
    """Cross-entropy + soft Tversky loss."""

    def __init__(
        self,
        ce_idk: list[int],
        tversky_idk: list[int],
        alpha: float,
        beta: float,
        eps: float = 1e-6,
    ):
        self.ce, self.tversky = (
            CrossEntropy(idk=ce_idk),
            SoftTversky(tversky_idk, alpha, beta, eps),
        )

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        return self.ce(probs, target) + self.tversky(probs, target)


@register("loss", "soft_tversky")
def build_soft_tversky(
    num_classes: int,
    alpha: float,
    beta: float,
    idk: list[int] | None = None,
    eps: float = 1e-6,
) -> SoftTversky:
    return SoftTversky(
        list(range(1, num_classes)) if idk is None else idk, alpha, beta, eps
    )


@register("loss", "tversky_ce")
def build_tversky_ce(
    num_classes: int,
    alpha: float,
    beta: float,
    ce_idk: list[int] | None = None,
    tversky_idk: list[int] | None = None,
    eps: float = 1e-6,
) -> TverskyCE:
    """CE over all classes by default, Tversky over the foreground classes only."""
    return TverskyCE(
        list(range(num_classes)) if ce_idk is None else ce_idk,
        list(range(1, num_classes)) if tversky_idk is None else tversky_idk,
        alpha,
        beta,
        eps,
    )
