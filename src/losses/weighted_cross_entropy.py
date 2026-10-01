from torch import einsum
import torch

from utils import simplex, sset


class WeightedCrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        self.weights = torch.tensor(kwargs['weights'], dtype=torch.float32)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()
        weights = self.weights[self.idk].to(pred_softmax.device)
        weights = weights.view(1, -1, 1, 1)

        weighted_mask = mask * weights

        loss = - einsum("bkwh,bkwh->", weighted_mask, log_p)
        loss /= weighted_mask.sum() + 1e-10

        return loss


class PartialWeightedCrossEntropy(WeightedCrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)

from src.registry import register


@register("loss", "weighted_ce")
def build_weighted_ce(num_classes: int, weights, idk: list[int] | None = None) -> WeightedCrossEntropy:
    return WeightedCrossEntropy(idk=list(range(num_classes)) if idk is None else idk, weights = weights)
