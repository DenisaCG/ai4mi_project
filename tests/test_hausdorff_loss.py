import unittest

import torch
import torch.nn.functional as F

import src.losses  # noqa: F401  registers losses
from src.registry import build
from utils import class2one_hot

K = 5


def gt_with_organ(hw=32):
    mask = torch.zeros(1, hw, hw, dtype=torch.long)
    mask[:, 10:16, 10:16] = 2
    return class2one_hot(mask, K)


class HausdorffLossTests(unittest.TestCase):
    def loss(self):
        return build("loss", "ce_dice_hd", num_classes=K)

    def test_hd_zero_for_perfect_prediction(self):
        gt = gt_with_organ()
        self.assertAlmostEqual(self.loss().hd(gt.float(), gt, None).item(), 0.0, places=5)

    def test_far_false_positive_costs_more_than_near(self):
        gt = gt_with_organ()
        near, far = gt.float().clone(), gt.float().clone()
        near[:, 2, 16:18, 10:12] = 1.0   # 1-2 px from the organ
        far[:, 2, 28:30, 28:30] = 1.0    # > 15 px away
        fn = self.loss()
        self.assertGreater(fn.hd(far, gt, None).item(), 3 * fn.hd(near, gt, None).item())

    def test_absent_class_hallucination_is_penalised_and_finite(self):
        gt = gt_with_organ()
        probs = gt.float().clone()
        probs[:, 4, 5:9, 5:9] = 1.0      # aorta predicted where the slice has none
        value = self.loss().hd(probs, gt, None)
        self.assertTrue(torch.isfinite(value))
        self.assertGreater(value.item(), 0.01)

    def test_gradient_flows_and_dist_maps_are_optional(self):
        gt = gt_with_organ()
        logits = torch.randn(1, K, 32, 32, requires_grad=True)
        out = self.loss()(F.softmax(logits, dim=1), gt)
        out.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
