import unittest

import numpy as np
import torch

import src.losses  # noqa: F401  registers losses
from src.engine import supervised_loss
from src.losses.boundary import foreground_dist_maps, one_hot2dist
from src.registry import build
from utils import class2one_hot

K, HW = 5, 64


def square_gt(top: int = 20, left: int = 20, size: int = 20) -> torch.Tensor:
    """(1, K, HW, HW) one-hot GT with a single organ, class 2, a square."""
    labels = torch.zeros(1, HW, HW, dtype=torch.int64)
    labels[:, top:top + size, left:left + size] = 2
    return class2one_hot(labels, K)


def loss(epochs: int = 25, **kwargs):
    return build("loss", "boundary_dice_ce", num_classes=K, epochs=epochs, **kwargs)


class DistanceMapTests(unittest.TestCase):
    def test_negative_inside_positive_outside(self):
        gt = square_gt()[0]
        dist = one_hot2dist(gt.numpy())[2]
        mask = gt[2].bool().numpy()
        self.assertLessEqual(dist[mask].max(), 0)  # the innermost boundary layer is exactly 0, as in the reference
        self.assertLess(dist[30, 30], 0)           # deep inside
        self.assertTrue((dist[~mask] > 0).all())
        self.assertEqual(dist[19, 30], 1)          # one pixel above the square
        self.assertEqual(dist[0, 30], 20)

    def test_absent_class_stays_zero_and_background_is_not_computed(self):
        maps = foreground_dist_maps(square_gt()[0])
        self.assertEqual(maps.shape, (K, HW, HW))
        self.assertTrue((maps[[0, 1, 3, 4]] == 0).all())
        self.assertTrue((maps[2] != 0).any())
        self.assertEqual(maps.dtype, torch.float32)


class BoundaryLossTests(unittest.TestCase):
    def test_perfect_prediction_beats_a_shifted_one(self):
        gt = square_gt()
        dist = foreground_dist_maps(gt[0])[None]
        fn = loss()
        shifted = square_gt(top=20, left=26)
        self.assertLess(fn.boundary(gt.float(), dist).item(), 0)
        self.assertLess(fn.boundary(gt.float(), dist).item(), fn.boundary(shifted.float(), dist).item())

    def test_alpha_schedule_runs_from_first_to_last_value(self):
        fn = loss(epochs=25, alpha_start=1.0, alpha_end=0.1)
        fn.set_epoch(0)
        self.assertEqual(fn.alpha, 1.0)
        fn.set_epoch(24)
        self.assertAlmostEqual(fn.alpha, 0.1)
        fn.set_epoch(12)
        self.assertAlmostEqual(fn.alpha, 0.55)
        self.assertEqual(loss(epochs=1).alpha_at(0), 1.0)  # a one-epoch run does not divide by zero

    def test_combine_weights_regional_and_boundary(self):
        gt = square_gt()
        dist = foreground_dist_maps(gt[0])[None]
        fn = loss()
        fn.set_epoch(24)
        total = fn.combine(torch.tensor(2.0), gt.float(), dist)
        self.assertAlmostEqual(total.item(), 0.1 * 2.0 + 0.9 * fn.boundary(gt.float(), dist).item(), places=5)

    def test_boundary_term_only_sees_the_full_resolution_output(self):
        gt = square_gt()
        dist = foreground_dist_maps(gt[0])[None]
        fn = loss()
        out = [torch.randn(1, K, HW, HW), torch.randn(1, K, HW // 2, HW // 2), torch.randn(1, K, HW // 4, HW // 4)]
        regional, probs = supervised_loss(out, gt, fn)  # deep supervision: plain Dice + CE per scale
        self.assertEqual(probs.shape, (1, K, HW, HW))
        total = fn.combine(regional, probs, dist)
        self.assertAlmostEqual(total.item(), (fn.alpha * regional + (1 - fn.alpha) * fn.boundary(probs, dist)).item(), places=5)

    def test_gradient_reaches_the_logits(self):
        gt = square_gt()
        dist = foreground_dist_maps(gt[0])[None]
        logits = torch.randn(1, K, HW, HW, requires_grad=True)
        fn = loss()
        fn.set_epoch(24)
        total = fn.combine(torch.tensor(0.0), torch.softmax(logits, 1), dist)
        total.backward()
        self.assertGreater(logits.grad.abs().sum().item(), 0)
        self.assertTrue(np.isfinite(logits.grad.numpy()).all())


if __name__ == "__main__":
    unittest.main()
