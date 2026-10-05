import unittest

import torch
import torch.nn.functional as F

from src.engine import supervised_loss
from utils import class2one_hot

K = 5


class Recorder:
    """A loss that returns the width of its input and records what it was given."""

    def __init__(self):
        self.gts = []

    def __call__(self, probs, gt):
        self.gts.append(gt)
        return torch.tensor(float(probs.shape[-1]))


def logits_at(*sizes: int, b: int = 2) -> list[torch.Tensor]:
    return [torch.randn(b, K, s, s) for s in sizes]


class SupervisedLossTests(unittest.TestCase):
    def test_tensor_output_is_supervised_as_is(self):
        out = torch.randn(2, K, 16, 16)
        gt = class2one_hot(torch.randint(0, K, (2, 16, 16)), K)
        loss_fn = Recorder()
        loss, probs = supervised_loss(out, gt, loss_fn)
        self.assertEqual(loss.item(), 16.0)
        self.assertTrue(torch.equal(probs, F.softmax(out, dim=1)))
        self.assertIs(loss_fn.gts[0], gt)

    def test_list_is_weighted_one_half_quarter_and_the_lowest_scale_gets_zero(self):
        gt = class2one_hot(torch.randint(0, K, (2, 64, 64)), K)
        loss, _ = supervised_loss(logits_at(64, 32, 16, 8), gt, Recorder())
        self.assertAlmostEqual(loss.item(), (64 + 32 / 2 + 16 / 4) / 1.75, places=5)

    def test_probs_are_the_full_resolution_ones(self):
        out = logits_at(64, 32, 16)
        gt = class2one_hot(torch.randint(0, K, (2, 64, 64)), K)
        _, probs = supervised_loss(out, gt, Recorder())
        self.assertTrue(torch.equal(probs, F.softmax(out[0], dim=1)))

    def test_gt_is_downsampled_by_nearest_neighbour_and_stays_one_hot(self):
        gt = class2one_hot(torch.randint(0, K, (2, 64, 64)), K)
        loss_fn = Recorder()
        supervised_loss(logits_at(64, 32, 16, 8), gt, loss_fn)
        self.assertEqual([g.shape[-1] for g in loss_fn.gts], [64, 32, 16])
        self.assertTrue(torch.equal(loss_fn.gts[1], gt[..., ::2, ::2]))
        for g in loss_fn.gts:
            self.assertEqual(g.dtype, gt.dtype)
            self.assertTrue(torch.all(g.sum(dim=1) == 1))

    def test_weights_sum_to_one(self):
        gt = class2one_hot(torch.randint(0, K, (2, 64, 64)), K)
        loss, _ = supervised_loss(
            logits_at(64, 32, 16, 8), gt, lambda p, g: torch.tensor(3.0)
        )
        self.assertAlmostEqual(loss.item(), 3.0, places=5)


if __name__ == "__main__":
    unittest.main()
