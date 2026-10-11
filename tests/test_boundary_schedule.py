"""The scheduled boundary loss ported from the boundary-loss branch: schedule, combination, distance maps, training-loop hooks."""
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from src.engine import supervised_loss
from src.losses.boundary_schedule import WithDistMaps, build_boundary_dice_ce, foreground_dist_maps, one_hot2dist
from src.registry import build


def one_hot(labels: torch.Tensor, k: int = 5) -> torch.Tensor:
    return F.one_hot(labels.long(), k).permute(0, 3, 1, 2).float()


class DistanceMapTests(unittest.TestCase):
    def test_sign_and_one_pixel_shift_inside(self):
        m = np.zeros((1, 9, 9), np.uint8)
        m[0, 3:6, 3:6] = 1
        d = one_hot2dist(m)[0]
        self.assertEqual(d[4, 4], -1.0)          # deepest pixel of a 3x3 organ: edt 2, shifted by one
        self.assertEqual(d[3, 3], 0.0)           # innermost boundary layer is 0
        self.assertEqual(d[3, 2], 1.0)           # one pixel outside
        self.assertGreater(d[0, 0], 3.0)

    def test_absent_class_is_all_zero_and_background_is_left_out(self):
        gt = one_hot(torch.zeros(1, 8, 8))[0]    # only background present
        maps = foreground_dist_maps(gt)
        self.assertEqual(maps.shape, gt.shape)
        self.assertEqual(float(maps.abs().sum()), 0.0)


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.loss = build("loss", "boundary_dice_ce_scheduled", num_classes=5, epochs=50)

    def test_alpha_runs_linearly_from_1_to_0p1(self):
        self.assertAlmostEqual(self.loss.alpha_at(0), 1.0)
        self.assertAlmostEqual(self.loss.alpha_at(49), 0.1)
        self.assertAlmostEqual(self.loss.alpha_at(24), 1.0 - 0.9 * 24 / 49)

    def test_set_epoch_changes_the_weight(self):
        self.loss.set_epoch(49)
        self.assertAlmostEqual(self.loss.alpha, 0.1)

    def test_call_is_plain_dice_ce_and_combine_adds_the_boundary_term_once(self):
        torch.manual_seed(0)
        gt = one_hot(torch.randint(0, 5, (2, 16, 16)))
        probs = F.softmax(torch.randn(2, 5, 16, 16), dim=1)
        maps = torch.stack([foreground_dist_maps(g) for g in gt])
        regional = self.loss(probs, gt)
        torch.testing.assert_close(regional, build_boundary_dice_ce(5, 50).regional(probs, gt))
        self.loss.set_epoch(10)
        a = self.loss.alpha
        expected = a * regional + (1 - a) * (probs[:, 1:] * maps[:, 1:]).mean()
        torch.testing.assert_close(self.loss.combine(regional, probs, maps), expected)


class TrainingHookTests(unittest.TestCase):
    def test_deep_supervision_scales_get_plain_dice_ce_and_boundary_is_added_after_the_sum(self):
        torch.manual_seed(1)
        gt = one_hot(torch.randint(0, 5, (2, 16, 16)))
        out = [torch.randn(2, 5, 16, 16), torch.randn(2, 5, 8, 8), torch.randn(2, 5, 4, 4)]
        loss_fn = build("loss", "boundary_dice_ce_scheduled", num_classes=5, epochs=10)
        loss_fn.set_epoch(5)
        regional, probs = supervised_loss(out, gt, loss_fn)          # no boundary inside the deep-supervision sum
        plain, _ = supervised_loss(out, gt, build("loss", "dice_ce", num_classes=5))
        torch.testing.assert_close(regional, plain)
        maps = torch.stack([foreground_dist_maps(g) for g in gt])
        combined = loss_fn.combine(regional, probs, maps)
        self.assertNotAlmostEqual(float(combined), float(regional), places=4)

    def test_dataset_wrapper_adds_maps_of_the_given_target(self):
        class Base(torch.utils.data.Dataset):
            def __len__(self): return 2
            def __getitem__(self, i): return {"images": torch.zeros(1, 8, 8), "gts": one_hot(torch.randint(0, 5, (1, 8, 8)))[0], "stems": f"P_{i:04d}"}
        item = WithDistMaps(Base())[1]
        self.assertEqual(tuple(item["dist_maps"].shape), (5, 8, 8))
        self.assertEqual(float(item["dist_maps"][0].abs().sum()), 0.0)   # background never used


if __name__ == "__main__":
    unittest.main()
