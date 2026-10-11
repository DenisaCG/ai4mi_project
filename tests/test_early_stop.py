"""Plateau LR schedule and early stopping rule."""

import unittest

import torch

from src.engine import should_stop
from src.registry import build


class EarlyStopTests(unittest.TestCase):
    def test_stops_after_patience_epochs_without_gain(self):
        self.assertFalse(should_stop(epoch=14, best_epoch=0, patience=15))
        self.assertTrue(should_stop(epoch=15, best_epoch=0, patience=15))

    def test_patience_zero_never_stops(self):
        self.assertFalse(should_stop(epoch=1000, best_epoch=0, patience=0))


class PlateauSchedulerTests(unittest.TestCase):
    def test_halves_lr_after_patience_and_respects_floor(self):
        import src.optim  # noqa: F401  registers schedulers

        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=4e-5)
        sched = build(
            "scheduler", "plateau", optimizer=opt, epochs=50, patience=2, min_lr=1e-5
        )
        lrs = []
        for _ in range(12):
            sched.step(0.8)  # never improves after the first call
            lrs.append(opt.param_groups[0]["lr"])
        self.assertEqual(lrs[0], 4e-5)
        self.assertAlmostEqual(lrs[3], 2e-5)
        self.assertAlmostEqual(min(lrs), 1e-5)


if __name__ == "__main__":
    unittest.main()
