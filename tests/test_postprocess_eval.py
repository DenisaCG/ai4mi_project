"""Shared evaluation helpers on tiny volumes: the largest 6-connected component per organ, and the
per-run and per-fold aggregation of the metrics."""

import unittest

import numpy as np

from tools.postprocess_eval import aggregate, largest_component


class LargestComponentTest(unittest.TestCase):
    def test_keeps_largest_component_per_organ(self):
        pred = np.zeros((1, 6, 6), dtype=np.uint8)
        pred[0, 0:3, 0:3] = 2  # 9 voxels: the heart
        pred[0, 5, 5] = 2  # a speck, far away
        pred[0, 0, 5] = 3  # trachea: a single component
        out = largest_component(pred, [2, 3])
        self.assertEqual(int((out == 2).sum()), 9)
        self.assertEqual(int(out[0, 5, 5]), 0)
        self.assertEqual(int((out == 3).sum()), 1)

    def test_diagonal_voxels_are_separate_components(self):
        pred = np.zeros((1, 3, 3), dtype=np.uint8)
        pred[0, 0, 0] = 1
        pred[0, 1, 1] = 1  # touches only diagonally: 6-connectivity keeps one
        out = largest_component(pred, [1])
        self.assertEqual(int((out == 1).sum()), 1)

    def test_empty_organ_and_input_unchanged(self):
        pred = np.zeros((1, 2, 2), dtype=np.uint8)
        pred[0, 0, 0] = 4
        before = pred.copy()
        out = largest_component(pred, [1])
        np.testing.assert_array_equal(pred, before)
        np.testing.assert_array_equal(out, before)


def row(variant, run, k, dice, hd95=1.0, patient="p"):
    return {
        "variant": variant, "run": run, "class_idx": k, "patient": patient,
        "dice": dice, "hd95": hd95, "assd": 0.0, "fp_slices": 1,
    }  # fmt: skip


class AggregateTest(unittest.TestCase):
    def rows(self):
        out = []
        for fold, (a, b) in enumerate([(0.8, 0.9), (0.6, 0.7)]):
            run = f"exp_fold{fold}/seed0"
            for k, base in ((1, a), (2, b)):
                out += [row("baseline", run, k, base), row("v", run, k, base + 0.1)]
        return out

    def test_fg_organ_and_fold_means(self):
        agg = aggregate(self.rows(), "dice", [1, 2])
        self.assertAlmostEqual(agg["baseline"]["fg", "all"], 0.75)
        self.assertAlmostEqual(agg["baseline"][1, "all"], 0.7)
        self.assertAlmostEqual(agg["v"]["fg", 1], 0.75)
        self.assertAlmostEqual(agg["v"][2, 0], 1.0)

    def test_nan_rows_are_dropped_from_the_mean(self):
        rows = [row("baseline", "e_fold0/seed0", 1, 1.0, hd95=float("nan")),
                row("baseline", "e_fold0/seed0", 1, 1.0, hd95=4.0, patient="q")]  # fmt: skip
        self.assertAlmostEqual(aggregate(rows, "hd95", [1])["baseline"][1, "all"], 4.0)


if __name__ == "__main__":
    unittest.main()
