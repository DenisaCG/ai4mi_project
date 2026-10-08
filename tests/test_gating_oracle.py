"""Oracle variants on tiny volumes: gating by GT slice presence and the largest 6-connected component per organ,
the variant set, the per-organ combo choice and the aggregation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.config import REPO
from tools.gating_oracle import (
    aggregate,
    choose_per_organ,
    count_table,
    emptied,
    gate,
    largest_component,
    train_pixel_mm2,
    variant_volumes,
    volume_rows,
)

NAMES = ["background", "esophagus", "heart", "trachea", "aorta"]
CLASSES = [1, 2, 3, 4]


class GateTest(unittest.TestCase):
    def test_organ_removed_only_on_slices_without_gt(self):
        gt = np.zeros((4, 4, 5), dtype=np.uint8)
        gt[:, :, 1:3] = 1  # organ 1 present on z = 1, 2
        pred = np.zeros_like(gt)
        pred[0, 0, 0:4] = 1  # z = 0 and 3 are false positives
        pred[1, 1, 2] = 2  # organ 2 has no GT anywhere
        out = gate(pred, gt, [1, 2])
        self.assertEqual(np.flatnonzero(out[0, 0] == 1).tolist(), [1, 2])
        self.assertFalse((out == 2).any())
        self.assertTrue((pred == 2).any())  # the input is not modified

    def test_other_organs_untouched(self):
        gt = np.zeros((2, 2, 3), dtype=np.uint8)
        gt[:, :, 0] = 1
        pred = np.zeros_like(gt)
        pred[0, 0, 2] = 3
        np.testing.assert_array_equal(gate(pred, gt, [1]), pred)


class LargestComponentTest(unittest.TestCase):
    def test_keeps_largest_component_per_organ(self):
        pred = np.zeros((6, 6, 6), dtype=np.uint8)
        pred[0:3, 0:3, 0:3] = 1  # 27 voxels
        pred[5, 5, 5] = 1  # isolated
        pred[0, 5, 0:2] = 2  # organ 2: 2 voxels
        pred[5, 0, 5] = 2  # organ 2: 1 voxel
        out = largest_component(pred, [1, 2])
        self.assertEqual(int((out == 1).sum()), 27)
        self.assertEqual(out[5, 5, 5], 0)
        self.assertEqual(int((out == 2).sum()), 2)

    def test_diagonal_voxels_are_separate_components(self):
        pred = np.zeros((3, 3, 3), dtype=np.uint8)
        pred[0, 0, 0] = pred[1, 1, 1] = pred[1, 1, 2] = (
            1  # the second pair touches by a face, the first voxel only by a corner
        )
        out = largest_component(pred, [1])
        self.assertEqual(int((out == 1).sum()), 2)
        self.assertEqual(out[0, 0, 0], 0)

    def test_empty_organ_and_input_unchanged(self):
        pred = np.zeros((3, 3, 3), dtype=np.uint8)
        pred[0, 0, 0] = pred[2, 2, 2] = 1
        before = pred.copy()
        largest_component(pred, [1, 2])
        np.testing.assert_array_equal(pred, before)


class VariantVolumesTest(unittest.TestCase):
    def setUp(self):
        self.gt = np.zeros((12, 12, 6), dtype=np.uint8)
        self.gt[2:10, 2:10, 1:5] = 2
        self.pred = self.gt.copy()
        self.pred[0, 0, 5] = 1  # esophagus speck, no GT
        self.pred[0, 11, 0] = 3  # trachea speck, no GT

    def volumes(self):
        return dict(variant_volumes(self.pred, self.gt, CLASSES, NAMES, 1.0))

    def test_variant_names(self):
        self.assertEqual(
            list(self.volumes()),
            [
                "baseline", "gate", "lcc", "gate_lcc",
                "lcc_esophagus", "lcc_heart", "lcc_trachea", "lcc_aorta", "lcc_heart_trachea",
                "esophagus_size_0.2x", "esophagus_size_0.05x", "esophagus_size_500vox",
                "heart_hull",
                "area_gate_5", "area_gate_10", "area_gate_25", "area_gate_50",
                "adjacent_gate",
                "zrun_3", "zrun_5", "zextent_0", "zextent_5", "slice_cc_0.1", "slice_cc_0.25",
                "contiguous_z_0", "contiguous_z_2", "contiguous_z_5",
                "trachea_anchor_20", "trachea_anchor_40", "trachea_anchor_60",
            ],
        )  # fmt: skip

    def test_per_organ_lcc_touches_only_its_organ(self):
        self.pred[0, 6, 5] = 2  # heart speck, not face-connected to the cube
        v = self.volumes()
        diff = v["lcc_heart"] != v["baseline"]
        self.assertEqual(int(diff.sum()), 1)
        np.testing.assert_array_equal(
            v["lcc_heart_trachea"] == 3, v["lcc"] == 3
        )  # trachea speck is a single component: kept
        self.assertEqual(int((v["lcc_trachea"] != v["baseline"]).sum()), 0)

    def test_oracle_gate_and_free_gates_agree_on_specks(self):
        v = self.volumes()
        self.assertEqual(v["gate"][0, 0, 5], 0)
        self.assertEqual(v["area_gate_5"][0, 0, 5], 0)
        self.assertEqual(v["adjacent_gate"][0, 0, 5], 0)
        self.assertEqual(int((v["area_gate_5"] == 2).sum()), int((self.pred == 2).sum()))

    def test_only_restricts_the_variants(self):
        got = dict(
            variant_volumes(self.pred, self.gt, CLASSES, NAMES, 1.0, only=("zrun_3", "gate"))
        )
        self.assertEqual(list(got), ["baseline", "gate", "zrun_3"])

    def test_baseline_is_the_input_and_nothing_mutates_it(self):
        before = self.pred.copy()
        v = self.volumes()
        np.testing.assert_array_equal(self.pred, before)
        np.testing.assert_array_equal(v["baseline"], before)


class VolumeRowsTest(unittest.TestCase):
    def test_gt_voxels_lost_counts_correct_baseline_voxels_the_variant_dropped(self):
        gt = np.zeros((2, 2, 2), dtype=np.uint8)
        gt[0, :, :] = 1  # 4 GT voxels
        baseline = gt.copy()
        baseline[1, 0, 0] = 1  # one false positive
        variant = baseline.copy()
        variant[0, 0, 0] = 0  # drops a true positive
        variant[1, 0, 0] = 0  # drops the false positive
        rows = volume_rows(
            REPO / "runs" / "e_fold0" / "seed0", "p", "v", variant, gt, (1.0, 1.0, 1.0), ["bg", "eso"], [1], {}, baseline
        )
        self.assertEqual(rows[0]["gt_voxels_lost"], 1)
        self.assertNotIn("gt_voxels_lost", volume_rows(
            REPO / "runs" / "e_fold0" / "seed0", "p", "v", variant, gt, (1.0, 1.0, 1.0), ["bg", "eso"], [1], {}
        )[0])


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

    def test_choice_is_strictly_better_than_baseline_and_never_the_oracle(self):
        rows = self.rows() + [
            row("gate", f"exp_fold{f}/seed0", k, 1.0) for f in (0, 1) for k in (1, 2)
        ]
        rows += [row("w", f"exp_fold{f}/seed0", 1, 0.0) for f in (0, 1)]
        rows += [row("w", f"exp_fold{f}/seed0", 2, 0.9 if f == 0 else 0.7) for f in (0, 1)]
        self.assertEqual(choose_per_organ(rows, [1, 2]), {1: "v", 2: "v"})
        rows = [r for r in rows if r["variant"] != "v"]
        self.assertEqual(choose_per_organ(rows, [1, 2]), {1: "baseline", 2: "baseline"})

    def test_hd95_choice_needs_a_better_hd95_and_a_small_dice_loss(self):
        rows = []
        for f in (0, 1):
            run = f"exp_fold{f}/seed0"
            rows += [row("baseline", run, 1, 0.80, hd95=10.0), row("baseline", run, 2, 0.80, hd95=10.0)]
            rows += [row("a", run, 1, 0.799, hd95=5.0), row("a", run, 2, 0.70, hd95=1.0)]  # fmt: skip
            rows += [row("gate", run, 1, 0.80, hd95=0.0), row("gate", run, 2, 0.80, hd95=0.0)]  # fmt: skip
        self.assertEqual(choose_per_organ(rows, [1, 2], by="hd95"), {1: "a", 2: "baseline"})
        self.assertEqual(choose_per_organ(rows, [1, 2]), {1: "baseline", 2: "baseline"})

    def test_emptied_counts_only_new_nans(self):
        nan = float("nan")
        self.assertEqual(emptied({"hd95": nan}, {"hd95": 3.0}), 1)
        self.assertEqual(emptied({"hd95": nan}, {"hd95": nan}), 0)
        self.assertEqual(emptied({"hd95": 3.0}, {"hd95": 3.0}), 0)

    def test_count_table_totals_against_the_baseline_row(self):
        rows = [row("baseline", "e_fold0/seed0", 1, 0.5, hd95=30.0),
                row("v", "e_fold0/seed0", 1, 0.5, hd95=10.0)]  # fmt: skip
        text = count_table("t", rows, [1], ["bg", "eso"], lambda r, b: int(r["hd95"] > 20))
        self.assertIn("| baseline | 1 | 1 |", text)
        self.assertIn("| v | 0 | 0 |", text)


class TrainPixelTest(unittest.TestCase):
    def test_crop_window_resized_to_the_training_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sliced"
            root.mkdir()
            (root / "roi_crop.json").write_text(
                json.dumps({"size": 320, "target_spacing": [0.8, 0.8, 2.5]})
            )
            cfg = {"data": {"root": "sliced", "preprocess": {"shape": [256, 256]}}}
            with mock.patch("tools.gating_oracle.REPO", Path(tmp)):
                self.assertAlmostEqual(train_pixel_mm2(cfg), (320 * 0.8 / 256) ** 2)


if __name__ == "__main__":
    unittest.main()
