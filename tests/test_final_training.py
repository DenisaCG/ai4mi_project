"""The final-run path: retains 0 (train on every labelled patient, no val split) and the test-set post-processing."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from slice_segthor import get_splits
from src.config import REPO, load_config
from src.data import has_val
from src.predict_test import largest_component
from src.splits import validate_split


class SplitTests(unittest.TestCase):
    def test_retains_zero_means_all_patients_train(self):
        validate_split(40, 0, 0)

    def test_retains_zero_rejects_folds(self):
        with self.assertRaises(ValueError):
            validate_split(40, 0, 1)
        with self.assertRaises(ValueError):
            validate_split(40, 0, 0, num_folds=4)

    def test_other_retains_values_still_validated(self):
        for retains in (-1, 40, 41):
            with self.assertRaises(ValueError):
                validate_split(40, retains, 0)
        validate_split(40, 10, 3, num_folds=4)

    def test_get_splits_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp)
            for i in range(1, 7):
                (src / "train" / f"Patient_{i:02d}").mkdir(parents=True)
            (src / "test").mkdir()
            for name in ("Patient_41.nii.gz", "Patient_42.nii.gz", "test.zip", ".DS_Store"):
                (src / "test" / name).write_text("x")
            train, val, test = get_splits(src, 0, 0)
        self.assertEqual(sorted(train), [f"Patient_{i:02d}" for i in range(1, 7)])
        self.assertEqual(val, [])
        self.assertEqual(test, ["Patient_41", "Patient_42"])  # test.zip next to the CTs is not a patient


class ConfigTests(unittest.TestCase):
    def test_final_configs_load_without_val(self):
        for name in ("final_enet_dice_ce", "final_best_model"):
            cfg = load_config(REPO / "configs" / f"{name}.yaml")
            self.assertEqual(cfg["data"]["preprocess"]["retains"], 0)
            self.assertFalse(has_val(cfg), name)
            self.assertIsNone(cfg["data"]["source_pattern"])

    def test_train_select_metric_only_for_final_runs(self):
        with self.assertRaises(ValueError):
            load_config(REPO / "configs" / "full_cv4_enet_dice_ce.yaml", ["train.select_metric=train_dice_fg"])
        load_config(REPO / "configs" / "final_enet_dice_ce.yaml", ["train.select_metric=train_dice_fg"])

    def test_cv_configs_keep_their_val_split(self):
        self.assertTrue(has_val(load_config(REPO / "configs" / "full_cv4_enet_dice_ce.yaml")))


class LargestComponentTests(unittest.TestCase):
    def volume(self):
        v = np.zeros((20, 20, 20), np.uint8)
        v[2:8, 2:8, 2:8] = 2      # heart, big
        v[15:17, 15:17, 15:17] = 2  # heart, small stray blob
        v[2:5, 12:15, 2:5] = 3    # trachea, big
        v[15:16, 2:3, 15:16] = 3  # trachea, one stray voxel
        v[10:12, 2:3, 10:12] = 1  # esophagus piece
        v[12:14, 2:3, 12:14] = 1  # a second esophagus piece
        v[10:12, 8:10, 10:12] = 4  # aorta
        v[13:15, 8:10, 13:15] = 4  # a second aorta piece
        return v

    def test_heart_and_trachea_keep_only_the_largest_component(self):
        v = self.volume()
        out = largest_component(v)
        self.assertEqual((out == 2).sum(), 216)
        self.assertEqual((out == 3).sum(), 27)
        self.assertEqual(out[15, 15, 15], 0)  # stray heart voxels become background
        self.assertEqual(out[15, 2, 15], 0)

    def test_esophagus_and_aorta_are_untouched(self):
        v = self.volume()
        out = largest_component(v)
        for k in (1, 4):
            np.testing.assert_array_equal(out == k, v == k)

    def test_input_is_not_modified_and_single_component_is_a_no_op(self):
        v = self.volume()
        before = v.copy()
        largest_component(v)
        np.testing.assert_array_equal(v, before)
        single = np.zeros((6, 6, 6), np.uint8)
        single[1:4, 1:4, 1:4] = 2
        np.testing.assert_array_equal(largest_component(single), single)

    def test_diagonal_neighbours_are_one_component(self):
        v = np.zeros((6, 6, 6), np.uint8)
        v[1, 1, 1] = 3
        v[2, 2, 2] = 3  # touches only at a corner: 26-connected
        np.testing.assert_array_equal(largest_component(v), v)


if __name__ == "__main__":
    unittest.main()
