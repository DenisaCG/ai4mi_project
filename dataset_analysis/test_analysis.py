"""Small numerical tests for conventions that could bias the scientific results."""
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image

from analyze_dataset import original_stats
from analyze_baseline import bin_index, class_summary
from utils import extent, load_png, normalized_z, overlap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from dataset_profile import hist_stats, identical_neighbour_slices, label_row, occupied_box, pair_row, slice_rows
from scan_geometry_figure import dot_stacks


class MeasurementTests(unittest.TestCase):
    def test_overlap_empty_false_positive_and_miss(self):
        zero = np.zeros((2, 3), dtype=bool)
        one = zero.copy()
        one[0, 1] = True
        self.assertTrue(np.isnan(overlap(zero, zero)["dice"]))
        self.assertTrue(overlap(zero, zero)["joint_empty"])
        self.assertEqual(overlap(zero, one)["fp_pixels"], 1)
        self.assertEqual(overlap(one, zero)["fn_pixels"], 1)
        self.assertEqual(overlap(one, one)["dice"], 1)
        other = one.copy()
        other[1, 1] = True
        self.assertAlmostEqual(overlap(one, other)["dice"], 2 / 3)

    def test_extent_including_absence_gaps_and_single_slice(self):
        self.assertIsNone(extent([], 0)["organ_relative_z"])
        self.assertIsNone(extent([1, 3], 2)["organ_relative_z"])
        self.assertEqual(extent([4], 4)["organ_relative_z"], .5)
        self.assertEqual(extent([1, 3, 5], 3)["distance_from_first"], 2)
        self.assertEqual(normalized_z(0, 1), 0)
        self.assertEqual(normalized_z(9, 10), 1)

    def test_original_volume_uses_physical_grid(self):
        data = np.zeros((4, 4, 3), dtype=np.uint8)
        data[1:3, 1:3, 0] = 1
        data[1:3, 1:3, 2] = 1
        nii = nib.Nifti1Image(data, np.diag([2, 3, 4, 1]))
        inv, rows = original_stats("Patient_99", "train", nii, data)
        self.assertEqual(rows[0]["voxel_count"], 8)
        self.assertAlmostEqual(rows[0]["volume_mm3"], 192)
        self.assertEqual(rows[0]["occupied_slice_count"], 2)
        self.assertEqual(rows[0]["bbox_extent_z_mm"], 12)
        self.assertEqual(rows[0]["normalized_last_z"], 1)
        self.assertFalse(rows[1]["present"])
        self.assertEqual(inv["foreground_voxels"], 8)

    def test_strict_png_encoding(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "mask.png"
            Image.fromarray(np.array([[0, 63, 126, 189]], dtype=np.uint8)).save(p)
            np.testing.assert_array_equal(load_png(p), [[0, 1, 2, 3]])
            Image.fromarray(np.array([[252]], dtype=np.uint8)).save(p)
            with self.assertRaises(ValueError):
                load_png(p)
            self.assertEqual(load_png(p, prediction=True)[0, 0], 4)
            Image.fromarray(np.array([[64]], dtype=np.uint8)).save(p)
            with self.assertRaises(ValueError):
                load_png(p, prediction=True)

    def test_summary_excludes_joint_empty_and_fp_only(self):
        zero, one = np.zeros((1, 1), bool), np.ones((1, 1), bool)
        rows = [{"class_id": k, "patient_id": "Patient_99", **overlap(g, p)}
                for k in (1, 2, 3) for g, p in ((zero, zero), (zero, one), (one, zero))]
        for r in class_summary(rows):
            self.assertEqual(r["positive_slice_dice_mean"], 0)
            self.assertEqual(r["joint_empty_slices"], 1)
            self.assertEqual(r["fp_only_slices"], 1)

    def test_bin_endpoints(self):
        edges = np.array([0, .2, .8, 1.])
        self.assertEqual(bin_index(0, edges), 0)
        self.assertEqual(bin_index(.2, edges), 1)
        self.assertEqual(bin_index(.8, edges), 2)
        self.assertEqual(bin_index(1, edges), 2)


class DatasetProfileTests(unittest.TestCase):
    def test_hist_stats_match_numpy(self):
        vals = np.random.RandomState(0).randint(-1000, 3000, size=5001)
        counts = np.bincount(vals - vals.min())
        stats = hist_stats(counts, int(vals.min()))
        for q, key in ((0.5, "p0.5"), (50, "median"), (99.5, "p99.5")):
            self.assertAlmostEqual(stats[key], np.percentile(vals, q))
        self.assertAlmostEqual(stats["mean"], vals.mean())
        self.assertAlmostEqual(stats["std"], vals.std())
        self.assertEqual((stats["min"], stats["max"], stats["n"]), (vals.min(), vals.max(), vals.size))
        self.assertIsNone(hist_stats(np.zeros(3), 0))

    def test_label_row_extent_and_pieces(self):
        mask = np.zeros((10, 10, 6), dtype=bool)
        mask[2:4, 2:4, 1:3] = True
        mask[4, 4, 3] = True
        mask[2:4, 2:4, 5] = True
        row = label_row(mask, np.array([1.0, 2.0, 2.5]))
        self.assertEqual(row["voxels"], 13)
        self.assertEqual(row["z_size_mm"], 12.5)
        self.assertEqual(row["y_size_mm"], 6.0)
        self.assertEqual(row["empty_slices_inside_range"], 1)
        self.assertEqual(row["pieces_sharing_a_face"], 3)
        self.assertEqual(row["pieces_sharing_face_edge_or_corner"], 2)
        self.assertEqual(label_row(np.zeros_like(mask), np.ones(3))["voxels"], 0)

    def test_slice_rows_sizes(self):
        mask = np.zeros((10, 10, 3), dtype=bool)
        mask[1:4, 2:7, 1] = True
        mask[8, 8, 1] = True
        (row,) = slice_rows(mask, np.array([0.5, 2.0, 3.0]))
        self.assertEqual((row["slice"], row["z_mm"]), (1, 3.0))
        self.assertEqual(row["area_mm2"], 16)
        self.assertEqual((row["x_size_mm"], row["y_size_mm"]), (4.0, 14.0))
        self.assertEqual(row["pieces_sharing_an_edge"], 2)

    def test_occupied_box_and_identical_slices(self):
        ct = np.full((8, 6, 5), -1000)
        ct[2:5, 1:3, 1:4] = 40
        ct[:, :, 2] = ct[:, :, 1]
        box = occupied_box(ct, np.array([1.0, 2.0, 2.5]))
        self.assertEqual((box["occupied_x_size_mm"], box["occupied_y_size_mm"], box["occupied_z_size_mm"]), (3, 4, 7.5))
        self.assertEqual((box["outside_box_x_low_voxels"], box["outside_box_x_high_voxels"]), (2, 3))
        self.assertAlmostEqual(box["occupied_pct_of_image"], 100 * 18 / 240)
        self.assertEqual(identical_neighbour_slices(ct), 2)

    def test_pair_row_touching_and_apart(self):
        a = np.zeros((6, 6, 6), dtype=bool)
        b = np.zeros_like(a)
        a[1, 1:3, 1:3] = True
        b[2, 1:3, 1:3] = True
        zooms = np.array([2.0, 1.0, 1.0])
        touching = pair_row(a, b, zooms)
        self.assertEqual(touching["closest_voxel_centers_mm"], 2.0)
        self.assertEqual(touching["shared_face_area_mm2"], 4.0)
        b[:] = False
        b[5, 1, 1] = True
        apart = pair_row(a, b, zooms)
        self.assertEqual((apart["closest_voxel_centers_mm"], apart["shared_face_area_mm2"]), (8.0, 0.0))
        self.assertTrue(np.isnan(pair_row(a, np.zeros_like(a), zooms)["closest_voxel_centers_mm"]))

    def test_dot_stacks_one_dot_per_value(self):
        x, y = dot_stacks(pd.Series([0.98, 0.976, 1.37, 2.0]), 0.01)
        self.assertEqual(len(x), 4)
        self.assertEqual(sorted(zip(np.round(x, 2), y)), [(0.98, 0.5), (0.98, 1.5), (1.37, 0.5), (2.0, 0.5)])


if __name__ == "__main__":
    unittest.main()
