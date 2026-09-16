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
from profile_figures.f01_03_scan_geometry import dot_stacks
from profile_figures.f05_label_intensity import binned_counts
from profile_figures.f06_07_label_size_intensity import resample_mask, shades
from profile_figures.f08_11_slices_and_change import slice_changes
from profile_figures.f13_label_pairs import label_grids
from profile_figures.f09_label_bounding_box import box_edges, label_boxes
from profile_figures.f07_label_shapes_and_sizes import label_boxes as shape_boxes
from profile_figures.f04_scan_intensity.common import nnunet_ct_normalisation, normalise, pool_histograms, range_percent, slice_regions


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

    def test_binned_counts_crops_and_sums(self):
        x, c = binned_counts(np.array([1, 2, 3, 4, 5]), start=-2, lo=0, hi=4, width=2)
        self.assertEqual(x.tolist(), [1.0, 3.0])
        self.assertEqual(c.tolist(), [7.0, 5.0])
        self.assertEqual(binned_counts(np.array([9]), start=50, lo=0, hi=4, width=2)[1].tolist(), [0.0, 0.0])

    def test_label_boxes_outer_faces_and_reference_shift(self):
        rows = []
        for patient, (lo, hi, centre) in {"P2": (4.0, 10.0, 7.0), "P1": (2.0, 6.0, 4.0)}.items():
            for label, shift in ((1, 0.0), (2, 1.0)):
                row = {"patient": patient, "label": label}
                for a in "xyz":
                    row |= {f"{a}_min_mm": lo + shift, f"{a}_max_mm": hi + shift, f"{a}_center_mm": centre + shift}
                    row[f"{a}_size_mm"] = hi - lo + 2.0
                rows.append(row)
        labels = pd.DataFrame(rows)
        start, size = label_boxes(labels, 1)
        self.assertEqual(start[:, 0].tolist(), [1.0, 3.0])
        self.assertEqual(size[:, 0].tolist(), [6.0, 8.0])
        shifted, _ = label_boxes(labels, 1, reference=2)
        self.assertEqual(shifted[:, 2].tolist(), [-4.0, -5.0])

    def test_box_edges_twelve_axis_aligned(self):
        edges = box_edges(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
        self.assertEqual(len(edges), 12)
        lengths = sorted(float(np.abs(b - a).sum()) for a, b in edges)
        self.assertEqual(lengths, [4.0] * 4 + [5.0] * 4 + [6.0] * 4)
        self.assertTrue(all(np.count_nonzero(b != a) == 1 for a, b in edges))

    def test_shades_light_to_full_colour(self):
        out = shades("#000000", 3)
        self.assertEqual(len(out), 3)
        self.assertTrue(np.allclose(out[-1], 0))
        self.assertTrue(out[0][0] > out[1][0] > out[2][0])

    def test_resample_mask_crops_and_rescales(self):
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[2:6, 3:5, 4:5] = True
        out = resample_mask(mask, np.array([1.0, 1.0, 2.0]), 1.0)
        self.assertEqual(out.shape, (4, 2, 2))
        self.assertTrue(out.all())


    def test_label_boxes_span_all_present_labels(self):
        labels = pd.DataFrame(
            {
                "patient": ["P1", "P1", "P1"],
                "label": [1, 2, 4],
                "voxels": [5, 7, 0],
                **{f"{a}_min_mm": [0.0, -3.0, -99.0] for a in "xyz"},
                **{f"{a}_max_mm": [4.0, 2.0, 99.0] for a in "xyz"},
            }
        )
        box = shape_boxes(labels).loc["P1"]
        self.assertEqual((box["x_min_mm"], box["x_max_mm"]), (-3.0, 4.0))

class ScanIntensityTests(unittest.TestCase):
    def test_pool_histograms_aligns_starts(self):
        counts, start = pool_histograms([(np.array([1, 2]), -3), (np.array([5]), -2), (np.array([7]), 1)])
        self.assertEqual(start, -3)
        self.assertEqual(counts.tolist(), [1, 7, 0, 0, 7])

    def test_range_percent_folds_values_outside_edges(self):
        vals = np.array([-1005, -1000, -990, -981, -980, 800, 900])
        counts = np.bincount(vals - vals.min())
        pct = range_percent(counts, int(vals.min()), np.array([-1000, -980, 800]))
        self.assertAlmostEqual(pct.sum(), 100)
        self.assertEqual((pct * 7 / 100).round().tolist(), [4, 3])

    def test_nnunet_normalisation_from_sample(self):
        vals = np.random.RandomState(1).randint(-900, 400, size=4001)
        counts = np.bincount(vals - vals.min())
        hists = {("P1", "nnunet sample"): (counts, int(vals.min())), ("P1", "scan"): (np.array([99]), 5000)}
        norm = nnunet_ct_normalisation(hists)
        self.assertAlmostEqual(norm["clip_low"], np.percentile(vals, 0.5))
        self.assertAlmostEqual(norm["clip_high"], np.percentile(vals, 99.5))
        self.assertAlmostEqual(norm["mean"], vals.mean())
        out = normalise(np.array([-5000.0, norm["mean"], 5000.0]), norm)
        self.assertAlmostEqual(out[1], 0)
        self.assertAlmostEqual(out[2], (norm["clip_high"] - norm["mean"]) / norm["std"])

    def test_slice_regions_thirds_and_largest_patch(self):
        ct = np.full((40, 40), -1000)
        ct[5:36, 5:36] = 0
        ct[18:23, 18:23] = -1000
        ct[0, 0] = 50
        regions = slice_regions(ct, (1.0, 1.0))
        self.assertEqual(regions[0, 0], 0)
        self.assertEqual(regions[5, 20], 1)
        self.assertEqual(regions[20, 20], 3)
        self.assertEqual(set(np.unique(regions[5:36, 5:36])), {1, 2, 3})
        self.assertFalse(slice_regions(np.full((4, 4), -1000), (1.0, 1.0)).any())

    def test_slice_changes_position_area_and_change(self):
        slices = pd.DataFrame({"patient": "P", "label": 1, "slice": [7, 5, 6], "area_mm2": [50.0, 100.0, 200.0]})
        out = slice_changes(slices)
        self.assertEqual(out["slice"].tolist(), [5, 6, 7])
        self.assertEqual(out["pos_pct"].tolist(), [0.0, 50.0, 100.0])
        self.assertEqual(out["area_pct_max"].tolist(), [50.0, 100.0, 25.0])
        self.assertTrue(np.isnan(out["change_pct"].iloc[0]))
        self.assertEqual(out["change_pct"].iloc[1:].tolist(), [100.0, -75.0])

    def test_label_grids_share_one_grid(self):
        seg = np.zeros((12, 12, 6), dtype=np.int16)
        seg[2:4, 2:4, 1:3] = 1
        seg[4:8, 2:4, 1:3] = 2
        grids = label_grids(seg, np.array([1.0, 1.0, 2.0]), [1, 2, 3], 1.0)
        self.assertEqual({g.shape for g in grids.values()}, {(8, 4, 6)})
        self.assertEqual((int(grids[1].sum()), int(grids[2].sum()), int(grids[3].sum())), (16, 32, 0))
        self.assertFalse(grids[1][0].any() or grids[2][-1].any())


if __name__ == "__main__":
    unittest.main()
