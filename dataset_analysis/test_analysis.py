"""Small numerical tests for conventions that could bias the scientific results."""

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from PIL import Image

from analyze_dataset import original_stats
from analyze_baseline import bin_index, class_summary
from shape import shape_descriptor
from figures import select_shape_examples, orthogonal_plane
from dataset_overview_figure import density, quantiles
from error_cases_figure import crop_render
from augmentation_examples import best_slice, random_shift
from augmentation_results import paired_change
from loss_training_curves import smooth
from utils import extent, load_png, normalized_z, overlap, spreads


class MeasurementTests(unittest.TestCase):
    def test_example_selection_extremes_median_ties_and_input_order(self):
        rows = [
            dict(class_id=k, patient_id=f"Patient_{p:02d}", volume_ml=v)
            for k in (1, 2, 3, 4)
            for p, v in ((4, 90), (3, 30), (2, 20), (1, 10))
        ]
        selected = select_shape_examples(rows)
        self.assertEqual(selected, select_shape_examples(rows[::-1]))
        self.assertEqual(len(selected), 12)
        for k in (1, 2, 3, 4):
            self.assertEqual(
                [r["patient_id"] for r in selected if r["class_id"] == k],
                ["Patient_01", "Patient_02", "Patient_04"],
            )

    def test_orthogonal_plane_preserves_lps_index_mapping(self):
        array = np.arange(4 * 5 * 6).reshape(4, 5, 6)
        self.assertEqual(orthogonal_plane(array, 2, 3)[2, 1], array[1, 2, 3])
        self.assertEqual(orthogonal_plane(array, 1, 2)[3, 1], array[1, 2, 3])
        self.assertEqual(orthogonal_plane(array, 0, 1)[3, 2], array[1, 2, 3])

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
        self.assertEqual(extent([4], 4)["organ_relative_z"], 0.5)
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
            Image.fromarray(np.array([[0, 63, 126, 189, 252]], dtype=np.uint8)).save(p)
            np.testing.assert_array_equal(load_png(p), [[0, 1, 2, 3, 4]])
            Image.fromarray(np.array([[64]], dtype=np.uint8)).save(p)
            with self.assertRaises(ValueError):
                load_png(p)

    def test_summary_excludes_joint_empty_and_fp_only(self):
        zero, one = np.zeros((1, 1), bool), np.ones((1, 1), bool)
        rows = [
            {"class_id": k, "patient_id": "Patient_99", **overlap(g, p)}
            for k in (1, 2, 3, 4)
            for g, p in ((zero, zero), (zero, one), (one, zero))
        ]
        for r in class_summary(rows):
            self.assertEqual(r["positive_slice_dice_mean"], 0)
            self.assertEqual(r["joint_empty_slices"], 1)
            self.assertEqual(r["fp_only_slices"], 1)

    def test_bin_endpoints(self):
        edges = np.array([0, 0.2, 0.8, 1.0])
        self.assertEqual(bin_index(0, edges), 0)
        self.assertEqual(bin_index(0.2, edges), 1)
        self.assertEqual(bin_index(0.8, edges), 2)
        self.assertEqual(bin_index(1, edges), 2)


class PhysicalShapeTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((4, 5, 6), dtype=bool)
        self.mask[1:3, 1:4, 2:5] = True
        self.affine = np.diag([-2.0, -3.0, 4.0, 1.0])
        self.affine[:3, 3] = [100, 200, -40]

    def test_anisotropic_volume_centroid_and_full_cell_extent(self):
        r = shape_descriptor(self.mask, self.affine)
        self.assertEqual(r["voxel_count"], 18)
        self.assertAlmostEqual(r["volume_mm3"], 432)
        np.testing.assert_allclose(
            [r["centroid_i"], r["centroid_j"], r["centroid_k"]], [1.5, 2, 3]
        )
        np.testing.assert_allclose(
            [
                r["centroid_world_x_mm"],
                r["centroid_world_y_mm"],
                r["centroid_world_z_mm"],
            ],
            [97, 194, -28],
        )
        self.assertAlmostEqual(r["normalized_si_centroid"], 0.6)
        self.assertEqual(r["si_extent_mm"], 12)
        self.assertEqual(r["axis_codes"], "LPS")

    def test_inferior_pointing_axis_preserves_world_descriptors(self):
        affine = self.affine.copy()
        affine[:3, 3] += affine[:3, 2] * (self.mask.shape[2] - 1)
        affine[:3, 2] *= -1
        a, b = (
            shape_descriptor(self.mask, self.affine),
            shape_descriptor(self.mask[:, :, ::-1], affine),
        )
        for field in (
            "volume_mm3",
            "centroid_world_z_mm",
            "normalized_si_centroid",
            "si_extent_mm",
        ):
            self.assertAlmostEqual(a[field], b[field])

    def test_permuted_axes_preserve_physical_measurements(self):
        a = shape_descriptor(self.mask, self.affine)
        b = shape_descriptor(self.mask.transpose(2, 0, 1), self.affine[:, [2, 0, 1, 3]])
        for field in (
            "volume_mm3",
            "centroid_world_x_mm",
            "centroid_world_y_mm",
            "centroid_world_z_mm",
            "normalized_si_centroid",
            "si_extent_mm",
        ):
            self.assertAlmostEqual(a[field], b[field])

    def test_oblique_extent_projects_complete_voxel_cells(self):
        c = 2**-0.5
        affine = np.array(
            [[2, 0, 0, 0], [0, 3 * c, -4 * c, 0], [0, 3 * c, 4 * c, 0], [0, 0, 0, 1.0]]
        )
        r = shape_descriptor(self.mask, affine)
        self.assertAlmostEqual(r["volume_mm3"], 432)
        self.assertAlmostEqual(r["si_extent_mm"], (3 * 3 + 3 * 4) * c)

    def test_empty_single_voxel_and_single_slice(self):
        r = shape_descriptor(np.zeros((1, 1, 1), bool), self.affine)
        self.assertEqual(r["volume_mm3"], 0)
        self.assertTrue(np.isnan(r["centroid_i"]))
        self.assertTrue(np.isnan(r["si_extent_mm"]))
        r = shape_descriptor(np.ones((1, 1, 1), bool), self.affine)
        self.assertEqual(r["si_extent_mm"], 4)
        self.assertEqual(r["normalized_si_centroid"], 0.5)
        single = np.zeros((4, 5, 1), bool)
        single[1, 2, 0] = True
        self.assertEqual(
            shape_descriptor(single, self.affine)["normalized_si_centroid"], 0.5
        )

    def test_gaps_are_included_in_span(self):
        mask = np.zeros((1, 1, 5), bool)
        mask[0, 0, [0, 4]] = True
        r = shape_descriptor(mask, self.affine)
        self.assertEqual(r["voxel_count"], 2)
        self.assertEqual(r["si_extent_mm"], 20)


class LossCurveTests(unittest.TestCase):
    def test_spreads_over_present_seeds_and_folds(self):
        runs = {(f, s): float(f + 10 * s) for f in range(4) for s in range(3)}
        seed_std, fold_std = spreads(runs)
        self.assertAlmostEqual(seed_std, 10.0)
        self.assertAlmostEqual(fold_std, np.std([0, 1, 2, 3], ddof=1))
        seed_std, fold_std = spreads({k: v for k, v in runs.items() if k[1] == 0})
        self.assertTrue(np.isnan(seed_std))
        self.assertAlmostEqual(fold_std, np.std([0, 1, 2, 3], ddof=1))

    def test_smooth_is_a_centred_moving_average_with_shrinking_ends(self):
        values = np.array([[0.0, 3.0, 6.0, 9.0], [1.0, 1.0, 1.0, 1.0]])
        out = smooth(values, 3)
        np.testing.assert_allclose(out[0], [1.5, 3.0, 6.0, 7.5])
        np.testing.assert_allclose(out[1], [1.0, 1.0, 1.0, 1.0])


class DatasetOverviewFigureTests(unittest.TestCase):
    def test_quantiles_of_a_known_sample(self):
        q = quantiles(np.arange(101.0))
        self.assertEqual((q["p5"], q["median"], q["p95"]), (5.0, 50.0, 95.0))

    def test_density_peaks_at_one_and_follows_the_sample(self):
        grid = np.linspace(0, 1, 201)
        d = density(np.random.default_rng(0).normal(0.8, 0.03, 200), grid)
        self.assertEqual(d.max(), 1.0)
        self.assertAlmostEqual(grid[d.argmax()], 0.8, delta=0.03)

    def test_crop_render_keeps_only_the_framed_image_without_the_label_rows(self):
        img = np.full((100, 120, 3), 255, np.uint8)
        img[10:80, 20:100] = 0  # frame and content of the image area
        img[11:79, 21:99] = 120
        img[90:95, 40:60] = 0  # axis label text below the frame
        out = crop_render(img, label_px=8)
        self.assertEqual(out.shape, (70 - 2 - 8, 80 - 2, 3))
        self.assertTrue((out == 120).all())


class AugmentationTests(unittest.TestCase):
    def test_best_slice_maximises_the_smallest_organ(self):
        gt = np.zeros((4, 4, 3), int)
        gt[0, 0, 0], gt[0, 1, 0] = 1, 2  # slice 0 lacks organs 3 and 4
        for k, n in ((1, 2), (2, 3), (3, 2), (4, 4)):
            gt[:, :, 2].flat[np.arange(n) + 4 * (k - 1)] = k
        self.assertEqual(best_slice(gt), (2, 2))
        self.assertEqual(best_slice(np.zeros((2, 2, 2), int)), (0, 0))

    def test_shift_moves_ct_and_labels_together_and_fills_with_background(self):
        image = torch.arange(16.0).reshape(1, 4, 4) + 1
        gt = torch.zeros(2, 4, 4)
        gt[0], gt[1, 1, 1] = 1, 1
        gt[0, 1, 1] = 0
        moved, labels = random_shift(1.0, 0.25, -1.0)(image, gt)
        dx = int(torch.nonzero(labels[1])[0, 1]) - 1
        dy = int(torch.nonzero(labels[1])[0, 0]) - 1
        self.assertEqual(float(moved[0, 1 + dy, 1 + dx]), float(image[0, 1, 1]))
        self.assertTrue(torch.equal(labels.sum(0), torch.ones(4, 4)))

    def test_paired_change_uses_only_shared_runs_and_flips_the_hd95_sign(self):
        run = lambda d, h: {"dice": d, "hd95": h, "assd": 1.0}  # noqa: E731
        reference = {
            (0, 0): run(0.80, 10.0),
            (0, 1): run(0.70, 20.0),
            (1, 0): run(0.90, 12.0),
        }
        arm = {(0, 0): run(0.82, 8.0), (1, 0): run(0.94, 11.0), (2, 0): run(0.10, 99.0)}
        n, dice, hd95 = paired_change(arm, reference)
        self.assertEqual(n, 2)
        self.assertAlmostEqual(dice, 0.03)
        self.assertAlmostEqual(hd95, 1.5)  # HD95 fell by 2 and 1 mm


if __name__ == "__main__":
    unittest.main()
