"""Small numerical tests for conventions that could bias the scientific results."""
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from analyze_dataset import original_stats
from analyze_baseline import bin_index, class_summary
from shape import shape_descriptor
from figures import select_shape_examples, orthogonal_plane
from utils import extent, load_png, normalized_z, overlap


class MeasurementTests(unittest.TestCase):
    def test_example_selection_extremes_median_ties_and_input_order(self):
        rows = [dict(class_id=k, patient_id=f'Patient_{p:02d}', volume_ml=v)
                for k in (1, 2, 3) for p, v in ((4, 90), (3, 30), (2, 20), (1, 10))]
        selected = select_shape_examples(rows)
        self.assertEqual(selected, select_shape_examples(rows[::-1]))
        self.assertEqual(len(selected), 9)
        for k in (1, 2, 3):
            self.assertEqual([r['patient_id'] for r in selected if r['class_id'] == k],
                             ['Patient_01', 'Patient_02', 'Patient_04'])

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


class PhysicalShapeTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((4, 5, 6), dtype=bool)
        self.mask[1:3, 1:4, 2:5] = True
        self.affine = np.diag([-2., -3., 4., 1.])
        self.affine[:3, 3] = [100, 200, -40]

    def test_anisotropic_volume_centroid_and_full_cell_extent(self):
        r = shape_descriptor(self.mask, self.affine)
        self.assertEqual(r['voxel_count'], 18)
        self.assertAlmostEqual(r['volume_mm3'], 432)
        np.testing.assert_allclose([r['centroid_i'], r['centroid_j'], r['centroid_k']], [1.5, 2, 3])
        np.testing.assert_allclose([r['centroid_world_x_mm'], r['centroid_world_y_mm'],
                                   r['centroid_world_z_mm']], [97, 194, -28])
        self.assertAlmostEqual(r['normalized_si_centroid'], .6)
        self.assertEqual(r['si_extent_mm'], 12)
        self.assertEqual(r['axis_codes'], 'LPS')

    def test_inferior_pointing_axis_preserves_world_descriptors(self):
        affine = self.affine.copy()
        affine[:3, 3] += affine[:3, 2] * (self.mask.shape[2] - 1)
        affine[:3, 2] *= -1
        a, b = shape_descriptor(self.mask, self.affine), shape_descriptor(self.mask[:, :, ::-1], affine)
        for field in ('volume_mm3', 'centroid_world_z_mm', 'normalized_si_centroid', 'si_extent_mm'):
            self.assertAlmostEqual(a[field], b[field])

    def test_permuted_axes_preserve_physical_measurements(self):
        a = shape_descriptor(self.mask, self.affine)
        b = shape_descriptor(self.mask.transpose(2, 0, 1), self.affine[:, [2, 0, 1, 3]])
        for field in ('volume_mm3', 'centroid_world_x_mm', 'centroid_world_y_mm',
                      'centroid_world_z_mm', 'normalized_si_centroid', 'si_extent_mm'):
            self.assertAlmostEqual(a[field], b[field])

    def test_oblique_extent_projects_complete_voxel_cells(self):
        c = 2 ** -.5
        affine = np.array([[2, 0, 0, 0], [0, 3*c, -4*c, 0],
                           [0, 3*c, 4*c, 0], [0, 0, 0, 1.]])
        r = shape_descriptor(self.mask, affine)
        self.assertAlmostEqual(r['volume_mm3'], 432)
        self.assertAlmostEqual(r['si_extent_mm'], (3*3 + 3*4)*c)

    def test_empty_single_voxel_and_single_slice(self):
        r = shape_descriptor(np.zeros((1, 1, 1), bool), self.affine)
        self.assertEqual(r['volume_mm3'], 0)
        self.assertTrue(np.isnan(r['centroid_i']))
        self.assertTrue(np.isnan(r['si_extent_mm']))
        r = shape_descriptor(np.ones((1, 1, 1), bool), self.affine)
        self.assertEqual(r['si_extent_mm'], 4)
        self.assertEqual(r['normalized_si_centroid'], .5)
        single = np.zeros((4, 5, 1), bool)
        single[1, 2, 0] = True
        self.assertEqual(shape_descriptor(single, self.affine)['normalized_si_centroid'], .5)

    def test_gaps_are_included_in_span(self):
        mask = np.zeros((1, 1, 5), bool)
        mask[0, 0, [0, 4]] = True
        r = shape_descriptor(mask, self.affine)
        self.assertEqual(r['voxel_count'], 2)
        self.assertEqual(r['si_extent_mm'], 20)


if __name__ == "__main__":
    unittest.main()
