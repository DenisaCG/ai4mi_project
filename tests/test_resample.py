"""Spacing resampling: labels never interpolated, image/label stay aligned, target spacing and affine are right."""
import unittest

import nibabel as nib
import numpy as np

from slice_segthor import median_target_spacing, resample_image, resample_label, resampled_affine
from src.evaluate import stitch

SPACING, TARGET = (1.27, 1.27, 2.0), (0.977, 0.977, 2.5)


def label_volume() -> np.ndarray:
    gt = np.zeros((40, 40, 20), dtype=np.int16)
    for k in range(1, 5):
        gt[4 * k:4 * k + 8, 5:25, 2 * k:2 * k + 6] = k
    return gt


class ResampleTest(unittest.TestCase):
    def test_labels_keep_only_original_values(self):
        gt = label_volume()
        out = resample_label(gt, SPACING, TARGET)
        self.assertEqual(set(np.unique(out)), set(np.unique(gt)))
        self.assertEqual(out.dtype, gt.dtype)

    def test_image_and_label_shapes_match(self):
        ct = np.random.default_rng(0).normal(size=label_volume().shape).astype(np.float32)
        self.assertEqual(resample_image(ct, SPACING, TARGET).shape, resample_label(label_volume(), SPACING, TARGET).shape)

    def test_same_spacing_is_identity_for_labels(self):
        gt = label_volume()
        np.testing.assert_array_equal(resample_label(gt, SPACING, SPACING), gt)

    def test_image_grid_alignment(self):
        """A linear ramp must resample to the ramp evaluated at the new voxel centres (pixel-edge alignment)."""
        ramp = np.broadcast_to(np.arange(16, dtype=np.float32)[:, None, None], (16, 4, 4))
        out = resample_image(ramp, (1, 1, 1), (0.5, 1, 1))
        self.assertEqual(out.shape, (32, 4, 4))
        m = np.arange(6, 26)
        np.testing.assert_allclose(out[m, 0, 0], 0.5 * m - 0.25, atol=1e-2)  # a half-voxel shift would be 0.25

    def test_target_is_median(self):
        spacings = [(0.977, 0.977, 2.5)] * 3 + [(0.977, 0.977, 2.0)] + [(1.27, 1.27, 2.5)]
        self.assertEqual(median_target_spacing(spacings), (0.977, 0.977, 2.5))

    def test_anisotropic_target_uses_10th_percentile_on_coarsest_axis(self):
        spacings = [(1.0, 1.0, z) for z in (3.0, 4.0, 5.0, 6.0, 7.0)]  # median z 5 -> ratio 5 >= 3
        target = median_target_spacing(spacings)
        self.assertEqual(target[:2], (1.0, 1.0))
        self.assertAlmostEqual(target[2], np.percentile([3.0, 4.0, 5.0, 6.0, 7.0], 10))

    def test_affine_keeps_field_of_view(self):
        affine = np.diag([1.0, 1.0, 2.5, 1.0])
        new = resampled_affine(affine, (10, 10, 10), (10, 10, 8))
        self.assertAlmostEqual(new[2, 2], 3.125)
        self.assertAlmostEqual(new[2, 3], 0.3125)  # centre of new voxel 0 = old index 0.125
        self.assertAlmostEqual(8 * new[2, 2], 10 * affine[2, 2])


class StitchResampledTest(unittest.TestCase):
    ref = nib.Nifti1Image(np.zeros((8, 8, 6), dtype=np.uint8), np.eye(4))
    slices = {z: np.full((4, 4), z % 2, dtype=np.uint8) for z in range(4)}

    def test_slice_count_mismatch_needs_resampled_flag(self):
        with self.assertRaises(ValueError):
            stitch(self.slices, self.ref, "P")
        self.assertEqual(stitch(self.slices, self.ref, "P", resampled=True).shape, (8, 8, 6))

    def test_missing_slice_still_rejected_when_resampled(self):
        with self.assertRaises(ValueError):
            stitch({0: self.slices[0], 2: self.slices[2]}, self.ref, "P", resampled=True)


if __name__ == "__main__":
    unittest.main()
