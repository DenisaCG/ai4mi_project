"""CT intensity normalization: fixed HU window -> 0..255, train-foreground stats, load-time z-score, labels untouched."""
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from slice_segthor import ct_norm_stats, foreground_hu, slice_patient, window_arr
from src.data import img_transform

RNG = np.random.default_rng(0)


def foreground_sample() -> np.ndarray:
    return np.concatenate([RNG.normal(40, 30, 20000), RNG.normal(-950, 20, 2000), RNG.uniform(-1000, 1500, 500)])


class WindowTest(unittest.TestCase):
    def test_nothing_exceeds_window_and_it_spans_0_255(self):
        hu = np.concatenate([foreground_sample(), [-1024, 3000, 31743]]).astype(np.int16)
        stats = ct_norm_stats(hu)
        out = window_arr(hu, stats["lo"], stats["hi"])
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual((out.min(), out.max()), (0, 255))
        self.assertEqual(window_arr(np.array([stats["lo"] - 500, stats["hi"] + 5000]), stats["lo"], stats["hi"]).tolist(), [0, 255])

    def test_mapping_is_patient_independent(self):
        """Same HU -> same grey value whatever else is in the volume (norm_arr's min-max is not)."""
        a, b = np.array([-1000, 0, 100, 200], dtype=np.int16), np.array([-1000, 0, 100, 30000], dtype=np.int16)
        np.testing.assert_array_equal(window_arr(a, -991, 248)[:3], window_arr(b, -991, 248)[:3])

    def test_stats_are_foreground_percentiles_and_zscore_clipped_foreground(self):
        fg = foreground_sample()
        stats = ct_norm_stats(fg)
        np.testing.assert_allclose([stats["lo"], stats["hi"]], np.percentile(fg, [0.5, 99.5]))
        z = (np.clip(fg, stats["lo"], stats["hi"]) - stats["mean"]) / stats["std"]
        self.assertAlmostEqual(z.mean(), 0, places=6)
        self.assertAlmostEqual(z.std(), 1, places=6)

    def test_loader_zscore_is_dequantized_hu_zscore(self):
        stats = {"lo": -991.0, "hi": 248.0, "mean": 13.0, "std": 200.0}
        img = Image.fromarray(np.array([[0, 255]], dtype=np.uint8))
        np.testing.assert_allclose(img_transform(img, stats)[0, 0].numpy(),
                                   [(-991 - 13) / 200, (248 - 13) / 200], rtol=1e-6)
        np.testing.assert_allclose(img_transform(img)[0, 0].numpy(), [0, 1])  # default path: plain /255


class SlicePatientTest(unittest.TestCase):
    """Synthetic 512x512x135 patient: labels must be identical with and without normalization."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = Path(self.tmp.name) / "src"
        pdir = self.src / "train" / "Patient_01"
        pdir.mkdir(parents=True)
        ct = RNG.normal(40, 60, (512, 512, 135)).astype(np.int16)
        ct[:50] = -1000
        ct[-1, -1, -1] = 31000
        gt = np.zeros(ct.shape, dtype=np.int16)
        for k in range(1, 5):
            gt[100 + 40 * k:130 + 40 * k, 200:260, 10 * k:10 * k + 40] = k
        affine = np.diag([1.0, 1.0, 3.0, 1.0])
        nib.save(nib.Nifti1Image(ct, affine), str(pdir / "Patient_01.nii.gz"))
        nib.save(nib.Nifti1Image(gt, affine), str(pdir / "GT.nii.gz"))
        self.ct, self.gt = ct, gt

    def tearDown(self):
        self.tmp.cleanup()

    def test_foreground_is_label_gt_0(self):
        fg = foreground_hu("Patient_01", self.src)
        self.assertEqual(fg.size, int((self.gt > 0).sum()))
        np.testing.assert_array_equal(fg, self.ct[self.gt > 0].astype(np.float32))

    def test_labels_byte_identical_and_image_windowed(self):
        stats = ct_norm_stats(foreground_hu("Patient_01", self.src))
        out = {}
        for name, norm in (("plain", None), ("window", stats)):
            dest = Path(self.tmp.name) / name
            slice_patient("Patient_01", dest, self.src, (256, 256), norm_stats=norm)
            out[name] = dest
        gts = sorted((out["plain"] / "gt").glob("*.png"))
        self.assertEqual(len(gts), 135)
        for g in gts:
            self.assertEqual(g.read_bytes(), (out["window"] / "gt" / g.name).read_bytes())
        imgs = [np.array(Image.open(f)) for f in sorted((out["window"] / "img").glob("*.png"))]
        # the outlier voxel (31000 HU) no longer squashes everything: soft tissue spreads over many grey levels
        self.assertGreater(len(np.unique(np.concatenate([i.ravel() for i in imgs]))), 100)
        self.assertLessEqual(max(i.max() for i in imgs), 255)


if __name__ == "__main__":
    unittest.main()
