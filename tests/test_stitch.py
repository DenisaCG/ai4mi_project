"""src.evaluate.stitch must rebuild exactly what the course stitch.py produced for the baseline.

stitch.py resizes each slice on its own in 2D; src.evaluate.stitch resizes the stacked volume once
in 3D, which is only equivalent because Z is unchanged. legacy_stitch below is that per-slice
algorithm, copied from stitch.py's merge_patient, so the equivalence is checked on every run
instead of only where the baseline's (gitignored) outputs happen to exist.
"""
import unittest

import nibabel as nib
import numpy as np
from PIL import Image
from skimage.transform import resize

from src.config import REPO, load_config
from src.evaluate import group_by_patient, stitch

LEGACY_PRED = REPO / "results" / "segthor" / "ce" / "best_epoch" / "val"
LEGACY_VOL = REPO / "volumes" / "segthor" / "ce"


def legacy_stitch(slices: dict[int, np.ndarray], shape: tuple[int, int, int]) -> np.ndarray:
    X, Y, Z = shape
    out = np.zeros((X, Y, Z), dtype=np.uint8)
    for z, pred in slices.items():
        out[:, :, z] = resize(pred, (X, Y), mode="constant", preserve_range=True,
                              anti_aliasing=False, order=0)
    return out


def reference(shape: tuple[int, int, int]) -> nib.Nifti1Image:
    return nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), np.eye(4))


class StitchTests(unittest.TestCase):
    def test_matches_per_slice_resize(self):
        rng = np.random.default_rng(0)
        for shape in ((512, 512, 7), (301, 277, 5), (256, 256, 3), (128, 190, 4)):
            slices = {z: rng.integers(0, 5, (256, 256), dtype=np.uint8) for z in range(shape[2])}
            got = stitch(slices, reference(shape), "Patient_99")
            np.testing.assert_array_equal(got, legacy_stitch(slices, shape), err_msg=str(shape))
            self.assertEqual(got.dtype, np.uint8)
            self.assertTrue(set(np.unique(got)) <= set(range(5)), shape)

    @unittest.skipUnless(LEGACY_PRED.is_dir() and LEGACY_VOL.is_dir(), "baseline outputs missing")
    def test_matches_legacy_stitch(self):
        cfg = load_config("configs/segthor_enet_ce.yaml")
        preds = {p.stem: np.asarray(Image.open(p)) // 63 for p in LEGACY_PRED.glob("*.png")}
        for patient, slices in group_by_patient(preds, cfg["data"]["patient_regex"]).items():
            ref = nib.load(REPO / cfg["data"]["source_pattern"].format(patient=patient))
            legacy = np.asarray(nib.load(LEGACY_VOL / f"{patient}.nii.gz").dataobj)
            np.testing.assert_array_equal(stitch(slices, ref, patient), legacy, err_msg=patient)

    def test_missing_slice_is_an_error(self):
        ref = nib.Nifti1Image(np.zeros((8, 8, 3), dtype=np.uint8), np.eye(4))
        with self.assertRaises(ValueError):
            stitch({0: np.zeros((4, 4), np.uint8), 2: np.zeros((4, 4), np.uint8)}, ref, "Patient_99")


if __name__ == "__main__":
    unittest.main()
