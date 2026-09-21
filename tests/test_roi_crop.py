"""ROI crop/pad to a fixed window: image-only window, nothing clipped, image and label aligned, air padding, exact inverse."""
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from slice_segthor import (crop_pad_inplane, ct_norm_stats, foreground_hu, paste_window_inplane, roi_centre,
                           roi_required_size, roi_start, roi_window_size, slice_patient)
from src.config import load_config
from src.evaluate import stitch

TARGET = (1.0, 1.0, 3.0)  # equal to the synthetic native spacing, so resampling is the identity and grids stay comparable
CENTRE, RADII = (40, 256), (35, 90)  # body hugging the left edge of the 512 frame: the window has to overhang and pad
BOXES = {1: (30, 36, 240, 250), 2: (36, 55, 235, 285), 3: (32, 40, 255, 262), 4: (45, 55, 225, 235)}  # x0, x1, y0, y1


def make_patient(root: Path, patient: str, boxes: dict, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """512x512x135 CT: air (-1000) around an ellipse body (~40 HU) and a separate couch slab; organs as HU-coded boxes."""
    xx, yy = np.meshgrid(np.arange(512), np.arange(512), indexing="ij")
    body = ((xx - CENTRE[0]) / RADII[0]) ** 2 + ((yy - CENTRE[1]) / RADII[1]) ** 2 < 1
    ct = np.full((512, 512, 135), -1000, dtype=np.int16)
    ct[body] = np.random.default_rng(seed).normal(40, 40, (int(body.sum()), 135)).astype(np.int16)
    ct[:, CENTRE[1] + RADII[1] + 6:CENTRE[1] + RADII[1] + 10, :] = 100  # couch: a separate component, must not move the centre
    gt = np.zeros(ct.shape, dtype=np.int16)
    for k, (x0, x1, y0, y1) in boxes.items():
        gt[x0:x1, y0:y1, 10:100] = k
        ct[x0:x1, y0:y1, 10:100] = 100 + 80 * k  # HU tells which class a grey pixel belongs to
    pdir = root / "train" / patient
    pdir.mkdir(parents=True)
    affine = np.diag([1.0, 1.0, 3.0, 1.0])
    nib.save(nib.Nifti1Image(ct, affine), str(pdir / f"{patient}.nii.gz"))
    nib.save(nib.Nifti1Image(gt, affine), str(pdir / "GT.nii.gz"))
    return ct, gt


class CropPadTest(unittest.TestCase):
    def test_window_inside_overhanging_and_larger_than_the_volume(self):
        arr = np.arange(10 * 12 * 2, dtype=np.int16).reshape(10, 12, 2) + 1
        for start, size in (((2, 3), 5), ((-3, -2), 6), ((7, 9), 6), ((-4, -4), 20)):
            out = crop_pad_inplane(arr, start, size, -7)
            self.assertEqual(out.shape, (size, size, 2), (start, size))
            for i in range(size):
                for j in range(size):
                    x, y = start[0] + i, start[1] + j
                    inside = 0 <= x < 10 and 0 <= y < 12
                    np.testing.assert_array_equal(out[i, j], arr[x, y] if inside else [-7, -7], err_msg=str((start, size, i, j)))

    def test_paste_is_the_inverse_and_drops_what_the_window_missed(self):
        gt = np.zeros((10, 12, 2), dtype=np.int16)
        gt[1:9, 2:10] = 3
        for start, size in (((2, 3), 5), ((-3, -2), 6), ((7, 9), 6), ((-4, -4), 20)):
            back = paste_window_inplane(crop_pad_inplane(gt, start, size, 0), start, gt.shape[:2])
            expected = np.zeros_like(gt)
            x0, y0 = max(start[0], 0), max(start[1], 0)
            x1, y1 = min(start[0] + size, 10), min(start[1] + size, 12)
            expected[x0:x1, y0:y1] = gt[x0:x1, y0:y1]
            np.testing.assert_array_equal(back, expected, err_msg=str((start, size)))
            self.assertEqual(back.dtype, gt.dtype)

    def test_window_size_rounding(self):
        self.assertEqual(roi_window_size(258), 288)  # 258 + 2 x 15 = 288 is already a multiple of 32
        self.assertEqual(roi_window_size(258.5), 320)
        self.assertEqual(roi_window_size(60), 96)

    def test_start_centres_the_window(self):
        self.assertEqual(roi_start(np.array([100.0, 40.0]), 32), (84, 24))
        self.assertEqual(roi_start(np.array([10.0, 256.0]), 96), (-38, 208))


class CentreIsImageOnlyTest(unittest.TestCase):
    def test_signature_has_no_label_argument(self):
        self.assertEqual(list(inspect.signature(roi_centre).parameters), ["ct"])

    def test_centre_is_the_body_not_the_couch(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ct, _ = make_patient(Path(tmp.name), "Patient_01", BOXES)
        np.testing.assert_allclose(roi_centre(ct.astype(np.float32)), CENTRE, atol=2)

    def test_all_air_is_an_error_not_a_silent_window(self):
        with self.assertRaises(ValueError):
            roi_centre(np.full((20, 20, 8), -1000, dtype=np.float32))


class SlicePatientCropTest(unittest.TestCase):
    """Whole path on synthetic patients: T from the train labels, slicing, and the inverse through src.evaluate.stitch."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.src = cls.root / "src"
        cls.ct, cls.gt = make_patient(cls.src, "Patient_01", BOXES)
        # same CT, other labels: the window must not move (labels never place it)
        _, cls.gt_other = make_patient(cls.src, "Patient_02", {2: (30, 60, 200, 300), 4: (35, 45, 220, 230)})
        cls.required = roi_required_size("Patient_01", cls.src, TARGET)
        cls.size = roi_window_size(cls.required)
        cls.stats = ct_norm_stats(foreground_hu("Patient_01", cls.src, TARGET))
        cls.out = {}
        for patient in ("Patient_01", "Patient_02"):
            cls.out[patient] = cls.root / "dest" / patient / "train"
            slice_patient(patient, cls.out[patient], cls.src, (cls.size, cls.size), target_spacing=TARGET,
                          norm_stats=cls.stats, crop_size=cls.size)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def crop_info(self, patient: str) -> dict:
        return json.loads((self.out[patient].parent / "roi_crop" / f"{patient}.json").read_text())

    def test_required_size_is_twice_the_farthest_label_edge(self):
        self.assertAlmostEqual(self.required, 2 * max(CENTRE[0] - 30, 55 - CENTRE[0], CENTRE[1] - 225, 285 - CENTRE[1]), delta=4)
        self.assertEqual(self.size, 96)

    def test_every_slice_is_exactly_t_by_t_and_the_window_overhangs(self):
        for kind in ("img", "gt"):
            files = sorted((self.out["Patient_01"] / kind).glob("*.png"))
            self.assertEqual(len(files), 135)
            for f in files:
                with Image.open(f) as im:
                    self.assertEqual(im.size, (self.size, self.size), f.name)
        self.assertLess(self.crop_info("Patient_01")["start"][0], 0)  # the padding path is really exercised

    def test_nothing_is_clipped_and_labels_are_the_window_of_the_original(self):
        info = self.crop_info("Patient_01")
        self.assertEqual(info["retained"], 1.0)
        expected = crop_pad_inplane(self.gt, tuple(info["start"]), self.size, 0)
        got = np.stack([np.array(Image.open(self.out["Patient_01"] / "gt" / f"Patient_01_{z:04d}.png")) for z in range(135)], axis=-1)
        np.testing.assert_array_equal(got, (expected * 63).astype(np.uint8))

    def test_image_and_label_are_aligned(self):
        """Each organ box was given its own HU: every pixel labelled k must have the grey value that HU maps to."""
        z = 50
        img = np.array(Image.open(self.out["Patient_01"] / "img" / f"Patient_01_{z:04d}.png"))
        gt = np.array(Image.open(self.out["Patient_01"] / "gt" / f"Patient_01_{z:04d}.png")) // 63
        for k in (2, 3, 4):
            self.assertEqual(len(np.unique(img[gt == k])), 1, k)
        self.assertTrue(len({int(np.unique(img[gt == k])[0]) for k in (2, 3, 4)}) == 3)
        self.assertEqual(set(np.unique(img[gt == 1])), {0})  # class 1's HU is the window's low end

    def test_padding_is_air(self):
        """The overhang is padded before normalisation with the window's low end: grey 0, like real air."""
        pad = -self.crop_info("Patient_01")["start"][0]
        self.assertGreater(pad, 3)
        img = np.array(Image.open(self.out["Patient_01"] / "img" / "Patient_01_0050.png"))
        self.assertEqual(set(np.unique(img[:pad - 1, :])), {0})  # rows = x, so the first rows are the overhang
        self.assertGreater(img[pad + 10:, :].max(), 0)  # ...while the organs inside the volume are not (tissue below `lo` is 0 too)

    def test_window_does_not_depend_on_the_labels(self):
        self.assertEqual(self.crop_info("Patient_01")["start"], self.crop_info("Patient_02")["start"])

    def test_stitch_puts_perfect_predictions_back_exactly(self):
        info = self.crop_info("Patient_01")
        slices = {z: np.array(Image.open(self.out["Patient_01"] / "gt" / f"Patient_01_{z:04d}.png")) // 63 for z in range(135)}
        reference = nib.Nifti1Image(np.zeros(self.gt.shape, dtype=np.uint8), np.eye(4))
        got = stitch(slices, reference, "Patient_01", resampled=True, crop=info)
        np.testing.assert_array_equal(got, self.gt.astype(np.uint8))

    def test_stitch_without_the_crop_info_would_misplace_everything(self):
        slices = {z: np.array(Image.open(self.out["Patient_01"] / "gt" / f"Patient_01_{z:04d}.png")) // 63 for z in range(135)}
        reference = nib.Nifti1Image(np.zeros(self.gt.shape, dtype=np.uint8), np.eye(4))
        self.assertFalse(np.array_equal(stitch(slices, reference, "Patient_01", resampled=True), self.gt.astype(np.uint8)))


class ConfigTest(unittest.TestCase):
    BASE = "configs/segthor_enet_ce_corrected_ct_window_zscore_median_spacing.yaml"
    CROP = "configs/segthor_enet_ce_corrected_ct_window_zscore_median_spacing_roi_crop.yaml"

    def test_crop_changes_the_dataset_and_leaves_existing_ones_alone(self):
        base, crop = load_config(self.BASE), load_config(self.CROP)
        self.assertNotIn("crop", base["data"]["preprocess"])  # no new default key: existing sliced_<hash> dirs stay valid
        self.assertEqual(crop["data"]["preprocess"]["crop"], "roi")
        self.assertNotEqual(base["data"]["root"], crop["data"]["root"])
        self.assertEqual({k: v for k, v in crop["data"]["preprocess"].items() if k != "crop"}, base["data"]["preprocess"])

    def test_crop_needs_the_common_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text("experiment: bad\ndata:\n  preprocess: {source_dir: x, gt_version: corrected, crop: roi}\n")
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
