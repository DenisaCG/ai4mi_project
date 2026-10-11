"""Probability-based post-processing on tiny arrays."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.prob_postprocess import confidence_gate, crf_labels, read_rows

try:
    import pydensecrf  # noqa: F401

    HAS_CRF = True
except ImportError:
    HAS_CRF = False


def onehot(labels: np.ndarray, k: int, confidence: float) -> np.ndarray:
    """(Z, K, H, W) probabilities: `confidence` on the label, the rest spread evenly."""
    probs = np.full((len(labels), k, *labels.shape[1:]), (1 - confidence) / (k - 1))
    for c in range(k):
        probs[:, c][labels == c] = confidence
    return probs.astype(np.float32)


class ConfidenceGateTest(unittest.TestCase):
    def test_organ_removed_on_slices_below_the_threshold_only(self):
        labels = np.zeros((3, 4, 4), dtype=np.uint8)
        labels[:, 1:3, 1:3] = 1
        probs = onehot(labels, 5, 0.95)
        probs[1, 1][labels[1] == 1] = 0.6  # slice 1: the organ is never above 0.6
        out = confidence_gate(probs, labels, 0.8, [1])
        self.assertEqual(int((out[0] == 1).sum()), 4)
        self.assertEqual(int((out[1] == 1).sum()), 0)
        self.assertEqual(int((out[2] == 1).sum()), 4)
        self.assertEqual(int((labels == 1).sum()), 12)  # input unchanged

    def test_one_confident_pixel_keeps_the_slice_and_other_organs_are_untouched(self):
        labels = np.zeros((1, 4, 4), dtype=np.uint8)
        labels[0, 0, :2] = 1
        labels[0, 3, :2] = 2
        probs = onehot(labels, 5, 0.5)
        probs[0, 1, 0, 0] = 0.99
        out = confidence_gate(probs, labels, 0.8, [1, 2])
        self.assertEqual(int((out == 1).sum()), 2)
        self.assertEqual(int((out == 2).sum()), 0)  # organ 2 never reaches 0.8 -> removed


@unittest.skipUnless(HAS_CRF, "pydensecrf not installed")
class CrfTest(unittest.TestCase):
    config = {"gauss": (3, 3), "bilateral": (20, 10, 5)}

    def setUp(self):
        self.image = np.zeros((1, 32, 32), dtype=np.uint8)
        self.image[:, :, :16] = 200
        labels = np.zeros((1, 32, 32), dtype=np.uint8)
        labels[:, :, :16] = 1
        self.probs = onehot(labels, 5, 0.9)

    def test_isolated_uncertain_pixel_is_smoothed_away_and_the_region_stays(self):
        self.probs[0, :, 16, 24] = [0.4, 0.6, 0.0, 0.0, 0.0]  # a weak organ pixel in the background
        before = self.probs.argmax(axis=1)
        self.assertEqual(before[0, 16, 24], 1)
        out = crf_labels(self.probs, self.image, self.config)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, before.shape)
        self.assertEqual(out[0, 16, 24], 0)
        self.assertTrue((out[0, 8:24, :14] == 1).all())
        self.assertTrue((out[0, 8:24, 18:] == 0).all())

    def test_slices_without_organ_probability_are_the_argmax(self):
        probs = onehot(np.zeros((2, 8, 8), dtype=np.uint8), 5, 0.95)
        out = crf_labels(probs, np.zeros((2, 8, 8), dtype=np.uint8), self.config)
        self.assertFalse(out.any())


class ReadRowsTest(unittest.TestCase):
    def test_variants_filtered_and_numbers_parsed(self):
        text = "run,patient,variant,class_idx,class_name,fp_slices,fp_voxels,pred_voxels,dice,hd95,assd\n"
        text += "r,p,baseline,1,eso,2,3,4,0.5,nan,1.5\nr,p,other,1,eso,0,0,0,0,0,0\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.csv"
            path.write_text(text)
            rows = read_rows(path, ("baseline",))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["class_idx"], 1)
        self.assertEqual(rows[0]["fp_slices"], 2)
        self.assertTrue(np.isnan(rows[0]["hd95"]))
        self.assertAlmostEqual(rows[0]["assd"], 1.5)


if __name__ == "__main__":
    unittest.main()
