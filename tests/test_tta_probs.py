"""Saved TTA probabilities: the layout tools.prob_postprocess reads, and the plain variant is the plain prediction."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.aggregate_pp_tta import cells
from src.eval_tta import VARIANTS, predict_variants, save_probs
from tools.prob_postprocess import patients_of
from src.config import REPO


def tiny_net():
    torch.manual_seed(0)
    return torch.nn.Conv2d(1, 5, 3, padding=1).eval()


class SavedProbsTests(unittest.TestCase):
    def setUp(self):
        self.net, self.images = tiny_net(), torch.rand(7, 1, 32, 32)

    def predict(self, **kw):
        probs = {v: None for v in kw.pop("keep")}
        labels = predict_variants(self.net, self.images, 0.0, torch.device("cpu"), batch=3, probs_out=probs, **kw)
        return labels, probs

    def test_probabilities_have_the_layout_prob_postprocess_reads(self):
        _, probs = self.predict(keep=["none", "all"])
        for v, p in probs.items():
            self.assertEqual(p.shape, (7, 5, 32, 32), v)  # (Z, K, H, W), z in slice order
            self.assertEqual(p.dtype, np.float16, v)
            np.testing.assert_allclose(p.astype(np.float32).sum(axis=1), 1.0, atol=2e-3)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_probs(root, Path("runs/exp_fold2/seed1"), "Patient_07", probs)
            for v in probs:
                saved = np.load(root / v / "exp_fold2" / "seed1" / "Patient_07.npy")
                np.testing.assert_array_equal(saved, probs[v])

    def test_none_variant_probabilities_reproduce_the_plain_prediction(self):
        labels, probs = self.predict(keep=["none"])
        plain = self.net(self.images).softmax(dim=1)
        np.testing.assert_allclose(probs["none"].astype(np.float32), plain.detach().numpy(), atol=1e-3)
        np.testing.assert_array_equal(labels["none"], plain.argmax(dim=1).numpy())
        agree = (probs["none"].astype(np.float32).argmax(axis=1) == labels["none"]).mean()
        self.assertGreater(agree, 0.999)  # float16 rounding may flip a near tie

    def test_all_variant_probabilities_are_the_average_of_the_passes(self):
        labels, probs = self.predict(keep=["all"])
        agree = (probs["all"].astype(np.float32).argmax(axis=1) == labels["all"]).mean()
        self.assertGreater(agree, 0.999)

    def test_default_is_unchanged_and_variants_limit_the_work(self):
        default = predict_variants(self.net, self.images, 0.0, torch.device("cpu"), batch=3)
        self.assertEqual(set(default), set(VARIANTS))
        few, _ = self.predict(keep=[], variants=["none", "all"])
        self.assertEqual(set(few), {"none", "all"})
        for v in few:
            np.testing.assert_array_equal(few[v], default[v])


class PatientsAndCellsTests(unittest.TestCase):
    def test_patients_come_from_the_saved_probabilities_when_there_are_no_stored_volumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = REPO / "runs" / "exp_fold0" / "seed0"
            d = Path(tmp) / "exp_fold0" / "seed0"
            d.mkdir(parents=True)
            for p in ("Patient_03", "Patient_01"):
                np.save(d / f"{p}.npy", np.zeros((1, 5, 2, 2), dtype=np.float16))
            self.assertEqual(patients_of(run, Path(tmp)), ["Patient_01", "Patient_03"])

    def test_cells_are_the_mean_of_organs_then_of_runs(self):
        rows = []
        for run, scale in (("e_fold0/avg", 1.0), ("e_fold1/avg", 0.5)):
            for k in (1, 2):
                rows.append({"variant": "avg", "run": run, "patient": "P", "class_idx": k, "dice": scale * k / 4,
                             "hd95": 10.0 * scale, "assd": 2.0 * scale})
        out = cells(rows, "avg", [1, 2], ["bg", "a", "b"])
        self.assertAlmostEqual(out["a_dice"], 0.1875)
        self.assertAlmostEqual(out["b_dice"], 0.375)
        self.assertAlmostEqual(out["fg_dice"], 0.28125)
        self.assertAlmostEqual(out["fg_hd95"], 7.5)
        self.assertAlmostEqual(out["fg_assd"], 1.5)


if __name__ == "__main__":
    unittest.main()
