"""Tests for the nnU-Net dataset conversion and split writer."""

import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from prepare_dataset import LABELS, convert, write_splits

IDS = [f"Patient_{i:02d}" for i in range(1, 21)]
EXTRA_FILES = ["GT_original.nii.gz", "GT.nii.seg.nrrd", "GT.nii.gz.labels.csv"]


def make_source(root: Path, ids: list[str] = IDS) -> Path:
    """Build a tiny corrected-release folder, including the files nnU-Net must not see."""
    data_dir = root / "train"
    for n, pid in enumerate(ids):
        folder = data_dir / pid
        folder.mkdir(parents=True)
        ct = np.full((4, 4, 4), n, dtype=np.int16)
        gt = (np.arange(64).reshape(4, 4, 4) % 5).astype(np.uint8)
        nib.save(nib.Nifti1Image(ct, np.eye(4)), folder / f"{pid}.nii.gz")
        nib.save(nib.Nifti1Image(gt, np.eye(4)), folder / "GT.nii.gz")
        for name in EXTRA_FILES:
            (folder / name).write_text("stray")
    (data_dir / ".DS_Store").write_text("stray")
    return data_dir


class ConvertTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.data_dir = make_source(self.root)
        self.dataset = self.root / "raw" / "Dataset102_SegTHOR_corrected"

    def test_layout_and_dataset_json(self):
        self.assertEqual(convert(self.data_dir, self.dataset), IDS)
        images = sorted(p.name for p in (self.dataset / "imagesTr").iterdir())
        labels = sorted(p.name for p in (self.dataset / "labelsTr").iterdir())
        self.assertEqual(images, [f"{i}_0000.nii.gz" for i in IDS])
        self.assertEqual(labels, [f"{i}.nii.gz" for i in IDS])
        meta = json.loads((self.dataset / "dataset.json").read_text())
        self.assertEqual(meta["channel_names"], {"0": "CT"})
        self.assertEqual(meta["labels"], LABELS)
        self.assertEqual(meta["numTraining"], 20)
        self.assertEqual(meta["file_ending"], ".nii.gz")

    def test_content_matches_source_and_is_linked(self):
        convert(self.data_dir, self.dataset)
        src = self.data_dir / "Patient_03" / "GT.nii.gz"
        dst = self.dataset / "labelsTr" / "Patient_03.nii.gz"
        self.assertEqual(dst.read_bytes(), src.read_bytes())
        self.assertTrue(dst.samefile(src))
        ct = nib.load(self.dataset / "imagesTr" / "Patient_03_0000.nii.gz")
        self.assertTrue((np.asarray(ct.dataobj) == 2).all())

    def test_rerun_is_idempotent(self):
        convert(self.data_dir, self.dataset)
        before = {p: p.read_bytes() for p in self.dataset.rglob("*") if p.is_file()}
        convert(self.data_dir, self.dataset)
        after = {p: p.read_bytes() for p in self.dataset.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_missing_or_empty_data_dir(self):
        with self.assertRaises(FileNotFoundError):
            convert(self.root / "nowhere", self.dataset)
        (self.root / "empty").mkdir()
        with self.assertRaises(FileNotFoundError):
            convert(self.root / "empty", self.dataset)

    def test_patient_without_ground_truth(self):
        (self.data_dir / "Patient_05" / "GT.nii.gz").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Patient_05"):
            convert(self.data_dir, self.dataset)


class SplitTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.dataset = self.root / "raw"
        self.preprocessed = self.root / "preprocessed"
        self.preprocessed.mkdir()
        convert(make_source(self.root), self.dataset)

    def read(self) -> list[dict]:
        return json.loads((self.preprocessed / "splits_final.json").read_text())

    def test_single_fold_with_fixed_validation_patients(self):
        write_splits(self.dataset, self.preprocessed)
        (fold,) = self.read()
        self.assertEqual(
            fold["val"],
            ["Patient_01", "Patient_11", "Patient_15", "Patient_17", "Patient_19"],
        )
        self.assertEqual(len(fold["train"]), 15)
        self.assertEqual(sorted(fold["train"] + fold["val"]), IDS)
        self.assertFalse(set(fold["train"]) & set(fold["val"]))

    def test_deterministic_output(self):
        write_splits(self.dataset, self.preprocessed)
        first = (self.preprocessed / "splits_final.json").read_bytes()
        write_splits(self.dataset, self.preprocessed)
        self.assertEqual((self.preprocessed / "splits_final.json").read_bytes(), first)

    def test_unknown_validation_patient(self):
        (self.dataset / "labelsTr" / "Patient_19.nii.gz").unlink()
        with self.assertRaisesRegex(ValueError, "Patient_19"):
            write_splits(self.dataset, self.preprocessed)
        self.assertFalse((self.preprocessed / "splits_final.json").exists())

    def test_requires_planned_dataset(self):
        with self.assertRaises(FileNotFoundError):
            write_splits(self.dataset, self.root / "not_planned")


if __name__ == "__main__":
    unittest.main()
