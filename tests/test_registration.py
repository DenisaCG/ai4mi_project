"""Fold-specific offline registration planning and train-only cache integration."""

import contextlib
import io
import random
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import SimpleITK as sitk
from PIL import Image

from slice_segthor import get_splits
from src.config import REPO, config_hash, load_config
from src.data import build_dataset
from src.folds import run_configs
from src.registration import (check_pair, cyclic_pairs, expected_manifest,
                              fold_patients, fold_plan, fold_root, processed_root,
                              synthetic_slice_files, warp_pair)


BASE = REPO / "configs" / "full_cv4_enet_ce_registration.yaml"
IMPROVED = REPO / "configs" / "full_cv4_enet_dice_ce_registration.yaml"


class RegistrationPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name)
        for index in range(1, 41):
            (self.source / "train" / f"Patient_{index:02d}").mkdir(parents=True)

    def test_pairing_is_deterministic_balanced_and_validation_free(self):
        for fold in range(4):
            train, val = fold_patients(self.source, 10, fold, 4, 0)
            state = random.getstate()
            try:
                random.seed(0)
                with contextlib.redirect_stdout(io.StringIO()):
                    reference_train, reference_val, _ = get_splits(self.source, 10, fold, 4)
            finally:
                random.setstate(state)
            self.assertEqual(set(train), set(reference_train))
            self.assertEqual(set(val), set(reference_val))
            pairs = cyclic_pairs(train, fold, 42)
            self.assertEqual(len(pairs), 30)
            self.assertEqual({p["source_patient"] for p in pairs}, set(train))
            self.assertEqual({p["target_patient"] for p in pairs}, set(train))
            self.assertFalse({p["source_patient"] for p in pairs} & set(val))
            self.assertFalse({p["target_patient"] for p in pairs} & set(val))
            self.assertEqual(pairs, cyclic_pairs(train, fold, 42))
        self.assertEqual(pairs[0]["synthetic_id"], "Synthetic_f3_00")

    def test_fold_zero_first_pair_emerges_from_the_general_algorithm(self):
        train, val = fold_patients(self.source, 10, 0, 4, 0)
        first = cyclic_pairs(train, 0, 42)[0]
        self.assertEqual((first["source_patient"], first["target_patient"]),
                         ("Patient_26", "Patient_19"))
        self.assertFalse({first["source_patient"], first["target_patient"]} & set(val))

    def test_training_seed_does_not_change_registration_source(self):
        for config in (BASE, IMPROVED):
            a = run_configs(load_config(config), fold=0)[0]
            b = run_configs(load_config(config, ["seed=2"]), fold=0)[0]
            self.assertEqual(fold_plan(a), fold_plan(b))
            self.assertEqual(fold_root(a), fold_root(b))
            self.assertEqual(processed_root(a), processed_root(b))

    def test_null_registration_keeps_existing_run_hash(self):
        cfg = load_config(REPO / "configs/full_cv4_enet_ce.yaml")
        previous = dict(cfg)
        previous["data"] = {k: v for k, v in cfg["data"].items() if k != "registration"}
        self.assertEqual(config_hash(cfg), config_hash(previous))

    def test_both_arms_share_niftis_and_keep_reference_settings(self):
        for fold in range(4):
            base = run_configs(load_config(BASE), fold=fold)[0]
            improved = run_configs(load_config(IMPROVED), fold=fold)[0]
            base_reference = run_configs(load_config(REPO / "configs/full_cv4_enet_ce.yaml"), fold=fold)[0]
            improved_reference = run_configs(load_config(REPO / "configs/full_cv4_enet_dice_ce.yaml"), fold=fold)[0]
            self.assertEqual(fold_root(base), fold_root(improved))
            self.assertNotEqual(processed_root(base), processed_root(improved))
            self.assertEqual(fold_plan(base), fold_plan(improved))
            for augmented, reference in ((base, base_reference), (improved, improved_reference)):
                self.assertEqual(augmented["data"]["root"], reference["data"]["root"])
                self.assertEqual(augmented["data"]["preprocess"], reference["data"]["preprocess"])
                self.assertEqual(augmented["data"]["source_pattern"], reference["data"]["source_pattern"])
                for key in ("model", "loss", "optim", "scheduler", "train", "eval"):
                    self.assertEqual(augmented[key], reference[key])
                self.assertEqual(augmented["data"]["augment"], [])
                self.assertIsNotNone(re.fullmatch(
                    augmented["data"]["patient_regex"], f"Synthetic_f{fold}_00_0000"
                ))

    def test_partial_registration_fold_cannot_be_used_for_training(self):
        cfg = load_config(BASE)
        cfg["data"]["preprocess"]["source_dir"] = str(self.source)
        fold_cfg = run_configs(cfg, fold=0)[0]
        manifest = expected_manifest(fold_cfg)
        with mock.patch("src.registration.load_manifest", return_value=manifest):
            with self.assertRaisesRegex(ValueError, "incomplete"):
                synthetic_slice_files(fold_cfg)


class RegistrationWarpTests(unittest.TestCase):
    def test_target_to_source_resampling_preserves_multiclass_gt(self):
        ct_array = np.full((8, 16, 16), -900, np.float32)
        gt_array = np.zeros_like(ct_array, dtype=np.uint8)
        for label, region in enumerate(((slice(1, 3), slice(1, 3)),
                                        (slice(5, 7), slice(1, 3)),
                                        (slice(1, 3), slice(5, 7)),
                                        (slice(5, 7), slice(5, 7))), start=1):
            gt_array[2:6, region[0], region[1]] = label
            ct_array[2:6, region[0], region[1]] = label * 50
        source_ct = sitk.GetImageFromArray(ct_array)
        source_gt = sitk.GetImageFromArray(gt_array)
        target_ct = sitk.Image(source_ct)
        target_ct.SetOrigin((1.0, 0.0, 0.0))
        # Target physical x=i+1 maps to source physical x=i. This tests the
        # fixed->moving convention rather than merely an identity transform.
        transform = sitk.TranslationTransform(3, (-1.0, 0.0, 0.0))
        ct, gt = warp_pair(source_ct, source_gt, target_ct, transform)
        check_pair(ct, gt)
        self.assertEqual(ct.GetOrigin(), target_ct.GetOrigin())
        np.testing.assert_array_equal(sitk.GetArrayFromImage(gt), gt_array)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(ct), ct_array.astype(np.int32))


class RegistrationDatasetTests(unittest.TestCase):
    def test_synthetic_slices_are_appended_to_train_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic = root / "synthetic"
            for split, stem in (("train", "Patient_01_0000"), ("val", "Patient_02_0000")):
                for kind in ("img", "gt"):
                    path = root / split / kind / f"{stem}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("L", (8, 8)).save(path)
            for kind in ("img", "gt"):
                path = synthetic / "train" / kind / "Synthetic_f0_00_0000.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("L", (8, 8)).save(path)
            cfg = load_config(REPO / "configs/segthor_enet_ce.yaml")
            cfg["data"]["root"] = str(root)
            cfg["data"]["registration"] = {"root": str(synthetic), "pairing_seed": 42}
            extra = (synthetic / "train" / "img" / "Synthetic_f0_00_0000.png",
                     synthetic / "train" / "gt" / "Synthetic_f0_00_0000.png")
            with mock.patch("src.registration.synthetic_slice_files", return_value=[extra]) as lookup:
                train = build_dataset(cfg, "train")
                val = build_dataset(cfg, "val")
            self.assertEqual((len(train), len(val)), (2, 1))
            lookup.assert_called_once_with(cfg)
            self.assertEqual([p.stem for p, _ in val.files], ["Patient_02_0000"])


if __name__ == "__main__":
    unittest.main()
