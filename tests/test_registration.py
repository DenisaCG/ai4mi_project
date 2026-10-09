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
from src.registration import (check_pair, cyclic_pairs, estimate_transform,
                              expected_manifest, fold_patients, fold_plan, fold_root,
                              parameters, processed_root, save_transform, synthetic_slice_files,
                              warp_pair)


IMPROVED = REPO / "configs" / "full_cv4_enet_dice_ce_registration.yaml"
RESENC = REPO / "configs" / "full_cv4_resenc_ds_unet_dice_ce_registration.yaml"


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
        for config in (IMPROVED, RESENC):
            a = run_configs(load_config(config), fold=0)[0]
            b = run_configs(load_config(config, ["seed=2"]), fold=0)[0]
            self.assertEqual(fold_plan(a), fold_plan(b))
            self.assertEqual(fold_root(a), fold_root(b))
            self.assertEqual(processed_root(a), processed_root(b))

    def test_null_registration_keeps_existing_run_hash(self):
        cfg = load_config(REPO / "configs/full_cv4_enet_dice_ce.yaml")
        previous = dict(cfg)
        previous["data"] = {k: v for k, v in cfg["data"].items() if k != "registration"}
        self.assertEqual(config_hash(cfg), config_hash(previous))

    def test_both_arms_share_niftis_and_keep_reference_settings(self):
        for fold in range(4):
            improved = run_configs(load_config(IMPROVED), fold=fold)[0]
            resenc = run_configs(load_config(RESENC), fold=fold)[0]
            improved_reference = run_configs(load_config(REPO / "configs/full_cv4_enet_dice_ce.yaml"), fold=fold)[0]
            resenc_reference = run_configs(load_config(REPO / "configs/full_cv4_resenc_ds_unet_dice_ce.yaml"), fold=fold)[0]
            self.assertEqual(fold_root(improved), fold_root(resenc))
            self.assertEqual(processed_root(improved), processed_root(resenc))
            self.assertEqual(fold_plan(improved), fold_plan(resenc))
            self.assertEqual(improved["data"]["registration"], resenc["data"]["registration"])
            train, val, pairs = fold_plan(improved)
            self.assertEqual((len(train), len(val), len(pairs)), (30, 10, 30))
            self.assertFalse({p["source_patient"] for p in pairs} & set(val))
            self.assertFalse({p["target_patient"] for p in pairs} & set(val))
            for augmented, reference in ((improved, improved_reference), (resenc, resenc_reference)):
                self.assertEqual({k for k in augmented if augmented[k] != reference[k]},
                                 {"experiment", "notes", "data", "paths"})
                self.assertEqual(augmented["paths"]["metrics_category"], "registration")
                self.assertEqual({k for k in augmented["paths"]
                                  if augmented["paths"][k] != reference["paths"][k]},
                                 {"metrics_category"})
                self.assertEqual({k for k in augmented["data"]
                                  if augmented["data"][k] != reference["data"][k]},
                                 {"registration", "patient_regex"})
                for key in ("model", "loss", "optim", "scheduler", "train", "eval"):
                    self.assertEqual(augmented[key], reference[key])
                self.assertEqual(augmented["data"]["augment"], [])
                self.assertIsNotNone(re.fullmatch(
                    augmented["data"]["patient_regex"], f"Synthetic_f{fold}_00_0000"
                ))

    def test_partial_registration_fold_cannot_be_used_for_training(self):
        for config in (IMPROVED, RESENC):
            cfg = load_config(config)
            cfg["data"]["preprocess"]["source_dir"] = str(self.source)
            fold_cfg = run_configs(cfg, fold=0)[0]
            manifest = expected_manifest(fold_cfg)
            cache_key = Path(fold_cfg["data"]["root"]).name
            for entry in manifest["pairs"][:29]:
                entry["registration_status"] = "complete"
                entry["processed"][cache_key] = "complete"
            with mock.patch("src.registration.load_manifest", return_value=manifest), \
                    mock.patch("pathlib.Path.is_file", return_value=True):
                with self.assertRaisesRegex(ValueError, r"incomplete \(1/30\)"):
                    synthetic_slice_files(fold_cfg)


class RegistrationWarpTests(unittest.TestCase):
    def test_nested_composite_hdf5_round_trip_and_failed_write_cleanup(self):
        reference = sitk.Image([16, 16, 16], sitk.sitkFloat32)
        reference.SetSpacing((1.5, 2., 2.5))
        reference.SetOrigin((-8., 4., 12.))
        euler = sitk.Euler3DTransform()
        euler.SetRotation(0., 0., 0.12)
        euler.SetTranslation((1., -2., 0.5))
        rigid_wrapper = sitk.CompositeTransform(3)
        rigid_wrapper.AddTransform(euler)
        bspline = sitk.BSplineTransformInitializer(reference, [1, 1, 1])
        coefficients = list(bspline.GetParameters())
        coefficients[:len(coefficients) // 3] = [0.4] * (len(coefficients) // 3)
        bspline.SetParameters(coefficients)
        nested = sitk.CompositeTransform(3)
        nested.AddTransform(rigid_wrapper)
        nested.AddTransform(bspline)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transform.h5"
            save_transform(nested, path, reference)
            loaded = sitk.CompositeTransform(sitk.ReadTransform(str(path)))
            self.assertEqual([loaded.GetNthTransform(i).GetName() for i in range(2)],
                             ["Euler3DTransform", "BSplineTransform"])
            for index in ((0, 0, 0), (8, 8, 8), (15, 15, 15)):
                point = reference.TransformIndexToPhysicalPoint(index)
                np.testing.assert_allclose(loaded.TransformPoint(point),
                                           nested.TransformPoint(point), rtol=0, atol=1e-6)
            original = path.read_bytes()
            with mock.patch("src.registration.sitk.WriteTransform",
                            side_effect=RuntimeError("simulated incomplete export")):
                with self.assertRaisesRegex(RuntimeError, "simulated incomplete export"):
                    save_transform(nested, path, reference)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["transform.h5"])

    def test_physical_mapping_with_rotated_direction_and_different_grids(self):
        ct_array = np.zeros((8, 8, 8), np.float32)
        gt_array = np.zeros((8, 8, 8), np.uint8)
        ct_array[1, 3, 2] = 321
        gt_array[1, 3, 2] = 4
        source_ct = sitk.GetImageFromArray(ct_array)
        source_gt = sitk.GetImageFromArray(gt_array)
        direction = (0., -1., 0., 1., 0., 0., 0., 0., 1.)
        for image in (source_ct, source_gt):
            image.SetSpacing((2., 3., 4.))
            image.SetOrigin((10., -5., 30.))
            image.SetDirection(direction)
        target_ct = sitk.Image([8, 8, 8], sitk.sitkFloat32)
        target_ct.SetSpacing((1., 2., 2.5))
        target_ct.SetOrigin((3., -7., 32.5))
        translation = sitk.TranslationTransform(3, (-5., 2., -1.))

        source_point = source_ct.TransformIndexToPhysicalPoint((2, 3, 1))
        target_point = target_ct.TransformIndexToPhysicalPoint((3, 2, 1))
        np.testing.assert_allclose(translation.TransformPoint(target_point), source_point)
        ct, gt = warp_pair(source_ct, source_gt, target_ct, translation)
        for image in (ct, gt):
            self.assertEqual(image.GetSize(), target_ct.GetSize())
            self.assertEqual(image.GetSpacing(), target_ct.GetSpacing())
            self.assertEqual(image.GetOrigin(), target_ct.GetOrigin())
            self.assertEqual(image.GetDirection(), target_ct.GetDirection())
        self.assertEqual(sitk.GetArrayFromImage(ct)[1, 2, 3], 321)
        self.assertEqual(sitk.GetArrayFromImage(gt)[1, 2, 3], 4)

    def test_same_mapping_uses_linear_ct_and_nearest_neighbor_gt(self):
        ct_array = np.zeros((4, 4, 4), np.float32)
        ct_array[:, :, 1] = 100
        gt_array = np.zeros((4, 4, 4), np.uint8)
        gt_array[:, :, 0] = 1
        gt_array[:, :, 1] = 2
        source_ct = sitk.GetImageFromArray(ct_array)
        source_gt = sitk.GetImageFromArray(gt_array)
        target_ct = sitk.Image(source_ct)
        target_ct.SetOrigin((3., 0., 0.))
        # Target x=3 maps to source x=0.25; reversing this transform samples outside source.
        transform = sitk.TranslationTransform(3, (-2.75, 0., 0.))
        ct, gt = warp_pair(source_ct, source_gt, target_ct, transform)
        self.assertEqual(sitk.GetArrayFromImage(ct)[0, 0, 0], 25)
        self.assertEqual(sitk.GetArrayFromImage(gt)[0, 0, 0], 1)
        self.assertEqual(ct.GetOrigin(), gt.GetOrigin())

    def test_estimate_transform_applies_bspline_before_fixed_rigid(self):
        image = sitk.Image([24, 24, 24], sitk.sitkFloat32)
        rigid = sitk.Euler3DTransform()
        rigid.SetRotation(0., 0., np.pi / 2)
        bspline = sitk.BSplineTransformInitializer(image, [1, 1, 1])
        coefficients = list(bspline.GetParameters())
        coefficients[:len(coefficients) // 3] = [0.75] * (len(coefficients) // 3)
        bspline.SetParameters(coefficients)

        class NoOptimization:
            def SetInitialTransform(self, transform, inPlace=False):
                self.transform = transform

            def SetMovingInitialTransform(self, transform):
                self.moving_transform = transform

            def SetOptimizerAsRegularStepGradientDescent(self, **kwargs):
                pass

            def SetOptimizerScalesFromPhysicalShift(self):
                pass

            def SetOptimizerAsLBFGSB(self, **kwargs):
                pass

            def Execute(self, fixed, moving):
                return self.transform

            def GetMetricValue(self):
                return 0.

            def GetOptimizerIteration(self):
                return 0

            def GetOptimizerStopConditionDescription(self):
                return "mocked: no optimization"

        methods = [NoOptimization(), NoOptimization()]
        with mock.patch("src.registration.estimation_image", return_value=image), \
                mock.patch("src.registration.metric_mask", return_value=image), \
                mock.patch("src.registration.registration_method",
                           side_effect=methods), \
                mock.patch("src.registration.sitk.CenteredTransformInitializer",
                           return_value=rigid), \
                mock.patch("src.registration.sitk.BSplineTransformInitializer",
                           return_value=bspline):
            transform, _ = estimate_transform(image, image, parameters())
        self.assertIs(methods[1].moving_transform, rigid)
        self.assertIs(methods[1].transform, bspline)
        point = (12., 12., 12.)
        expected = rigid.TransformPoint(bspline.TransformPoint(point))
        reversed_order = bspline.TransformPoint(rigid.TransformPoint(point))
        np.testing.assert_allclose(transform.TransformPoint(point), expected)
        self.assertGreater(np.linalg.norm(np.subtract(expected, reversed_order)), 0.1)
        self.assertEqual(transform.GetNthTransform(0).GetName(), "Euler3DTransform")
        self.assertEqual(transform.GetNthTransform(1).GetName(), "BSplineTransform")

    def test_real_rigid_to_bspline_optimizer_and_resampling(self):
        z, y, x = np.mgrid[:16, :16, :16]
        ct_array = (-200 + 130 * np.exp(-((x - 5)**2 + (y - 8)**2 + (z - 7)**2) / 20)
                    + 75 * np.exp(-((x - 11)**2 + (y - 4)**2 + (z - 10)**2) / 12)).astype(np.float32)
        source_ct = sitk.GetImageFromArray(ct_array)
        gt_array = np.zeros(ct_array.shape, np.uint8)
        gt_array[3:7, 3:7, 3:7] = 1
        gt_array[8:12, 8:12, 8:12] = 2
        source_gt = sitk.GetImageFromArray(gt_array)
        target_ct = sitk.Image(source_ct)
        small = parameters({"estimation_spacing": [1., 1., 1.], "mi_bins": 16,
                            "sampling_fraction": 0.25, "shrink_factors": [1],
                            "smoothing_sigmas_mm": [0.], "rigid_iterations": 1,
                            "bspline_iterations": 1, "threads": 1})
        # This executes the metric derivative that raised the CompositeTransform
        # position-Jacobian exception in the first real smoketest.
        transform, info = estimate_transform(source_ct, target_ct, small)
        self.assertEqual(transform.GetNumberOfTransforms(), 2)
        fitted_rigid = transform.GetNthTransform(0)
        effective_rigid = fitted_rigid
        while effective_rigid.GetName() == "CompositeTransform":
            self.assertEqual(effective_rigid.GetNumberOfTransforms(), 1)
            effective_rigid = effective_rigid.GetNthTransform(0)
        self.assertEqual(effective_rigid.GetName(), "Euler3DTransform")
        self.assertEqual(transform.GetNthTransform(1).GetName(), "BSplineTransform")
        point = target_ct.TransformIndexToPhysicalPoint((8, 8, 8))
        np.testing.assert_allclose(fitted_rigid.TransformPoint(point),
                                   effective_rigid.TransformPoint(point))
        composed = transform.GetNthTransform(0).TransformPoint(
            transform.GetNthTransform(1).TransformPoint(point))
        np.testing.assert_allclose(transform.TransformPoint(point), composed)
        self.assertIn("stop", info["rigid"])
        self.assertIn("stop", info["bspline"])
        ct, gt = warp_pair(source_ct, source_gt, target_ct, transform)
        self.assertEqual(ct.GetSpacing(), gt.GetSpacing())
        self.assertEqual(ct.GetOrigin(), gt.GetOrigin())
        self.assertEqual(ct.GetDirection(), gt.GetDirection())
        self.assertTrue(np.isfinite(sitk.GetArrayFromImage(ct)).all())
        self.assertLessEqual(set(np.unique(sitk.GetArrayFromImage(gt))), {0, 1, 2})

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
