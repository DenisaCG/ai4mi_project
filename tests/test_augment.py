"""Online augmentation geometry, labels, noise, and train-only loading."""

import tempfile
import unittest
from unittest.mock import patch

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import affine

from dataset import SliceDataset
from src.augment import (build_gaussian_blur, build_gaussian_noise, build_random_brightness_contrast,
                         build_random_elastic, build_random_gamma, build_random_rotation, build_random_scaling, build_random_shear,
                         build_random_shift)
from src.data import Augmented, build_dataset


class AugmentTests(unittest.TestCase):
    def setUp(self):
        self.image = torch.full((1, 32, 32), -5.0)
        self.gt = torch.zeros((5, 32, 32), dtype=torch.int32)
        self.gt[0] = 1
        self.gt[0, 7:18, 12:20] = 0
        self.gt[1, 7:18, 12:20] = 1

    def test_geometric_transforms_share_geometry_and_preserve_one_hot(self):
        for factory, kwargs, sampled in (
            (build_random_rotation, {"degrees": [-10.0, 10.0], "fill": -5.0}, 10.0),
            (build_random_scaling, {"scales": [0.9, 1.1], "fill": -5.0}, 0.9),
        ):
            with self.subTest(factory=factory.__name__):
                with patch("src.augment.random.random", return_value=0.0), \
                     patch("src.augment.random.uniform", return_value=sampled), \
                     patch("src.augment.affine", wraps=affine) as warp:
                    image, gt = factory(p=1.0, **kwargs)(self.image, self.gt)
                self.assertEqual(warp.call_count, 2)
                ct_call, gt_call = warp.call_args_list
                for key in ("angle", "translate", "scale", "shear"):
                    self.assertEqual(ct_call.kwargs[key], gt_call.kwargs[key])
                self.assertEqual(ct_call.kwargs["interpolation"], InterpolationMode.BILINEAR)
                self.assertEqual(ct_call.kwargs["fill"], [-5.0])
                self.assertEqual(gt_call.kwargs["interpolation"], InterpolationMode.NEAREST)
                self.assertEqual(image.shape, self.image.shape)
                self.assertEqual(gt.shape, self.gt.shape)
                self.assertEqual(gt.dtype, self.gt.dtype)
                self.assertTrue(torch.all((gt == 0) | (gt == 1)))
                self.assertTrue(torch.all(gt.sum(dim=0) == 1))

    def test_gaussian_noise_changes_only_image(self):
        with patch("src.augment.random.random", return_value=0.0), \
             patch("src.augment.torch.randn_like", return_value=torch.ones_like(self.image)):
            image, gt = build_gaussian_noise(p=1.0, sigma=0.25)(self.image, self.gt)
        self.assertTrue(torch.equal(image, self.image + 0.25))
        self.assertIs(gt, self.gt)

    def test_shift_and_shear_keep_one_hot_labels(self):
        for factory, kwargs in ((build_random_shift, {"max_fraction": 0.1}), (build_random_shear, {"degrees": [-5.0, 5.0]})):
            with self.subTest(factory=factory.__name__):
                image, gt = factory(p=1.0, fill=-5.0, **kwargs)(self.image, self.gt)
                self.assertEqual(image.shape, self.image.shape)
                self.assertEqual(gt.dtype, self.gt.dtype)
                self.assertTrue(torch.all(gt.sum(dim=0) == 1))

    def test_elastic_keeps_one_hot_labels_and_moves_image(self):
        ramp = torch.linspace(-2, 3, 32 * 32).reshape(1, 32, 32)
        image, gt = build_random_elastic(p=1.0, alpha=50.0, sigma=5.0, fill=-5.0)(ramp, self.gt)
        self.assertEqual(image.shape, ramp.shape)
        self.assertEqual(gt.dtype, self.gt.dtype)
        self.assertTrue(torch.all((gt == 0) | (gt == 1)))
        self.assertTrue(torch.all(gt.sum(dim=0) == 1))
        self.assertFalse(torch.equal(image, ramp))
        with patch("src.augment.random.random", return_value=0.99):
            same_image, same_gt = build_random_elastic(p=0.5, alpha=50.0, sigma=5.0, fill=-5.0)(ramp, self.gt)
        self.assertIs(same_image, ramp)
        self.assertIs(same_gt, self.gt)

    def test_intensity_transforms_change_only_image(self):
        ramp = torch.linspace(-2, 3, 32 * 32).reshape(1, 32, 32)
        for factory, kwargs in ((build_random_gamma, {"gamma": [0.7, 1.5]}),
                                (build_random_brightness_contrast, {"factor": [0.75, 1.25]}),
                                (build_gaussian_blur, {"sigma": [0.5, 1.0]})):
            with self.subTest(factory=factory.__name__):
                image, gt = factory(p=1.0, **kwargs)(ramp, self.gt)
                self.assertEqual(image.shape, ramp.shape)
                self.assertIs(gt, self.gt)
                self.assertFalse(torch.equal(image, ramp))
                self.assertTrue(torch.isfinite(image).all())
        image, _ = build_random_gamma(p=1.0, gamma=[1.0, 1.0])(ramp, self.gt)
        self.assertTrue(torch.allclose(image, ramp, atol=1e-5))

    def test_unset_experiment_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            build_random_rotation(p=None, degrees=[-10, 10], fill=-5.0)
        with self.assertRaises(ValueError):
            build_gaussian_noise(p=1.0, sigma=None)

    def test_only_training_dataset_is_wrapped(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = {"seed": 0, "train": {"debug_samples": 0}, "data": {
                "root": root, "preprocess": None, "num_classes": 5, "label_scale": 63,
                "augment": [{"name": "random_rotation", "kwargs": {"p": 1.0, "degrees": [-10, 10], "fill": -5.0}}],
            }}
            self.assertIsInstance(build_dataset(cfg, "train"), Augmented)
            self.assertIsInstance(build_dataset(cfg, "val"), SliceDataset)
            self.assertIsInstance(build_dataset(cfg, "test"), SliceDataset)


if __name__ == "__main__":
    unittest.main()
