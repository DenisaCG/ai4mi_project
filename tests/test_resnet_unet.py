"""Tests for the ResNet34 U-Net: shapes, deep supervision, pretrained weights and the arm configs."""

import os
import tempfile
import unittest
from unittest import mock

import torch

import src.models  # noqa: F401  registers models
from src.config import REPO, load_config
from src.models.resnet_unet import WEIGHTS, WIDTHS, ResNetUNet, weights_path
from src.registry import build

K = 5
CACHED = weights_path().is_file()
ARMS = ("pretrained", "scratch")


def build_net(**kwargs) -> ResNetUNet:
    """Builds the ResNet34 U-Net with one input channel and `K` classes."""
    return build("model", "resnet34_unet", in_channels=1, num_classes=K, **kwargs)


class ResNetUNetTests(unittest.TestCase):
    """Architecture checks that do not need the pretrained weights."""

    def test_output_shape_matches_input(self):
        net = build_net(pretrained=False).eval()
        for size in (256, 288):
            with torch.no_grad():
                out = net(torch.randn(1, 1, size, size))
            self.assertEqual(out.shape, (1, K, size, size))

    def test_encoder_exposes_every_scale_including_full_resolution(self):
        net = build_net(pretrained=False).eval()
        with torch.no_grad():
            maps = net.encode(torch.randn(1, 1, 256, 256))
        self.assertEqual([m.shape[1] for m in maps], WIDTHS)
        self.assertEqual([m.shape[-1] for m in maps], [256, 128, 64, 32, 16, 8])

    def test_deep_supervision_returns_full_resolution_first(self):
        net = build_net(pretrained=False, deep_supervision=True).train()
        outs = net(torch.randn(2, 1, 256, 256))
        self.assertEqual([o.shape[-1] for o in outs], [256, 128, 64, 32, 16])
        self.assertEqual(outs[0].shape, (2, K, 256, 256))

    def test_non_divisible_input_raises(self):
        with self.assertRaisesRegex(ValueError, "divisible by 32"):
            build_net(pretrained=False)(torch.randn(1, 1, 250, 256))

    def test_multi_channel_input_raises(self):
        with self.assertRaisesRegex(ValueError, "in_channels must be 1"):
            build("model", "resnet34_unet", in_channels=3, num_classes=K)

    def test_missing_pretrained_weights_fail_loudly(self):
        with (
            tempfile.TemporaryDirectory() as empty,
            mock.patch.dict(os.environ, {"TORCH_HOME": empty}),
        ):
            with self.assertRaisesRegex(FileNotFoundError, WEIGHTS):
                build_net(pretrained=True)
            build_net(pretrained=False)

    def test_random_init_needs_no_weights_cache(self):
        """Evaluation builds with pretrained=False and loads a checkpoint, so no cache may be needed."""
        source = build_net(pretrained=False)
        with (
            tempfile.TemporaryDirectory() as empty,
            mock.patch.dict(os.environ, {"TORCH_HOME": empty}),
        ):
            self.assertFalse(weights_path().exists())
            net = build_net(pretrained=False)
            net.load_state_dict(source.state_dict())
        for a, b in zip(net.state_dict().values(), source.state_dict().values()):
            self.assertTrue(torch.equal(a, b))


@unittest.skipUnless(CACHED, f"ResNet34 weights not cached at {weights_path()}")
class PretrainedWeightsTests(unittest.TestCase):
    """Checks against the cached torchvision ImageNet weights."""

    @classmethod
    def setUpClass(cls):
        cls.source = torch.load(weights_path(), weights_only=True)

    def test_pretrained_parameters_equal_the_source_weights(self):
        net = build_net(pretrained=True)
        stem, layer1 = net.encoder[1], net.encoder[2]
        self.assertTrue(torch.equal(stem[0].weight, self.source["conv1.weight"]))
        self.assertTrue(
            torch.equal(layer1[1][0].conv1.weight, self.source["layer1.0.conv1.weight"])
        )
        self.assertTrue(
            torch.equal(
                net.encoder[5][2].bn2.running_var,
                self.source["layer4.2.bn2.running_var"],
            )
        )

    def test_random_init_differs_from_the_source_weights(self):
        net = build_net(pretrained=False)
        self.assertFalse(
            torch.equal(net.encoder[1][0].weight, self.source["conv1.weight"])
        )


class ArmConfigTests(unittest.TestCase):
    """The two arms differ only in the pretrained flag and the name."""

    def test_arms_differ_only_in_pretrained(self):
        cfgs = {
            a: load_config(
                REPO / "configs" / f"full_cv4_resnet34_{a}_unet_dice_ce.yaml"
            )
            for a in ARMS
        }
        for arm, cfg in cfgs.items():
            self.assertEqual(cfg["experiment"], f"full_cv4_resnet34_{arm}_unet_dice_ce")
            self.assertEqual(cfg["model"]["kwargs"]["pretrained"], arm == "pretrained")
        a, b = (dict(c, experiment=None, notes=None) for c in cfgs.values())
        a["model"] = dict(
            a["model"], kwargs={**a["model"]["kwargs"], "pretrained": None}
        )
        b["model"] = dict(
            b["model"], kwargs={**b["model"]["kwargs"], "pretrained": None}
        )
        self.assertEqual(a, b)

    def test_recipe_matches_the_residual_encoder_baseline(self):
        base = load_config(REPO / "configs" / "full_cv4_resenc_ds_unet_dice_ce.yaml")
        cfg = load_config(
            REPO / "configs" / "full_cv4_resnet34_pretrained_unet_dice_ce.yaml"
        )
        for key in ("seed", "data", "loss", "optim", "scheduler", "train", "eval"):
            self.assertEqual(cfg[key], base[key])


if __name__ == "__main__":
    unittest.main()
