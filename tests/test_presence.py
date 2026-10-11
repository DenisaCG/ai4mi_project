"""Slice-presence head: targets, loss, gating, confusion counts, the U-Net head and the config contract."""

import unittest
from pathlib import Path

import numpy as np
import torch

import src.models  # noqa: F401  registers models
from src.config import config_hash, load_config
from src.presence import (
    gate_prediction,
    presence_counts,
    presence_loss,
    presence_target,
)
from src.registry import build
from utils import class2one_hot

K = 5
CFG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "full_cv4_resenc_ds_unet_dice_ce_presence.yaml"
)
BASELINE_CFG = CFG.with_name("full_cv4_resenc_ds_unet_dice_ce.yaml")


def onehot(labels: list[np.ndarray]) -> torch.Tensor:
    return class2one_hot(torch.tensor(np.stack(labels), dtype=torch.int64), K)


class TargetTests(unittest.TestCase):
    def test_one_voxel_is_enough_and_background_is_ignored(self):
        a = np.zeros((8, 8), dtype=np.int64)  # background only
        b = a.copy()
        b[0, 0], b[7, 7] = 1, 4  # a single voxel of esophagus and of aorta
        c = a.copy()
        c[2:5, 2:5] = 2
        target = presence_target(onehot([a, b, c]))
        self.assertEqual(target.shape, (3, K - 1))
        self.assertEqual(
            target.tolist(),
            [[0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 0, 0]],
        )

    def test_loss_is_zero_for_confident_correct_logits(self):
        a = np.zeros((4, 4), dtype=np.int64)
        a[0, 0] = 3
        logits = torch.tensor([[-30.0, -30.0, 30.0, -30.0]])
        self.assertLess(presence_loss(logits, onehot([a])).item(), 1e-6)
        self.assertGreater(presence_loss(-logits, onehot([a])).item(), 10)


class GateTests(unittest.TestCase):
    def test_organs_below_threshold_become_background(self):
        pred = np.arange(K, dtype=np.uint8).repeat(2).reshape(2, 5)
        gated = gate_prediction(pred, np.array([0.9, 0.49, 0.5, 0.1]))
        self.assertEqual(gated[0].tolist(), [0, 0, 1, 1, 0])
        self.assertEqual(gated[1].tolist(), [0, 3, 3, 0, 0])
        self.assertEqual(pred[0, 3], 1)  # the input is not modified

    def test_counts(self):
        prob = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.7]])
        target = np.array([[1, 0], [0, 0], [1, 1]])
        counts = presence_counts(prob, target)
        self.assertEqual(
            counts.tolist(), [[1, 1, 1, 0], [1, 0, 0, 2]]
        )  # tp, fp, fn, tn per organ


class HeadTests(unittest.TestCase):
    def test_head_logits_and_unchanged_segmentation_shapes(self):
        net = build(
            "model",
            "unet",
            in_channels=1,
            num_classes=K,
            base_features=4,
            slice_presence=True,
        )
        x = torch.randn(2, 1, 64, 64)
        out = net(x)
        self.assertEqual(out.shape, (2, K, 64, 64))
        self.assertEqual(net.presence_logits.shape, (2, K - 1))

    def test_no_head_by_default(self):
        net = build("model", "unet", in_channels=1, num_classes=K, base_features=4)
        net(torch.randn(1, 1, 64, 64))
        self.assertIsNone(net.presence)
        self.assertIsNone(net.presence_logits)

    def test_loss_reaches_the_head_and_the_bottleneck(self):
        net = build(
            "model",
            "unet",
            in_channels=1,
            num_classes=K,
            base_features=4,
            slice_presence=True,
        )
        net(torch.randn(2, 1, 64, 64))
        labels = np.zeros((2, 64, 64), dtype=np.int64)
        labels[0, :4, :4] = 2
        presence_loss(net.presence_logits, onehot(list(labels))).backward()
        self.assertGreater(net.presence.weight.grad.abs().sum().item(), 0)
        self.assertGreater(net.encoder[-1][0][0].weight.grad.abs().sum().item(), 0)


class ConfigTests(unittest.TestCase):
    def test_weight_and_head_go_together(self):
        load_config(CFG, ["seed=0"])
        with self.assertRaises(ValueError):
            load_config(CFG, ["train.presence_weight=0"])
        with self.assertRaises(ValueError):
            load_config(BASELINE_CFG, ["train.presence_weight=0.5"])

    def test_hash_of_runs_without_the_head_is_unchanged(self):
        base = load_config(BASELINE_CFG, ["seed=0"])
        self.assertNotIn("presence_weight", {k for k, v in base["train"].items() if v})
        self.assertNotEqual(
            config_hash(base), config_hash(load_config(CFG, ["seed=0"]))
        )


if __name__ == "__main__":
    unittest.main()
