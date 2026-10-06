"""2.5D input: SliceStack channel order, edge clamping, patient isolation, subsampling, ENet widths, config rules."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ENet import ENet
from src.config import config_hash, load_config
from src.data import SliceStack, build_dataset

CFG = Path(__file__).resolve().parents[1] / "configs" / "segthor_enet_dice_ce_all_corrected_ct_window_zscore_median_spacing_roi_crop.yaml"
BASELINE_HASH = "738be6904c56"  # this config's hash before data.context existed (the older manifest hash predates later config keys)
PATIENTS, SLICES = (1, 2), 4


def make_root(root: Path) -> None:
    """Patient p, slice z: image pixel value 10*p + z (constant), gt class (z % 4) + 1 everywhere (label_scale 63)."""
    for split in ("train", "val"):
        for sub in ("img", "gt"):
            (root / split / sub).mkdir(parents=True)
        for p in PATIENTS:
            for z in range(SLICES):
                stem = f"Patient_{p:02d}_{z:04d}"
                Image.fromarray(np.full((8, 8), 10 * p + z, np.uint8)).save(root / split / "img" / f"{stem}.png")
                Image.fromarray(np.full((8, 8), 63 * (z % 4 + 1), np.uint8)).save(root / split / "gt" / f"{stem}.png")


def cfg_for(root: Path, context: int, **train) -> dict:
    overrides = [f"data.root={root}", "data.preprocess=null", f"data.context={context}", f"data.in_channels={2 * context + 1}",
                 "data.num_workers=0", "data.augment=[]"] + [f"train.{k}={v}" for k, v in train.items()]
    cfg = load_config(CFG, overrides)
    return cfg


def pixel(item, channel: int) -> int:
    return round(float(item["images"][channel, 0, 0]) * 255)


class SliceStackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        make_root(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def stack(self, context=1, **train):
        return build_dataset(cfg_for(self.root, context, **train), "val")

    def item(self, ds, patient, z):
        return next(ds[i] for i in range(len(ds)) if ds[i]["stems"] == f"Patient_{patient:02d}_{z:04d}")

    def test_channel_order_and_shape(self):
        item = self.item(self.stack(), 1, 1)
        self.assertEqual(tuple(item["images"].shape), (3, 8, 8))
        self.assertEqual([pixel(item, c) for c in range(3)], [10, 11, 12])

    def test_edges_repeat_the_edge_slice(self):
        ds = self.stack()
        self.assertEqual([pixel(self.item(ds, 1, 0), c) for c in range(3)], [10, 10, 11])
        self.assertEqual([pixel(self.item(ds, 1, SLICES - 1), c) for c in range(3)], [12, 13, 13])

    def test_no_leakage_across_patients(self):
        ds = self.stack(context=2)
        for p in PATIENTS:
            for z in range(SLICES):
                vals = [pixel(self.item(ds, p, z), c) for c in range(5)]
                self.assertTrue(all(v // 10 == p for v in vals), (p, z, vals))

    def test_gt_is_the_centre_slice(self):
        item = self.item(self.stack(), 2, 2)
        self.assertEqual(tuple(item["gts"].shape), (5, 8, 8))
        self.assertEqual(int(item["gts"].argmax(dim=0)[0, 0]), 2 % 4 + 1)

    def test_debug_samples_still_finds_neighbours(self):
        ds = self.stack(debug_samples=3)
        self.assertEqual(len(ds), 3)
        for i in range(3):
            item = ds[i]
            z = int(item["stems"][-4:])
            self.assertEqual([pixel(item, c) for c in range(3)],
                             [10 * int(item["stems"][8:10]) + min(max(z + d, 0), SLICES - 1) for d in (-1, 0, 1)])

    def test_context_zero_is_the_plain_dataset(self):
        ds = self.stack(context=0)
        self.assertNotIsInstance(ds, SliceStack)
        self.assertEqual(tuple(ds[0]["images"].shape), (1, 8, 8))

    def test_rotation_augmentation_handles_all_channels(self):
        from src.augment import build_random_rotation
        image, gt = torch.randn(3, 16, 16), torch.zeros(5, 16, 16)
        gt[0] = 1
        out, _ = build_random_rotation(p=1.0, degrees=[5.0, 5.0], fill=-1.0)(image, gt)
        self.assertEqual(tuple(out.shape), (3, 16, 16))


class EnetAndConfigTests(unittest.TestCase):
    def test_enet_forward_for_several_input_widths(self):
        for c in (1, 3, 5):
            net = ENet(c, 5, kernels=8, factor=2).eval()
            self.assertEqual(tuple(net(torch.randn(2, c, 64, 64)).shape), (2, 5, 64, 64), c)

    def test_enet_one_channel_parameters_unchanged(self):
        net = ENet(1, 5, kernels=8, factor=2)
        self.assertEqual(tuple(net.conv0.weight.shape), (7, 1, 3, 3))  # K - 1 as before

    def test_enet_rejects_in_dim_not_below_kernels(self):
        with self.assertRaisesRegex(AssertionError, "kernels"):
            ENet(8, 5, kernels=8, factor=2)

    def test_validate_rejects_mismatched_channels(self):
        for over in (["data.context=1"], ["data.in_channels=3"], ["data.context=1", "data.in_channels=2"],
                     ["data.context=-1", "data.in_channels=-1"], ["data.context=1.0", "data.in_channels=3"]):
            with self.assertRaises(ValueError, msg=over):
                load_config(CFG, over)
        load_config(CFG, ["data.context=2", "data.in_channels=5"])

    def test_config_hash_unchanged_for_context_zero(self):
        self.assertEqual(config_hash(load_config(CFG, ["seed=0"])), BASELINE_HASH)
        self.assertNotEqual(config_hash(load_config(CFG, ["seed=0", "data.context=1", "data.in_channels=3"])), BASELINE_HASH)


if __name__ == "__main__":
    unittest.main()
