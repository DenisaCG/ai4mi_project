import pickle
import unittest

import torch

import src.models  # noqa: F401  registers models
from src.models.dino_unet import WEIGHTS, DinoUNet, cache_dir
from src.registry import build

K = 5
CACHED = (cache_dir() / "hub" / "checkpoints" / WEIGHTS).is_file()


@unittest.skipUnless(
    CACHED, f"DINOv2 weights not cached in {cache_dir()}; see src/models/dino_unet.py"
)
class DinoUNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nets = {
            c: build("model", "dino_unet", in_channels=c, num_classes=K).eval()
            for c in (1, 3)
        }

    def test_output_shape_matches_input(self):
        for channels, net in self.nets.items():
            for size in (288, 320):
                with torch.no_grad():
                    out = net(torch.randn(1, channels, size, size))
                self.assertEqual(out.shape, (1, K, size, size))

    def test_without_stem_output_shape_and_no_stem_weights(self):
        net = build(
            "model", "dino_unet", in_channels=1, num_classes=K, stem=False
        ).eval()
        with torch.no_grad():
            out = net(torch.randn(1, 1, 288, 320))
        self.assertEqual(out.shape, (1, K, 288, 320))
        self.assertIsNone(net.stem)
        self.assertEqual(len(net.decoder), 3)
        self.assertLess(
            sum(p.numel() for p in net.parameters() if p.requires_grad),
            sum(p.numel() for p in self.nets[1].parameters() if p.requires_grad),
        )

    def test_registered(self):
        self.assertIsInstance(self.nets[1], DinoUNet)

    def test_non_divisible_input_raises(self):
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            self.nets[1](torch.randn(1, 1, 250, 256))

    def test_encoder_is_frozen_and_stays_in_eval_mode(self):
        net = self.nets[1]
        self.assertFalse(any(p.requires_grad for p in net.encoder.parameters()))
        net.train()
        self.assertTrue(net.training)
        self.assertFalse(net.encoder.training)
        self.assertFalse(any(m.training for m in net.encoder.modules()))
        net.eval()

    def test_gradients_reach_the_input_conv_but_not_the_encoder(self):
        net = self.nets[1].train()
        net(torch.randn(1, 1, 64, 64)).sum().backward()
        self.assertGreater(net.to_rgb.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in net.encoder.parameters()))
        net.eval()

    def test_pickle_roundtrip(self):
        net = self.nets[1]
        clone = pickle.loads(pickle.dumps(net)).eval()
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            self.assertTrue(torch.equal(net(x), clone(x)))
