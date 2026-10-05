import pickle
import unittest

import torch
from torch import nn

import src.models  # noqa: F401  registers models
from src.models.unet import UNet
from src.registry import build

K = 5


class UNetTests(unittest.TestCase):
    def test_output_shape_matches_input(self):
        for channels in (1, 3):
            net = build(
                "model", "unet", in_channels=channels, num_classes=K, base_features=4
            )
            for size in (288, 256):
                with torch.no_grad():
                    out = net(torch.randn(1, channels, size, size))
                self.assertEqual(out.shape, (1, K, size, size))

    def test_registered_with_defaults(self):
        net = build("model", "unet", in_channels=1, num_classes=K)
        self.assertIsInstance(net, UNet)

    def test_feature_widths_per_stage(self):
        net = build("model", "unet", in_channels=1, num_classes=K)
        self.assertEqual(net.features, [32, 64, 128, 256, 512, 512])
        widths = [stage[1][0].out_channels for stage in net.encoder]
        self.assertEqual(widths, [32, 64, 128, 256, 512, 512])

    def test_downsampling_is_strided_conv(self):
        net = build("model", "unet", in_channels=1, num_classes=K)
        strides = [stage[0][0].stride for stage in net.encoder]
        self.assertEqual(strides, [(1, 1)] + [(2, 2)] * 5)
        self.assertFalse(any(isinstance(m, nn.MaxPool2d) for m in net.modules()))

    def test_non_divisible_input_raises(self):
        net = build("model", "unet", in_channels=1, num_classes=K, base_features=4)
        with self.assertRaisesRegex(ValueError, "divisible by 32"):
            net(torch.randn(1, 1, 250, 256))

    def test_pickle_roundtrip(self):
        net = build(
            "model", "unet", in_channels=1, num_classes=K, base_features=4
        ).eval()
        clone = pickle.loads(pickle.dumps(net)).eval()
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            self.assertTrue(torch.equal(net(x), clone(x)))


if __name__ == "__main__":
    unittest.main()
