import pickle
import unittest

import torch
from torch import nn

import src.models  # noqa: F401  registers models
from src.models.unet import RESIDUAL_BLOCKS, AttentionGate, ResidualBlock, UNet
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


def residual_net(**kwargs) -> UNet:
    return build(
        "model", "unet", in_channels=1, num_classes=K, block="residual", **kwargs
    )


class ResidualEncoderTests(unittest.TestCase):
    def test_output_shape_matches_input(self):
        for channels in (1, 3):
            net = build(
                "model",
                "unet",
                in_channels=channels,
                num_classes=K,
                base_features=4,
                block="residual",
            )
            with torch.no_grad():
                self.assertEqual(
                    net(torch.randn(1, channels, 64, 64)).shape, (1, K, 64, 64)
                )

    def test_residual_blocks_per_encoder_stage(self):
        net = residual_net(base_features=4)
        counts = [
            sum(isinstance(m, ResidualBlock) for m in stage.modules())
            for stage in net.encoder
        ]
        self.assertEqual(counts, [1, 3, 4, 6, 6, 6])
        self.assertEqual(tuple(counts), RESIDUAL_BLOCKS)

    def test_stem_conv_precedes_the_first_stage(self):
        net = build(
            "model",
            "unet",
            in_channels=3,
            num_classes=K,
            base_features=4,
            block="residual",
        )
        stem, blocks = net.encoder[0]
        self.assertEqual(
            (stem[0].in_channels, stem[0].out_channels, stem[0].stride), (3, 4, (1, 1))
        )
        self.assertIsInstance(blocks[0], ResidualBlock)

    def test_one_conv_per_decoder_stage(self):
        net = residual_net(base_features=4)
        for stage in net.decoder:
            self.assertEqual(sum(isinstance(m, nn.Conv2d) for m in stage.modules()), 1)

    def test_skip_is_pool_conv_norm_only_when_the_shape_changes(self):
        self.assertIsInstance(ResidualBlock(8, 8).skip, nn.Identity)
        skip = ResidualBlock(8, 16, stride=2).skip
        self.assertEqual(
            [type(m) for m in skip], [nn.AvgPool2d, nn.Conv2d, nn.InstanceNorm2d]
        )
        self.assertEqual(skip[1].kernel_size, (1, 1))
        self.assertEqual(
            [type(m) for m in ResidualBlock(8, 16).skip], [nn.Conv2d, nn.InstanceNorm2d]
        )

    def test_block_adds_the_skip_before_the_activation(self):
        block = ResidualBlock(4, 4)
        for p in block.conv2.parameters():
            nn.init.zeros_(p)
        x = torch.randn(1, 4, 8, 8)
        expected = nn.functional.leaky_relu(x, 0.01)
        with torch.no_grad():
            self.assertTrue(torch.allclose(block(x.clone()), expected))

    def test_unknown_block_raises(self):
        with self.assertRaisesRegex(ValueError, "block"):
            build("model", "unet", in_channels=1, num_classes=K, block="dense")

    def test_pickle_roundtrip(self):
        net = residual_net(base_features=4).eval()
        clone = pickle.loads(pickle.dumps(net)).eval()
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            self.assertTrue(torch.equal(net(x), clone(x)))


class AttentionGateTests(unittest.TestCase):
    def test_gate_scales_the_skip_by_a_sigmoid_coefficient(self):
        gate = AttentionGate(4, 4, 2)
        for p in gate.psi.parameters():
            nn.init.zeros_(p)  # sigmoid(0) = 0.5 everywhere
        skip = torch.randn(1, 4, 8, 8)
        with torch.no_grad():
            self.assertTrue(
                torch.allclose(gate(skip, torch.randn(1, 4, 8, 8)), 0.5 * skip)
            )

    def test_gate_is_built_from_1x1_convs(self):
        gate = AttentionGate(8, 8, 4)
        for conv in (gate.w_x, gate.w_g, gate.psi):
            self.assertEqual(conv.kernel_size, (1, 1))
        self.assertEqual(
            (gate.w_x.out_channels, gate.w_g.out_channels, gate.psi.out_channels),
            (4, 4, 1),
        )

    def test_one_gate_per_skip_connection(self):
        for block in ("plain", "residual"):
            net = build(
                "model",
                "unet",
                in_channels=1,
                num_classes=K,
                base_features=4,
                block=block,
                attention=True,
            )
            self.assertEqual(len(net.gates), len(net.features) - 1)
            with torch.no_grad():
                self.assertEqual(net(torch.randn(1, 1, 64, 64)).shape, (1, K, 64, 64))

    def test_off_by_default(self):
        net = build("model", "unet", in_channels=1, num_classes=K, base_features=4)
        self.assertIsNone(net.gates)
        self.assertFalse(any(isinstance(m, AttentionGate) for m in net.modules()))

    def test_pickle_roundtrip(self):
        net = build(
            "model",
            "unet",
            in_channels=1,
            num_classes=K,
            base_features=4,
            attention=True,
        ).eval()
        clone = pickle.loads(pickle.dumps(net)).eval()
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            self.assertTrue(torch.equal(net(x), clone(x)))


if __name__ == "__main__":
    unittest.main()
