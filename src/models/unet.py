"""Plain 2D U-Net following the nnU-Net v2 2D template: strided-conv downsampling,
transposed-conv upsampling, instance norm and leaky ReLU, no residuals or deep supervision."""

import torch
from torch import nn

from src.registry import register


def conv_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    """Conv 3x3 -> instance norm -> leaky ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=True),
        nn.InstanceNorm2d(out_channels, eps=1e-5, affine=True),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
    )


def double_conv(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        conv_block(in_channels, out_channels, stride),
        conv_block(out_channels, out_channels),
    )


class UNet(nn.Module):
    """U-Net with `n_stages` stages; input height and width must be divisible by 2 ** (n_stages - 1).

    Args:
        in_channels: Number of input channels (stacked slices for 2.5D input).
        num_classes: Number of output classes.
        n_stages: Number of encoder stages, including the bottleneck.
        base_features: Feature maps at stage 0, doubled per stage.
        max_features: Upper bound on the feature maps of any stage.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        n_stages: int = 6,
        base_features: int = 32,
        max_features: int = 512,
    ):
        super().__init__()
        self.multiple = 2 ** (n_stages - 1)
        self.features = [
            min(base_features * 2**i, max_features) for i in range(n_stages)
        ]
        self.encoder = nn.ModuleList(
            double_conv(
                in_channels if i == 0 else self.features[i - 1],
                f,
                stride=1 if i == 0 else 2,
            )
            for i, f in enumerate(self.features)
        )
        skips = self.features[:-1]
        self.upsample = nn.ModuleList(
            nn.ConvTranspose2d(below, f, kernel_size=2, stride=2)
            for below, f in zip(self.features[1:], skips)
        )
        self.decoder = nn.ModuleList(double_conv(2 * f, f) for f in skips)
        self.head = nn.Conv2d(self.features[0], num_classes, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, a=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Maps (B, in_channels, H, W) to logits (B, num_classes, H, W)."""
        if x.shape[-2] % self.multiple or x.shape[-1] % self.multiple:
            raise ValueError(
                f"input size {tuple(x.shape[-2:])} must be divisible by {self.multiple} "
                f"(2 ** (n_stages - 1) with n_stages={len(self.features)})"
            )
        skips = []
        for stage in self.encoder:
            x = stage(x)
            skips.append(x)
        skips.pop()  # the bottleneck output feeds the decoder directly
        for up, stage in zip(reversed(self.upsample), reversed(self.decoder)):
            x = stage(torch.cat([skips.pop(), up(x)], dim=1))
        return self.head(x)


@register("model", "unet")
def build_unet(
    in_channels: int,
    num_classes: int,
    n_stages: int = 6,
    base_features: int = 32,
    max_features: int = 512,
) -> nn.Module:
    """Builds the plain U-Net; see `UNet`."""
    return UNet(
        in_channels,
        num_classes,
        n_stages=n_stages,
        base_features=base_features,
        max_features=max_features,
    )
