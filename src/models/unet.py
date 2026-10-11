"""2D U-Net following the nnU-Net v2 2D template: strided-conv downsampling, transposed-conv
upsampling, instance norm and leaky ReLU. Plain double-conv stages by default; `block="residual"`
switches to the residual encoder."""

import torch
import torch.nn.functional as F
from torch import nn

from src.registry import register

RESIDUAL_BLOCKS = (
    1,
    3,
    4,
    6,
    6,
    6,
)  # residual blocks per encoder stage; the last count repeats for deeper nets


def norm(channels: int) -> nn.InstanceNorm2d:
    return nn.InstanceNorm2d(channels, eps=1e-5, affine=True)


def act() -> nn.LeakyReLU:
    return nn.LeakyReLU(negative_slope=0.01, inplace=True)


def conv_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    """Conv 3x3 -> instance norm -> leaky ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=True),
        norm(out_channels),
        act(),
    )


def double_conv(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        conv_block(in_channels, out_channels, stride),
        conv_block(out_channels, out_channels),
    )


class ResidualBlock(nn.Module):
    """Conv-norm-act, conv-norm, plus a skip connection, then act.

    When the stride or the channel count changes, the skip is average pooling, a 1x1 conv and a norm
    (ResNet-D, arXiv:1812.01187); otherwise it is the identity.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = conv_block(in_channels, out_channels, stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True),
            norm(out_channels),
        )
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            pool = [nn.AvgPool2d(stride)] if stride > 1 else []
            self.skip = nn.Sequential(
                *pool,
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                norm(out_channels),
            )
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


def residual_stage(
    in_channels: int, out_channels: int, stride: int, n_blocks: int
) -> nn.Sequential:
    first = ResidualBlock(in_channels, out_channels, stride)
    return nn.Sequential(
        first, *(ResidualBlock(out_channels, out_channels) for _ in range(n_blocks - 1))
    )


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al. 2018, arXiv:1804.03999) on a skip connection.

    The skip and the gating signal go through 1x1 convs, are added, passed through ReLU, a 1x1 conv to
    one channel and a sigmoid; the result scales the skip per pixel. Here the gating signal is the
    upsampled decoder feature, so both inputs already have the same height and width.
    """

    def __init__(self, skip_channels: int, gate_channels: int, inter_channels: int):
        super().__init__()
        self.w_x = nn.Conv2d(skip_channels, inter_channels, 1)
        self.w_g = nn.Conv2d(gate_channels, inter_channels, 1)
        self.psi = nn.Conv2d(inter_channels, 1, 1)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return skip * torch.sigmoid(self.psi(F.relu(self.w_x(skip) + self.w_g(gate))))


class UNet(nn.Module):
    """U-Net with `n_stages` stages; input height and width must be divisible by 2 ** (n_stages - 1).

    Args:
        in_channels: Number of input channels (stacked slices for 2.5D input).
        num_classes: Number of output classes.
        n_stages: Number of encoder stages, including the bottleneck.
        base_features: Feature maps at stage 0, doubled per stage.
        max_features: Upper bound on the feature maps of any stage.
        block: "plain" for two convs per stage in the encoder and the decoder, or "residual" for a stem
            conv, `RESIDUAL_BLOCKS` residual blocks per encoder stage and one conv per decoder stage.
        attention: Whether to put an attention gate on every skip connection.
        deep_supervision: Whether, in train mode, to return the logits of every decoder stage as a list
            (full resolution first, then half, a quarter, ...). Eval mode returns the full-resolution
            logits only.
        features: Feature maps per stage, replacing the `base_features` / `max_features` rule. A subclass
            that replaces `encoder` must pass the widths of its own stages.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        n_stages: int = 6,
        base_features: int = 32,
        max_features: int = 512,
        block: str = "plain",
        attention: bool = False,
        deep_supervision: bool = False,
        features: list[int] | None = None,
    ):
        super().__init__()
        if deep_supervision and n_stages < 3:
            raise ValueError(
                f"deep supervision needs at least 3 stages, got n_stages={n_stages}"
            )
        if block not in ("plain", "residual"):
            raise ValueError(f"block must be 'plain' or 'residual', got {block!r}")
        self.multiple = 2 ** (n_stages - 1)
        self.features = features or [
            min(base_features * 2**i, max_features) for i in range(n_stages)
        ]
        ins = [in_channels, *self.features[:-1]]
        strides = [1] + [2] * (n_stages - 1)
        if block == "plain":
            stages = [
                double_conv(c, f, s) for c, f, s in zip(ins, self.features, strides)
            ]
        else:
            counts = [
                RESIDUAL_BLOCKS[min(i, len(RESIDUAL_BLOCKS) - 1)]
                for i in range(n_stages)
            ]
            ins[0] = self.features[
                0
            ]  # the stem conv maps in_channels to the stage-0 width
            stages = [
                residual_stage(c, f, s, n)
                for c, f, s, n in zip(ins, self.features, strides, counts)
            ]
            stages[0] = nn.Sequential(
                conv_block(in_channels, self.features[0]), stages[0]
            )
        self.encoder = nn.ModuleList(stages)
        skips = self.features[:-1]
        self.upsample = nn.ModuleList(
            nn.ConvTranspose2d(below, f, kernel_size=2, stride=2)
            for below, f in zip(self.features[1:], skips)
        )
        decoder_stage = double_conv if block == "plain" else conv_block
        self.decoder = nn.ModuleList(decoder_stage(2 * f, f) for f in skips)
        self.head = nn.Conv2d(self.features[0], num_classes, 1)
        self.aux_heads = (
            nn.ModuleList(
                nn.Conv2d(f, num_classes, 1) for f in self.features[1 : n_stages - 1]
            )
            if deep_supervision
            else None
        )
        self.gates = (
            nn.ModuleList(AttentionGate(f, f, f // 2) for f in skips)
            if attention
            else None
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, a=0.01)
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Outputs of every encoder stage, full resolution first, bottleneck last."""
        outputs = []
        for stage in self.encoder:
            x = stage(x)
            outputs.append(x)
        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """Maps (B, in_channels, H, W) to logits (B, num_classes, H, W).

        With deep supervision in train mode it returns a list of such logits at decreasing resolution.
        """
        if x.shape[-2] % self.multiple or x.shape[-1] % self.multiple:
            raise ValueError(
                f"input size {tuple(x.shape[-2:])} must be divisible by {self.multiple} "
                f"(2 ** (n_stages - 1) with n_stages={len(self.features)})"
            )
        skips = self.encode(x)
        x = skips.pop()  # the bottleneck output feeds the decoder directly
        features = []
        for i in reversed(range(len(self.decoder))):
            up, skip = self.upsample[i](x), skips.pop()
            if self.gates is not None:
                skip = self.gates[i](skip, up)
            x = self.decoder[i](torch.cat([skip, up], dim=1))
            features.append(x)
        logits = self.head(x)
        if self.aux_heads is None or not self.training:
            return logits
        return [logits] + [head(f) for head, f in zip(self.aux_heads, features[-2::-1])]


@register("model", "unet")
def build_unet(
    in_channels: int,
    num_classes: int,
    n_stages: int = 6,
    base_features: int = 32,
    max_features: int = 512,
    block: str = "plain",
    attention: bool = False,
    deep_supervision: bool = False,
) -> nn.Module:
    """Builds the U-Net; see `UNet`."""
    return UNet(
        in_channels,
        num_classes,
        n_stages=n_stages,
        base_features=base_features,
        max_features=max_features,
        block=block,
        attention=attention,
        deep_supervision=deep_supervision,
    )
