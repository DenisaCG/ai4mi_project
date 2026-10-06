"""2D U-Net with a ResNet34 encoder (torchvision), optionally ImageNet-pretrained.

The decoder, deep supervision and stage-0 block are those of the residual-encoder U-Net in
`src/models/unet.py`. ResNet34 has no stride-1 feature map, so stage 0 stays the trainable
full-resolution block of the residual-encoder U-Net, and the ResNet34 stages follow at strides
2, 4, 8, 16 and 32.
"""

import os
from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet34

from src.models.unet import UNet
from src.registry import register

WEIGHTS = "resnet34-b627a593.pth"  # torchvision ResNet34_Weights.IMAGENET1K_V1
WIDTHS = [
    32,
    64,
    64,
    128,
    256,
    512,
]  # stage 0, then the ResNet34 stem, layer1 .. layer4


def weights_path() -> Path:
    """Where the pretrained weights must be cached: $TORCH_HOME/hub/checkpoints, else ~/torch_cache."""
    home = Path(os.environ.get("TORCH_HOME") or Path.home() / "torch_cache")
    return home / "hub" / "checkpoints" / WEIGHTS


class ResNetUNet(UNet):
    """U-Net whose encoder is stage 0 of the residual-encoder U-Net followed by a ResNet34.

    The one-channel input is repeated to three channels for the ResNet34; stage 0 sees it as is.
    Input height and width must be divisible by 32.

    Args:
        in_channels: Number of input channels; must be 1.
        num_classes: Number of output classes.
        pretrained: Whether to load the ImageNet weights from `weights_path()`.
        deep_supervision: See `UNet`.

    Raises:
        ValueError: If `in_channels` is not 1.
        FileNotFoundError: If `pretrained` and the weights are not cached.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        pretrained: bool = True,
        deep_supervision: bool = False,
    ):
        if in_channels != 1:
            raise ValueError(f"in_channels must be 1, got {in_channels}")
        path = weights_path()
        if pretrained and not path.is_file():
            raise FileNotFoundError(
                f"ImageNet ResNet34 weights missing: {path}; download "
                f"https://download.pytorch.org/models/{WEIGHTS} to it on the login node"
            )
        super().__init__(
            in_channels,
            num_classes,
            block="residual",
            deep_supervision=deep_supervision,
            features=WIDTHS,
        )
        net = resnet34()
        if pretrained:
            net.load_state_dict(torch.load(path, weights_only=True))
        self.encoder = nn.ModuleList(
            [
                self.encoder[0],
                nn.Sequential(net.conv1, net.bn1, net.relu),
                nn.Sequential(net.maxpool, net.layer1),
                net.layer2,
                net.layer3,
                net.layer4,
            ]
        )

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Stage-0 output on the input, then the ResNet34 stages on the input repeated to 3 channels."""
        outputs = [self.encoder[0](x)]
        x = x.expand(-1, 3, -1, -1)
        for stage in self.encoder[1:]:
            x = stage(x)
            outputs.append(x)
        return outputs


@register("model", "resnet34_unet")
def build_resnet34_unet(
    in_channels: int,
    num_classes: int,
    pretrained: bool = True,
    deep_supervision: bool = False,
) -> nn.Module:
    """Builds the ResNet34 U-Net; see `ResNetUNet`."""
    return ResNetUNet(
        in_channels,
        num_classes,
        pretrained=pretrained,
        deep_supervision=deep_supervision,
    )
