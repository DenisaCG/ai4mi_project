"""2D U-Net with a frozen pretrained DINOv2 ViT-S/14 encoder (Dino U-Net, arXiv:2508.20909).

The ViT features sit at 1/14 of the (resized) input resolution, so a small trainable conv stem on the
input supplies the full- and half-resolution skips. The decoder is the step-1 U-Net decoder
(`src/models/unet.py`): transposed-conv upsampling, skip concatenation, two convs per stage.
"""

import getpass
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from src.models.unet import conv_block, double_conv
from src.registry import register

PATCH = 14
VIT_CHANNELS = 384
# ViT blocks spread over the 12 (DPT-style), used at strides 2, 4, 8, 16
VIT_BLOCKS = (2, 5, 8, 11)
WIDTHS = (32, 64, 128, 256, 512)  # step-1 U-Net widths at strides 1, 2, 4, 8, 16
HUB_REPO = "facebookresearch_dinov2_main"
WEIGHTS = "dinov2_vits14_pretrain.pth"


def cache_dir(torch_home: str | None = None) -> Path:
    """Torch cache holding the DINOv2 hub repo and weights.

    Uses `torch_home`, else $TORCH_HOME, else /scratch-shared/$USER/torch_cache.
    """
    return Path(
        torch_home
        or os.environ.get("TORCH_HOME")
        or f"/scratch-shared/{getpass.getuser()}/torch_cache"
    )


def load_encoder(torch_home: str | None = None) -> nn.Module:
    """Loads the pretrained DINOv2 ViT-S/14 offline from the cache, frozen and in eval mode."""
    hub = cache_dir(torch_home) / "hub"
    torch.hub.set_dir(str(hub))  # the weights are read from <hub>/checkpoints
    vit = torch.hub.load(str(hub / HUB_REPO), "dinov2_vits14", source="local")
    return vit.requires_grad_(False).eval()


class DinoUNet(nn.Module):
    """U-Net whose encoder is a frozen DINOv2 ViT-S/14.

    Input height and width must be divisible by 16. The ViT sees the input resized to the nearest
    multiple of 14; the stem and the decoder run at the input resolution, so the logits come out at
    the input size without a resize.

    Args:
        in_channels: Number of input channels (stacked slices for 2.5D input).
        num_classes: Number of output classes.
        torch_home: Torch cache with the DINOv2 repo and weights; see `cache_dir`.
    """

    def __init__(
        self, in_channels: int, num_classes: int, torch_home: str | None = None
    ):
        super().__init__()
        self.to_rgb = nn.Conv2d(in_channels, 3, 1)
        self.projections = nn.ModuleList(
            nn.Conv2d(VIT_CHANNELS, width, 1) for width in WIDTHS[1:]
        )
        # full- and half-resolution stem on the raw input
        self.stem = nn.ModuleList(
            [
                conv_block(in_channels, WIDTHS[0]),
                conv_block(WIDTHS[0], WIDTHS[1], stride=2),
            ]
        )
        skips = WIDTHS[:-1]
        self.upsample = nn.ModuleList(
            nn.ConvTranspose2d(below, f, kernel_size=2, stride=2)
            for below, f in zip(WIDTHS[1:], skips)
        )
        self.decoder = nn.ModuleList(double_conv(2 * f, f) for f in skips)
        self.head = nn.Conv2d(WIDTHS[0], num_classes, 1)
        for m in (
            self.modules()
        ):  # before the encoder exists, so its weights stay pretrained
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, a=0.01)
                nn.init.zeros_(m.bias)
        self.encoder = load_encoder(torch_home)

    def train(self, mode: bool = True) -> "DinoUNet":
        """Sets the mode but keeps the frozen encoder in eval mode."""
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Maps (B, in_channels, H, W) to logits (B, num_classes, H, W)."""
        h, w = x.shape[-2:]
        if h % 16 or w % 16:
            raise ValueError(f"input size {(h, w)} must be divisible by 16")
        vit_size = (round(h / PATCH) * PATCH, round(w / PATCH) * PATCH)
        rgb = F.interpolate(
            self.to_rgb(x), size=vit_size, mode="bilinear", align_corners=False
        )
        # Gradients flow through the frozen ViT to `to_rgb`, so no torch.no_grad here.
        features = self.encoder.get_intermediate_layers(rgb, n=VIT_BLOCKS, reshape=True)
        pyramid = [
            F.interpolate(
                proj(f),
                size=(h // 2 ** (i + 1), w // 2 ** (i + 1)),
                mode="bilinear",
                align_corners=False,
            )
            for i, (proj, f) in enumerate(zip(self.projections, features))
        ]
        full = self.stem[0](x)
        half = self.stem[1](full)
        skips = [full, half + pyramid[0], pyramid[1], pyramid[2]]
        x = pyramid[3]
        for i in reversed(range(len(self.decoder))):
            x = self.decoder[i](torch.cat([skips[i], self.upsample[i](x)], dim=1))
        return self.head(x)


@register("model", "dino_unet")
def build_dino_unet(
    in_channels: int, num_classes: int, torch_home: str | None = None
) -> nn.Module:
    """Builds the DINOv2 U-Net; see `DinoUNet`."""
    return DinoUNet(in_channels, num_classes, torch_home=torch_home)
