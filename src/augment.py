"""Online slice augmentations; factories are selected by data.augment."""

import random

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import affine

from src.registry import register


def _check_probability(p: float) -> None:
    if p is None or not 0 <= p <= 1:
        raise ValueError("augmentation p must be chosen in [0, 1]")


def _paired_affine(image: torch.Tensor, gt: torch.Tensor, angle: float, scale: float, fill: float):
    if image.shape[-2:] != gt.shape[-2:]:
        raise ValueError("augmentation image and GT sizes differ")
    params = dict(angle=angle, translate=[0, 0], scale=scale, shear=[0.0, 0.0])
    image = affine(image, **params, interpolation=InterpolationMode.BILINEAR, fill=[fill])
    gt = affine(gt, **params, interpolation=InterpolationMode.NEAREST,
                fill=[1.0] + [0.0] * (gt.shape[0] - 1))  # outside the source is background
    return image, gt


@register("augment", "random_rotation")
def build_random_rotation(p: float, degrees: list[float], fill: float):
    _check_probability(p)
    if len(degrees) != 2 or degrees[0] > degrees[1]:
        raise ValueError("rotation degrees must be [min, max]")

    def rotate(image, gt):
        if random.random() < p:
            return _paired_affine(image, gt, angle=random.uniform(*degrees), scale=1.0, fill=fill)
        return image, gt

    return rotate


@register("augment", "random_scaling")
def build_random_scaling(p: float, scales: list[float], fill: float):
    _check_probability(p)
    if len(scales) != 2 or not 0 < scales[0] <= scales[1]:
        raise ValueError("scaling scales must be positive [min, max]")

    def scale(image, gt):
        if random.random() < p:
            return _paired_affine(image, gt, angle=0.0, scale=random.uniform(*scales), fill=fill)
        return image, gt

    return scale


@register("augment", "gaussian_noise")
def build_gaussian_noise(p: float, sigma: float):
    _check_probability(p)
    if sigma is None or sigma < 0:
        raise ValueError("Gaussian noise sigma must be chosen in normalized intensity units (>= 0)")

    def add_noise(image, gt):
        if random.random() < p:
            return image + sigma * torch.randn_like(image), gt
        return image, gt

    return add_noise
