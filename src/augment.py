"""Online slice augmentations; factories are selected by data.augment."""

import random

import torch
from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode, v2
from torchvision.transforms.functional import affine, gaussian_blur

from src.registry import register


def _check_probability(p: float) -> None:
    if p is None or not 0 <= p <= 1:
        raise ValueError("augmentation p must be chosen in [0, 1]")


def _paired_affine(image: torch.Tensor, gt: torch.Tensor, angle: float, scale: float, fill: float,
                   translate: tuple[int, int] = (0, 0), shear: tuple[float, float] = (0.0, 0.0)):
    if image.shape[-2:] != gt.shape[-2:]:
        raise ValueError("augmentation image and GT sizes differ")
    params = dict(angle=angle, translate=list(translate), scale=scale, shear=list(shear))
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


@register("augment", "random_shift")
def build_random_shift(p: float, max_fraction: float, fill: float):
    _check_probability(p)
    if max_fraction is None or not 0 <= max_fraction < 1:
        raise ValueError("shift max_fraction must be chosen in [0, 1)")

    def shift(image, gt):
        if random.random() < p:
            h, w = image.shape[-2:]
            translate = (round(random.uniform(-max_fraction, max_fraction) * w),
                         round(random.uniform(-max_fraction, max_fraction) * h))
            return _paired_affine(image, gt, angle=0.0, scale=1.0, fill=fill, translate=translate)
        return image, gt

    return shift


@register("augment", "random_elastic")
def build_random_elastic(p: float, alpha: float, sigma: float, fill: float):
    """torchvision v2.ElasticTransform on the CT and GT jointly (same field; bilinear CT, nearest GT)."""
    _check_probability(p)
    if alpha is None or alpha <= 0 or sigma is None or sigma <= 0:
        raise ValueError("elastic alpha and sigma must be chosen (> 0)")
    warp = v2.ElasticTransform(alpha=alpha, sigma=sigma, interpolation=InterpolationMode.BILINEAR,
                               fill={tv_tensors.Image: fill, tv_tensors.Mask: 0})

    def elastic(image, gt):
        if random.random() >= p:
            return image, gt
        if image.shape[-2:] != gt.shape[-2:]:
            raise ValueError("augmentation image and GT sizes differ")
        image, mask = warp(tv_tensors.Image(image), tv_tensors.Mask(gt))
        gt = mask.as_subclass(torch.Tensor).to(gt.dtype)
        gt[0] = gt[0] | (gt.sum(dim=0) == 0)   # outside the source is background
        return image.as_subclass(torch.Tensor), gt

    return elastic


@register("augment", "random_shear")
def build_random_shear(p: float, degrees: list[float], fill: float):
    _check_probability(p)
    if len(degrees) != 2 or degrees[0] > degrees[1]:
        raise ValueError("shear degrees must be [min, max]")

    def shear(image, gt):
        if random.random() < p:
            return _paired_affine(image, gt, angle=0.0, scale=1.0, fill=fill,
                                  shear=(random.uniform(*degrees), 0.0))
        return image, gt

    return shear


@register("augment", "random_gamma")
def build_random_gamma(p: float, gamma: list[float]):
    """nnU-Net-style gamma: the slice is rescaled to [0, 1], raised to a random power, then mapped back."""
    _check_probability(p)
    if len(gamma) != 2 or not 0 < gamma[0] <= gamma[1]:
        raise ValueError("gamma range must be positive [min, max]")

    def apply_gamma(image, gt):
        if random.random() < p:
            lo, hi = image.min(), image.max()
            if hi > lo:
                unit = (image - lo) / (hi - lo)
                image = unit.pow(random.uniform(*gamma)) * (hi - lo) + lo
        return image, gt

    return apply_gamma


@register("augment", "random_brightness_contrast")
def build_random_brightness_contrast(p: float, factor: list[float]):
    """nnU-Net-style: brightness multiplies the slice; contrast scales around its mean and is clipped to the original range."""
    _check_probability(p)
    if len(factor) != 2 or not 0 < factor[0] <= factor[1]:
        raise ValueError("brightness/contrast factor must be positive [min, max]")

    def adjust(image, gt):
        if random.random() < p:
            image = image * random.uniform(*factor)
        if random.random() < p:
            lo, hi, mean = image.min(), image.max(), image.mean()
            image = ((image - mean) * random.uniform(*factor) + mean).clamp(lo, hi)
        return image, gt

    return adjust


@register("augment", "gaussian_blur")
def build_gaussian_blur(p: float, sigma: list[float]):
    _check_probability(p)
    if len(sigma) != 2 or not 0 < sigma[0] <= sigma[1]:
        raise ValueError("blur sigma must be positive [min, max]")

    def blur(image, gt):
        if random.random() < p:
            s = random.uniform(*sigma)
            size = 2 * int(3 * s + 0.5) + 1
            return gaussian_blur(image, [size, size], [s, s]), gt
        return image, gt

    return blur
