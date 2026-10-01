"""Class weights computed from a fold's processed training masks."""

import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset import SliceDataset
from src.data import Augmented


def compute_class_weights(dataset: Dataset, num_classes: int) -> list[float]:
    """Return log normalized inverse pixel frequencies in class-index order.

    Pass the current fold's training dataset, never validation or test data.
    Read each mask once using its configured label transform, bypassing online
    augmentation and image loading. Absent classes receive zero weight; an
    empty dataset raises ValueError. Returns the natural logarithm of inverse
    frequency weights, adding 1 to each weight to avoid negative weights.
    """
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    while isinstance(dataset, Augmented):
        dataset = dataset.base
    if not isinstance(dataset, SliceDataset):
        raise TypeError("Expected a SliceDataset or an Augmented SliceDataset")

    counts = torch.zeros(num_classes, dtype=torch.float64)
    for _, mask_path in dataset.files:
        if mask_path is None:
            raise ValueError("Class weights require ground-truth masks")
        with Image.open(mask_path) as image:
            mask = dataset.gt_transform(image)
        if mask.ndim != 3 or mask.shape[0] != num_classes:
            raise ValueError(f"Expected a ({num_classes}, H, W) mask: {mask_path}")
        counts += mask.sum(dim=(1, 2), dtype=torch.float64)

    total = counts.sum()
    if total == 0:
        raise ValueError("Cannot compute class weights from an empty dataset")
    weights = torch.zeros_like(counts)
    present = counts > 0
    weights[present] = torch.log((total / counts[present])) + 1
    return weights.tolist()
