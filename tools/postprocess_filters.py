"""Ground-truth-free post-processing filters on 3D label volumes (z is the last axis), used by tools.gating_oracle.

Every filter returns a copy, leaves the input untouched and acts on each organ independently.
"""

import numpy as np
from scipy import ndimage
from skimage.morphology import convex_hull_image


def take_organs(base: np.ndarray, source: np.ndarray, classes: list[int]) -> np.ndarray:
    """Copy of `base` where, for each organ in `classes`, the voxels of `base` with that label are
    replaced by `source` (label k kept or set to whatever `source` has there)."""
    out = base.copy()
    for k in classes:
        mask = base == k
        out[mask] = source[mask]
    return out


def remove_small_components(
    pred: np.ndarray, k: int, rel: float = 0.0, min_voxels: int = 0
) -> np.ndarray:
    """Removes the 6-connected components of organ `k` smaller than `rel` times the largest one or
    than `min_voxels` voxels. The largest component is always kept, so no organ is emptied."""
    out = pred.copy()
    labels, n = ndimage.label(out == k)
    if n > 1:
        sizes = np.bincount(labels.ravel())
        keep = sizes >= max(rel * sizes[1:].max(), min_voxels)
        keep[0] = False
        keep[1 + np.argmax(sizes[1:])] = True
        out[(labels > 0) & ~keep[labels]] = 0
    return out


def heart_hull(pred: np.ndarray, k: int) -> np.ndarray:
    """Fills the 3D convex hull of organ `k` into the background voxels around it. The hull is taken
    of the mask as given (apply the largest component first); other organs are never overwritten, and
    it cannot leave the volume."""
    out = pred.copy()
    mask = pred == k
    if not mask.any():
        return out
    coords = np.argwhere(mask)
    box = tuple(slice(a, b + 1) for a, b in zip(coords.min(0), coords.max(0)))
    hull = convex_hull_image(mask[box])
    sub = out[box]
    sub[hull & (sub == 0)] = k
    return out


def area_gate(pred: np.ndarray, classes: list[int], min_voxels: float) -> np.ndarray:
    """Sets organ k to background on the z slices where it covers fewer than `min_voxels` voxels."""
    out = pred.copy()
    for k in classes:
        area = (out == k).sum(axis=(0, 1))
        out[(out == k) & (area < min_voxels)[None, None, :]] = 0
    return out


def adjacent_gate(pred: np.ndarray, classes: list[int]) -> np.ndarray:
    """Sets organ k to background on the z slices where it is predicted but not on slice z-1 or z+1
    (judged on the input, not iteratively)."""
    out = pred.copy()
    for k in classes:
        present = (pred == k).any(axis=(0, 1))
        near = np.zeros_like(present)
        near[1:] |= present[:-1]
        near[:-1] |= present[1:]
        out[(out == k) & ~near[None, None, :]] = 0
    return out
