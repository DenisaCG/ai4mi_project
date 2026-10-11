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


def min_run_length(pred: np.ndarray, classes: list[int], length: int) -> np.ndarray:
    """Sets organ k to background on every run of consecutive z slices shorter than `length` where it
    is predicted (length 2 is adjacent_gate)."""
    out = pred.copy()
    for k in classes:
        runs, n = ndimage.label((pred == k).any(axis=(0, 1)))
        short = np.flatnonzero(np.bincount(runs, minlength=n + 1)[1:] < length) + 1
        out[(out == k) & np.isin(runs, short)[None, None, :]] = 0
    return out


def z_extent(pred: np.ndarray, classes: list[int], margin: int) -> np.ndarray:
    """Keeps organ k only on the z slices spanned by its largest 6-connected component, widened by
    `margin` slices on each side."""
    out = pred.copy()
    for k in classes:
        labels, n = ndimage.label(pred == k)
        if not n:
            continue
        largest = 1 + np.argmax(np.bincount(labels.ravel())[1:])
        z = np.flatnonzero((labels == largest).any(axis=(0, 1)))
        keep = np.zeros(pred.shape[2], dtype=bool)
        keep[max(z[0] - margin, 0) : z[-1] + margin + 1] = True
        out[(out == k) & ~keep[None, None, :]] = 0
    return out


def slice_components(pred: np.ndarray, classes: list[int], rel: float) -> np.ndarray:
    """On every z slice, removes the 4-connected pieces of organ k smaller than `rel` times the largest
    piece of that organ on the slice."""
    out = pred.copy()
    for k in classes:
        for z in np.flatnonzero((pred == k).any(axis=(0, 1))):
            labels, n = ndimage.label(pred[:, :, z] == k)
            if n > 1:
                sizes = np.bincount(labels.ravel())
                small = sizes < rel * sizes[1:].max()
                small[0] = False
                out[:, :, z][small[labels]] = 0
    return out


def majority_vote(volumes: list[np.ndarray]) -> np.ndarray:
    """Per voxel, the label most volumes agree on. Where several labels tie for the most votes, the
    label of the first volume is used."""
    labels = np.unique(np.concatenate([np.unique(v) for v in volumes]))
    votes = np.stack([sum((v == c).astype(np.uint8) for v in volumes) for c in labels])
    top = votes.max(axis=0)
    tied = (votes == top).sum(axis=0) > 1
    return np.where(tied, volumes[0], labels[votes.argmax(axis=0)]).astype(volumes[0].dtype)


def threshold_vote(
    volumes: list[np.ndarray], classes: list[int], thresholds: dict[int, int]
) -> np.ndarray:
    """Per voxel, the organ with most votes among those that have at least `thresholds[k]` votes (1 =
    union, the number of volumes = intersection); background if none does. A tie goes to the label of
    the first volume, then to the lower class index."""
    first = volumes[0]
    out = np.zeros_like(first)
    best = np.zeros(first.shape, dtype=np.int16)
    for k in classes:
        votes = sum((v == k).astype(np.int16) for v in volumes)
        score = np.where(votes >= thresholds[k], 2 * votes + (first == k), 0)
        better = score > best
        out[better] = k
        best[better] = score[better]
    return out


def contiguous_z(pred: np.ndarray, classes: list[int], gap: int) -> np.ndarray:
    """Keeps organ k only on its longest run of predicted z slices, where slices separated by at most
    `gap` empty slices belong to one run (the longest has the most predicted slices; ties go to the lowest z)."""
    out = pred.copy()
    for k in classes:
        z = np.flatnonzero((pred == k).any(axis=(0, 1)))
        if not len(z):
            continue
        runs = np.split(z, np.flatnonzero(np.diff(z) > gap + 1) + 1)
        best = max(runs, key=len)
        keep = np.zeros(pred.shape[2], dtype=bool)
        keep[best[0] : best[-1] + 1] = True
        out[(out == k) & ~keep[None, None, :]] = 0
    return out


def trachea_anchor(pred: np.ndarray, organ: int, trachea: int, radius: float) -> np.ndarray:
    """Removes the 6-connected components of `organ` whose centroid is more than `radius` in-plane voxels
    from the trachea centroid on the component's centroid slice (the nearest slice with trachea where that
    slice has none). Without any trachea nothing is removed."""
    out = pred.copy()
    present = np.flatnonzero((pred == trachea).any(axis=(0, 1)))
    labels, n = ndimage.label(pred == organ)
    if not len(present) or not n:
        return out
    anchor = {
        z: np.argwhere(pred[:, :, z] == trachea).mean(axis=0) for z in present
    }
    centroids = ndimage.center_of_mass(np.ones_like(labels), labels, range(1, n + 1))
    remove = np.zeros(n + 1, dtype=bool)
    for i, (cx, cy, cz) in enumerate(centroids, start=1):
        z = present[np.argmin(np.abs(present - cz))]
        remove[i] = np.hypot(cx - anchor[z][0], cy - anchor[z][1]) > radius
    out[remove[labels]] = 0
    return out
