"""Post-processing of 3D label volumes (z is the last axis) that the pipeline uses: keeping only the organs of one
volume, and voting over the seeds' volumes. Each function returns a new array and leaves its inputs untouched.
"""

import numpy as np


def take_organs(base: np.ndarray, source: np.ndarray, classes: list[int]) -> np.ndarray:
    """Copy of `base` where, for each organ in `classes`, the voxels of `base` with that label are
    replaced by `source` (label k kept or set to whatever `source` has there)."""
    out = base.copy()
    for k in classes:
        mask = base == k
        out[mask] = source[mask]
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
