"""Slice-presence auxiliary head: per-slice organ targets, their BCE loss, gating of 2D predictions and
accuracy counts. Organ k (1..K-1) is present on a slice if the GT has any voxel of it; output k-1 of the head
is the logit of that."""

import numpy as np
import torch
import torch.nn.functional as F

PRESENCE_THRESHOLD = 0.5  # fixed, never tuned on the validation folds


def presence_target(gt_onehot: torch.Tensor) -> torch.Tensor:
    """Slice-level organ presence of a one-hot GT.

    Args:
        gt_onehot: One-hot GT of shape (B, K, H, W).

    Returns:
        Float tensor of shape (B, K - 1): 1 where the slice has any voxel of the foreground organ, else 0.
    """
    return gt_onehot[:, 1:].amax(dim=(2, 3)).float()


def presence_loss(logits: torch.Tensor, gt_onehot: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy between presence logits (B, K - 1) and the presence of a one-hot GT (B, K, H, W)."""
    return F.binary_cross_entropy_with_logits(logits, presence_target(gt_onehot))


def gate_prediction(pred: np.ndarray, presence_prob: np.ndarray) -> np.ndarray:
    """Sets every organ whose presence probability is below the threshold to background.

    Args:
        pred: Class map of one slice, shape (H, W), with values 0..K-1.
        presence_prob: Presence probabilities of organs 1..K-1, shape (K - 1,).

    Returns:
        A copy of `pred` with the gated organs set to 0.
    """
    out = pred.copy()
    for k, prob in enumerate(presence_prob, start=1):
        if prob < PRESENCE_THRESHOLD:
            out[out == k] = 0
    return out


def presence_counts(prob: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Confusion counts of the thresholded presence probabilities.

    Args:
        prob: Presence probabilities, shape (N, K - 1).
        target: Presence targets (0 or 1), shape (N, K - 1).

    Returns:
        Integer array of shape (K - 1, 4) with true positives, false positives, false negatives and true
        negatives per organ.
    """
    pred, target = prob >= PRESENCE_THRESHOLD, target.astype(bool)
    return np.stack(
        [
            (pred & target).sum(0),
            (pred & ~target).sum(0),
            (~pred & target).sum(0),
            (~pred & ~target).sum(0),
        ],
        axis=1,
    )
