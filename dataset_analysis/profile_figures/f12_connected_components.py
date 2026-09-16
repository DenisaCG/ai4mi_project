#!/usr/bin/env python3
"""
Connected components of labels 1, 2 and 3 under five connectivity rules, all
patients pooled.

Rules:
  2D · 4   pixels in an axial slice touch through an edge
  2D · 8   pixels in an axial slice touch through an edge or corner
  3D · 6   voxels of the whole label touch through a face
  3D · 18  voxels touch through a face or edge
  3D · 26  voxels touch through a face, edge or corner

For 2D rules every axial slice containing the label counts once; for 3D rules
every patient counts once. Each heatmap cell is the share of slices or patients
in which the label has that many pieces.

Usage:
    python dataset_analysis/profile_figures/f12_connected_components.py \
        --data-dir data/segthor_part1/train --profile-dir figures/profile
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import generate_binary_structure, label

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profile_figures.common import LABEL_COLORS, build_arg_parser, header, out_subdir  # noqa: E402

LABELS = list(LABEL_COLORS)
# (rule name, dimensions, scipy connectivity rank)
RULES = [("2D · 4", 2, 1), ("2D · 8", 2, 2), ("3D · 6", 3, 1), ("3D · 18", 3, 2), ("3D · 26", 3, 3)]
MAX_PIECES = 3


def piece_counts(mask: np.ndarray) -> dict[str, np.ndarray]:
    """Number of connected pieces of one label mask under every rule.

    Args:
        mask: Boolean label mask (x, y, z), z = axial slice index.

    Returns:
        Rule name -> piece counts: one per axial slice containing the label (2D rules),
        or a single count for the whole mask (3D rules). Empty arrays for an empty mask.
    """
    if not mask.any():
        return {name: np.array([], dtype=int) for name, _, _ in RULES}
    mask = mask[tuple(slice(i.min(), i.max() + 1) for i in np.nonzero(mask))]
    slices = np.flatnonzero(mask.any(axis=(0, 1)))
    counts = {}
    for name, dims, rank in RULES:
        structure = generate_binary_structure(dims, rank)
        if dims == 2:
            counts[name] = np.array([label(mask[:, :, z], structure)[1] for z in slices])
        else:
            counts[name] = np.array([label(mask, structure)[1]])
    return counts


def piece_shares(counts: pd.DataFrame) -> pd.DataFrame:
    """Share of slices (2D) or patients (3D) with each number of pieces.

    Args:
        counts: One row per slice or patient with columns label, rule, pieces.

    Returns:
        Percentages indexed by (label, rule) with one column per piece count 1..MAX_PIECES;
        counts above MAX_PIECES are added to the last column.
    """
    capped = counts.assign(pieces=counts.pieces.clip(upper=MAX_PIECES))
    table = capped.groupby(["label", "rule", "pieces"]).size().unstack("pieces", fill_value=0)
    table = table.reindex(columns=range(1, MAX_PIECES + 1), fill_value=0)
    return 100 * table.div(table.sum(axis=1), axis=0)


def main():
    """Count pieces for every patient and draw the pooled heatmaps."""
    ap = build_arg_parser(__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/segthor_part1/train"))
    args = ap.parse_args()

    rows = []
    patients = sorted(p.name for p in args.data_dir.glob("Patient_*"))
    for patient in patients:
        gt = np.asarray(nib.load(str(args.data_dir / patient / "GT.nii.gz")).dataobj)
        for k in LABELS:
            for rule, values in piece_counts(gt == k).items():
                rows += [{"label": k, "rule": rule, "pieces": int(v)} for v in values]
    shares = piece_shares(pd.DataFrame(rows))

    sns.set_theme(style="white", font_scale=1.1)
    fig, axes = plt.subplots(1, len(LABELS), figsize=(15, 5.2), sharey=True)
    for ax, k in zip(axes, LABELS):
        cmap = LinearSegmentedColormap.from_list(f"label{k}", ["#FFFFFF", LABEL_COLORS[k]])
        table = shares.loc[k].reindex([name for name, _, _ in RULES])
        sns.heatmap(
            table,
            ax=ax,
            cmap=cmap,
            vmin=0,
            vmax=100,
            linewidths=2,
            linecolor="white",
            cbar=k == LABELS[-1],
            cbar_kws={"label": "% of slices (2D) or patients (3D)"},
        )
        ax.set_title(f"label {k}", fontsize=16, fontweight="bold", loc="left", color=LABEL_COLORS[k])
        ax.set(xlabel="pieces", ylabel="")
        ax.set_xticklabels([*map(str, range(1, MAX_PIECES)), f"{MAX_PIECES}+"])
        ax.tick_params(axis="y", rotation=0)
    header(
        fig,
        "Connected Components per Label",
        f"All {len(patients)} patients pooled; each cell = % of axial slices (2D rows) or patients (3D rows) with "
        "that many pieces.",
        subtitle_y=0.9,
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.14, wspace=0.08)
    out = out_subdir(args.profile_dir, "12_connected_components") / "connected_components.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
