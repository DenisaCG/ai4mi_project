#!/usr/bin/env python3
"""
Scan geometry overview: voxel spacing, slices per scan and physical size of
every CT scan, drawn as dot histograms (one dot per scan).

Reads patients.csv written by tools/dataset_profile.py.

Usage:
    python tools/profile_figures/f01_03_scan_geometry.py --profile-dir figures/profile
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import apply_ticks_style  # noqa: E402
from profile_figures.common import build_arg_parser, header, out_subdir  # noqa: E402

MEDIAN_COLOR = "#D1495B"
# (column, panel title, x-axis label, bin width); bins are centred on multiples of the width
PANELS = [
    ("x_spacing_mm", "Patients by pixel size", "pixel size, x = y (mm)", 0.01),
    ("z_spacing_mm", "Patients by slice spacing", "slice spacing, z (mm)", 0.01),
    ("spacing_ratio", "Patients by spacing ratio", "slice spacing ÷ pixel size", 0.05),
    ("z_voxels", "Patients by number of slices", "number of slices", 15),
    ("x_extent_mm", "Patients by image width", "image width (mm)", 25),
    ("z_extent_mm", "Patients by scan length", "scan length (mm)", 40),
]


def dot_stacks(values: pd.Series, width: float) -> tuple[np.ndarray, np.ndarray]:
    """Bin values into stacks of dots, one dot per value.

    Args:
        values: One value per scan.
        width: Bin width; bins are centred on multiples of it.

    Returns:
        x (bin centre) and y (stack position, 0.5, 1.5, ...) for every value.
    """
    centres = np.round(values.to_numpy(dtype=float) / width) * width
    x, y = [], []
    for centre in np.unique(centres):
        n = int((centres == centre).sum())
        x += [centre] * n
        y += list(np.arange(n) + 0.5)
    return np.array(x), np.array(y)


def main():
    """Draw the scan geometry overview from patients.csv."""
    ap = build_arg_parser(__doc__)
    args = ap.parse_args()

    scans = pd.read_csv(args.profile_dir / "patients.csv")
    scans["spacing_ratio"] = scans.z_spacing_mm / scans.x_spacing_mm
    apply_ticks_style()
    colors = sns.color_palette("crest", len(PANELS))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (col, title, xlabel, width), color in zip(axes.ravel(), PANELS, colors):
        x, y = dot_stacks(scans[col], width)
        ax.scatter(x, y, s=110, color=color, edgecolor="white", lw=1.2)
        ax.plot(
            scans[col].median(),
            0,
            marker="^",
            markersize=13,
            color=MEDIAN_COLOR,
            clip_on=False,
            zorder=6,
            transform=ax.get_xaxis_transform(),
        )
        ax.set_ylim(0, y.max() + 1.1)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(title, fontsize=15, fontweight="bold", loc="left")
        ax.set_xlabel(xlabel, fontsize=13)
        sns.despine(ax=ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("patients")
    fig.legend(
        handles=[plt.Line2D([], [], marker="^", ls="", color=MEDIAN_COLOR, markersize=12, label="median")],
        loc="lower center",
        frameon=False,
    )
    header(
        fig,
        "Scan Geometry: Voxel Spacing, Slices and Size",
        f"Voxel spacing, number of slices and physical size of the CT scan of each of the {len(scans)} patients. "
        "Each dot is one patient.",
        subtitle_y=0.935,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    out = out_subdir(args.profile_dir, "01-03_scan_geometry") / "scan_geometry.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
