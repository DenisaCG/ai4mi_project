#!/usr/bin/env python3
"""
Scan geometry: voxel spacing, slices per scan and physical size of every CT scan.

Writes two figures into figures/profile/01-03_scan_geometry/:
  scan_geometry.png     dot histograms, one dot per scan.
  grid_per_patient.png  one row per patient, one column per measurement, with
                        the median of all patients as the bottom row.

Reads patients.csv written by tools/dataset_profile.py.

Usage:
    python dataset_analysis/profile_figures/f01_03_scan_geometry.py --profile-dir figures/profile
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
from profile_figures.common import apply_ticks_style, build_arg_parser, header, out_subdir  # noqa: E402

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

# (column, column title) for the per-patient grid
GRID_COLUMNS = [
    ("x_spacing_mm", "x spacing (mm)"),
    ("y_spacing_mm", "y spacing (mm)"),
    ("z_spacing_mm", "z spacing (mm)"),
    ("x_voxels", "x voxels"),
    ("y_voxels", "y voxels"),
    ("z_voxels", "z voxels"),
    ("x_extent_mm", "x size (mm)"),
    ("y_extent_mm", "y size (mm)"),
    ("z_extent_mm", "z size (mm)"),
    ("voxels_millions", "total voxels (millions)"),
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


def draw_overview(scans: pd.DataFrame, out_dir: Path) -> None:
    """Draw the dot-histogram overview of scan geometry.

    Args:
        scans: patients.csv with a spacing_ratio column added.
        out_dir: Folder to write scan_geometry.png into.
    """
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
    out = out_dir / "scan_geometry.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def draw_grid_per_patient(scans: pd.DataFrame, out_dir: Path) -> None:
    """Draw every patient's geometry as rows, one column per measurement, median row at the bottom.

    Args:
        scans: patients.csv.
        out_dir: Folder to write grid_per_patient.png into.
    """
    table = scans.assign(voxels_millions=scans.voxels / 1e6).sort_values(["x_spacing_mm", "z_spacing_mm"])
    table["patient"] = table.patient.str.replace("Patient_", "patient ")
    cols = [col for col, _ in GRID_COLUMNS]
    median = pd.DataFrame([{"patient": "median of all patients", **table[cols].median().to_dict()}])
    table = pd.concat([table[["patient", *cols]], median], ignore_index=True)
    n_rows = len(table)
    colors = sns.color_palette("crest", len(GRID_COLUMNS))

    fig, axes = plt.subplots(1, len(GRID_COLUMNS), figsize=(30, 10), sharey=True)
    for ax, (col, label), color in zip(axes, GRID_COLUMNS, colors):
        ax.scatter(table[col][:-1], range(n_rows - 1), s=90, color=color, edgecolor="white", lw=1, zorder=3)
        ax.scatter(table[col].iloc[-1], n_rows - 1, marker="^", s=150, color=MEDIAN_COLOR, zorder=4)
        ax.axhline(n_rows - 1.5, color="#999999", lw=0.8)
        ax.yaxis.grid(True, color="#E6E6E6")
        ax.set_title(label, fontsize=14, fontweight="bold", loc="left")
        ax.tick_params(axis="y", left=False)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        sns.despine(ax=ax, left=True)
    axes[0].set_yticks(range(n_rows))
    axes[0].set_yticklabels(table.patient)
    axes[0].set_ylim(n_rows - 0.5, -0.5)
    header(
        fig,
        "Scan Geometry per Patient",
        "One row per patient, sorted by pixel size then slice spacing; the bottom row is the median of all patients, "
        "which nnU-Net uses as its target spacing.",
        subtitle_y=0.915,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = out_dir / "grid_per_patient.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    """Draw both scan geometry figures from patients.csv."""
    ap = build_arg_parser(__doc__)
    args = ap.parse_args()

    scans = pd.read_csv(args.profile_dir / "patients.csv")
    scans["spacing_ratio"] = scans.z_spacing_mm / scans.x_spacing_mm
    out_dir = out_subdir(args.profile_dir, "01-03_scan_geometry")
    apply_ticks_style()
    draw_overview(scans, out_dir)
    draw_grid_per_patient(scans, out_dir)


if __name__ == "__main__":
    main()
