#!/usr/bin/env python3
"""
Per-patient CT intensity along the superior-inferior axis: for every axial slice, the
median, middle 50% (25th-75th percentile) and middle 90% (5th-95th
percentile) of that slice's HU values, computed from every pixel (no
subsampling).

Slice 0 is the lowest slice of the scan; height above it is the slice index
times the z voxel spacing, in mm. The nnU-Net clip values are read from
figures/profile/intensity_histograms.npz via nnunet_ct_normalisation().

Usage:
    python tools/profile_figures/f04_scan_intensity/bands_along_height.py --data-dir data/segthor_part1/train
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset_profile import load_tables  # noqa: E402
from plot_style import apply_ticks_style, title_block  # noqa: E402
from profile_figures.f04_scan_intensity.common import (
    output_path,  # noqa: E402
    DATA_DIR,
    PROFILE_DIR,
    load_ct,
    nnunet_ct_normalisation,
    patient_name,
    patients_in,
)

N_GRID_COLORS = 20  # size of the per-patient color scale, independent of how many patients are plotted
PERCENTILES = [5, 25, 50, 75, 95]


def _slice_percentiles(data_dir_and_patient: tuple[Path, str]) -> tuple[str, np.ndarray, np.ndarray]:
    """Per-slice HU percentiles for one patient, using every pixel of every slice.

    Args:
        data_dir_and_patient: (data_dir, patient id), packed together for Pool.map.

    Returns:
        Patient id, height above the lowest slice per slice (mm), and a
        (n_slices, 5) array of p5/p25/median/p75/p95 per slice.
    """
    data_dir, patient = data_dir_and_patient
    ct, spacing = load_ct(data_dir, patient)
    heights = np.arange(ct.shape[2]) * spacing[2]
    pct = np.percentile(ct.reshape(-1, ct.shape[2]), PERCENTILES, axis=0).T
    return patient, heights, pct


def _clip_hlines(ax, norm: dict) -> None:
    """Dashed horizontal lines at the nnU-Net clip values.

    Args:
        ax: Axes to draw on.
        norm: Dict with "clip_low" and "clip_high", as returned by nnunet_ct_normalisation().
    """
    for value in (norm["clip_low"], norm["clip_high"]):
        ax.axhline(value, color="#555555", ls="--", lw=1, zorder=1)


def main():
    """Draw the per-patient percentile-bands-along-height figure."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile-dir", type=Path, default=PROFILE_DIR)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = ap.parse_args()

    hists = load_tables(args.profile_dir)["histograms"]
    norm = nnunet_ct_normalisation(hists)
    patients = patients_in(hists)

    with mp.Pool() as pool:
        results = pool.map(_slice_percentiles, [(args.data_dir, patient) for patient in patients])

    apply_ticks_style()
    colors = sns.color_palette("crest", N_GRID_COLORS)
    fig, axes = plt.subplots(4, 5, figsize=(22, 15), sharey=True)

    for ax, (patient, heights, pct), color in zip(axes.flat, results, colors):
        ax.fill_between(heights, pct[:, 0], pct[:, 4], color=color, alpha=0.25, linewidth=0)
        ax.fill_between(heights, pct[:, 1], pct[:, 3], color=color, alpha=0.55, linewidth=0)
        ax.plot(heights, pct[:, 2], color=color, linewidth=1.4)
        _clip_hlines(ax, norm)
        ax.set_title(patient_name(patient), fontsize=13, fontweight="bold", loc="left")
        sns.despine(ax=ax)

    for ax in axes[-1, :]:
        ax.set_xlabel("height above lowest slice (mm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("HU")

    legend_handles = [
        plt.Line2D([], [], color="black", linewidth=1.4, label="median"),
        plt.Rectangle((0, 0), 1, 1, facecolor="black", alpha=0.55, label="middle 50% (p25–p75)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="black", alpha=0.25, label="middle 90% (p5–p95)"),
        plt.Line2D([], [], color="#555555", ls="--", linewidth=1, label="nnU-Net clip"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False, fontsize=11)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    title_block(
        fig,
        "Scan Intensity: Percentile Bands Along Height",
        "Per slice, all pixels: median, middle 50% (25th–75th percentile) and middle 90% (5th–95th) "
        "of intensity, from the lowest to the highest slice.",
    )
    fig.subplots_adjust(top=0.88)

    out_path = output_path(args.profile_dir, "bands_along_height.png")
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
