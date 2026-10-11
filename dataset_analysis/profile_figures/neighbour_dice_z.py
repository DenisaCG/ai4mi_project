#!/usr/bin/env python3
"""
How much of an organ the neighbouring slices show: Dice between an organ's mask on a slice and
on the slice d mm above or below ("copy-the-neighbour Dice"), per organ.

Labels are resampled along z to 2.5 mm (nearest neighbour), the slice spacing the 2.5D runs
train on, so offsets of 1/2/4 slices are the ±2.5/±5/±10 mm a 3-slice input reaches at
stride 1/2/4. Every slice that contains the organ is a centre slice; it falls in one of
three regions, by distance to that organ's top and bottom slice in the same patient:
  top     within EDGE_SLICES slices of the organ's most superior slice
  bottom  within EDGE_SLICES slices of its most inferior slice
  middle  everything else
A neighbour outside the scan is skipped; a neighbour without the organ scores 0.

Writes into <out-dir>/neighbour_dice_z/:
  neighbour_dice_z.png/.pdf  one panel per region, one line per organ (median over slices,
                             band = 25-75%), x = neighbour offset (mm, below ... above).
  neighbour_dice_z_single.png/.pdf  one small panel per organ: mean Dice against distance
                             (below and above pooled), solid = middle, dashed = near an end
                             (neighbours past the end left out).
  neighbour_dice_z.csv       one row per patient, organ, centre slice and offset.
  neighbour_dice_z_log.json  patients, spacing, orientation, git commit, time.

Usage:
    python dataset_analysis/profile_figures/neighbour_dice_z.py \
        --data-dir data/segthor_train_full/train --out-dir figures/dataset_40
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from plot_style import EARTH_LABEL_COLORS, apply_style, decorate, legend_below, tint  # noqa: E402
from utils import CLASSES, NAMES  # noqa: E402

TARGET_MM = 2.5
MAX_OFFSET = 4  # slices; stride 4 with context 1 reaches ±4 slices
EDGE_SLICES = 4  # centre slices within 10 mm of an organ end count as "top"/"bottom"
STRIDES = [1, 2, 4]
REGIONS = [
    ("top", "Near the organ's top (within 10 mm)"),
    ("middle", "Middle of the organ"),
    ("bottom", "Near the organ's bottom (within 10 mm)"),
]


def resampled_labels(gt_path: Path) -> tuple[np.ndarray, dict]:
    """Load a label volume, check orientation and resample it along z to TARGET_MM.

    Args:
        gt_path: Patient's GT.nii.gz.

    Returns:
        Labels with slice index increasing toward superior, and the patient's geometry for the log.
    """
    nii = nib.load(gt_path)
    codes = "".join(nib.aff2axcodes(nii.affine))
    assert codes == "LPS", f"{gt_path}: orientation {codes}, expected LPS"
    spacing = [float(s) for s in nii.header.get_zooms()[:3]]
    seg = np.asarray(nii.dataobj)
    n = seg.shape[2]
    keep = np.round(np.arange(0, (n - 1) * spacing[2] + 1e-6, TARGET_MM) / spacing[2]).astype(int)
    seg = seg[:, :, keep]
    geometry = {
        "patient": gt_path.parent.name,
        "orientation": codes,
        "spacing_mm": spacing,
        "shape": [*seg.shape[:2], n],
        "resampled_slices": int(seg.shape[2]),
    }
    return seg, geometry


def neighbour_dice(seg: np.ndarray, pid: str) -> pd.DataFrame:
    """Dice between each organ slice and its neighbours up to MAX_OFFSET slices away.

    Args:
        seg: Labels on the TARGET_MM grid, z toward superior (LPS).
        pid: Patient id.

    Returns:
        One row per organ, centre slice and offset (positive = superior) with dice and region.
    """
    rows = []
    for c in CLASSES:
        mask = seg == c
        occupied = np.flatnonzero(mask.any(axis=(0, 1)))
        if occupied.size == 0:
            continue
        xs, ys = np.flatnonzero(mask.any(axis=(1, 2))), np.flatnonzero(mask.any(axis=(0, 2)))
        mask = mask[xs[0] : xs[-1] + 1, ys[0] : ys[-1] + 1]  # crop to the organ, Dice is unchanged
        area = mask.sum(axis=(0, 1))
        bottom, top = occupied[0], occupied[-1]
        for z in occupied:
            to_top, to_bottom = top - z, z - bottom
            if min(to_top, to_bottom) > EDGE_SLICES:
                region = "middle"
            else:
                region = "top" if to_top <= to_bottom else "bottom"
            for d in range(-MAX_OFFSET, MAX_OFFSET + 1):
                if not 0 <= z + d < mask.shape[2]:
                    continue
                overlap = np.logical_and(mask[:, :, z], mask[:, :, z + d]).sum()
                rows.append((pid, c, int(z), d, d * TARGET_MM, region, 2 * overlap / (area[z] + area[z + d])))
    return pd.DataFrame(rows, columns=["patient", "label", "slice", "offset", "offset_mm", "region", "dice"])


def draw(df: pd.DataFrame, labels: list[int], n_patients: int, out_dir: Path) -> None:
    """One panel per region: median copy-the-neighbour Dice per organ against offset.

    Args:
        df: Output rows of neighbour_dice for every patient.
        labels: Labels with voxels in at least one patient.
        n_patients: Number of patients pooled.
        out_dir: Folder to write the figure into.
    """
    apply_style()
    stats = (
        df.groupby(["region", "label", "offset_mm"])
        .dice.agg(median="median", q25=lambda d: d.quantile(0.25), q75=lambda d: d.quantile(0.75))
        .reset_index()
    )
    fig, axes = plt.subplots(1, len(REGIONS), figsize=(16, 6), sharey=True)
    for ax, (region, title) in zip(axes, REGIONS):
        for s in STRIDES:
            for side in (-1, 1):
                ax.axvline(side * s * TARGET_MM, color="#BBBBBB", lw=0.9, ls=":", zorder=1)
        for c in labels:
            st = stats[(stats.region == region) & (stats.label == c)]
            ax.fill_between(st.offset_mm, st.q25, st.q75, color=tint(EARTH_LABEL_COLORS[c], 0.25), lw=0, zorder=2)
            ax.plot(st.offset_mm, st["median"], color=EARTH_LABEL_COLORS[c], lw=2.2, marker="o", ms=4,
                    label=NAMES[c], zorder=3)
        n_slices = df[(df.region == region) & (df.offset == 0)].groupby("label").size()
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=18)
        ax.text(0, 1.0, "slices: " + ", ".join(f"{NAMES[c].lower()} {n_slices.get(c, 0)}" for c in labels),
                transform=ax.transAxes, fontsize=9, color="#666666", va="bottom")
        ax.set_xticks(np.arange(-MAX_OFFSET, MAX_OFFSET + 1) * TARGET_MM)
        ax.set_xticklabels(
            [f"{d * TARGET_MM:g}" + (f"\ns{abs(d)}" if abs(d) in STRIDES else "\n") for d in range(-MAX_OFFSET, MAX_OFFSET + 1)],
            fontsize=9,
        )
        ax.set_xlabel("neighbour offset (mm)\n←  below      above  →")
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("Dice of copying the neighbour's mask")
    legend_below(axes[0], ncol=len(labels))

    decorate(
        fig,
        "Inside an Organ the Neighbour Slices Are Near-Copies; Near Its Ends, and for Thin Organs, They Drift Fast",
        subtitle=(
            "How well the organ mask on a slice nearby matches this slice (1 = identical, 0 = organ absent or "
            f"moved). Median over all slices of {n_patients} patients, band = 25–75%. Dotted lines (s1/s2/s4 under the axis): what a "
            "3-slice 2.5D input sees at stride 1, 2 and 4."
        ),
        footnote_text=(
            "Data: data/segthor_train_full/train, corrected labels (aorta separate), labels resampled along z to "
            f"{TARGET_MM:g} mm (nearest neighbour, as the 2.5D runs); in-plane resolution kept. Top/bottom = centre "
            f"slices within {EDGE_SLICES} slices of the organ's own most superior/inferior slice in that patient. "
            "Neighbours outside the scan are skipped."
        ),
    )
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"neighbour_dice_z.{ext}")
    plt.close(fig)
    print(f"wrote {out_dir / 'neighbour_dice_z.png'}")


def draw_single(df: pd.DataFrame, labels: list[int], n_patients: int, out_dir: Path) -> None:
    """One small panel per organ: mean copy-the-neighbour Dice against distance, middle vs near an end.

    Below and above are pooled (distance = |offset|), and the top and bottom ends are pooled. Panels share
    the y-axis and are ordered from the organ whose middle changes least to the one that changes most.

    Args:
        df: Output rows of neighbour_dice for every patient.
        labels: Labels with voxels in at least one patient.
        n_patients: Number of patients pooled.
        out_dir: Folder to write the figure into.
    """
    apply_style()
    df = df.assign(distance=df.offset.abs(), where=np.where(df.region == "middle", "middle", "end"))
    # past an organ end the neighbour is empty (Dice 0) for a fixed share of edge slices, the same for every
    # organ, which would hide the differences between organs; keep only neighbours that still show the organ
    df = df[(df["where"] == "middle") | (df.dice > 0)]
    means = df.groupby(["where", "label", "distance"]).dice.mean()
    order = sorted(labels, key=lambda c: -means["middle", c, MAX_OFFSET])
    x = np.arange(MAX_OFFSET + 1) * TARGET_MM

    fig, axes = plt.subplots(1, len(order), figsize=(14, 5.6), sharey=True)
    for ax, c in zip(axes, order):
        color = EARTH_LABEL_COLORS[c]
        middle, end = means["middle", c].to_numpy(), means["end", c].to_numpy()
        for s in STRIDES:
            ax.axvline(s * TARGET_MM, color="#BBBBBB", lw=0.9, ls=":", zorder=1)
        ax.fill_between(x, end, middle, color=tint(color, 0.18), lw=0, zorder=2)
        ax.plot(x, middle, color=color, lw=2.6, marker="o", ms=4.5, zorder=3)
        ax.plot(x, end, color=color, lw=2.6, ls="--", marker="o", ms=4.5, mfc="white", zorder=3)
        for y in (middle[-1], end[-1]):
            ax.annotate(f"{y:.2f}", (x[-1], y), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=10, color=color, fontweight="bold", annotation_clip=False)
        ax.set_title(NAMES[c], loc="left", fontsize=14, fontweight="bold", color=color)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" + (f"\ns{d}" if d in STRIDES else "\n") for d, v in enumerate(x)], fontsize=10)
        ax.set_xlim(-0.4, x[-1] + 0.4)
        ax.set_xlabel("distance (mm)")
    axes[0].set_ylim(0.4, 1.02)  # every mean is above 0.5
    axes[0].set_ylabel("Dice of copying the neighbour's mask")
    axes[0].plot([], [], color="#666666", lw=2.6, marker="o", ms=4.5, label="middle of organ")
    axes[0].plot([], [], color="#666666", lw=2.6, ls="--", marker="o", ms=4.5, mfc="white",
                 label=f"within {EDGE_SLICES * TARGET_MM:g} mm of an organ end")
    legend_below(axes[0], ncol=2)

    decorate(
        fig,
        "Neighbour Slices Are Near-Copies Inside the Heart; Thin Organs and Organ Ends Change Fastest",
        subtitle=(
            "How well the organ mask on a slice above or below matches this slice (1 = identical), by distance. "
            f"Mean over all slices of {n_patients} patients. Dotted lines s1/s2/s4: what a 3-slice 2.5D input sees "
            "at stride 1, 2 and 4. The shaded gap is how much more the neighbours differ near an organ end."
        ),
        footnote_text=(
            "Data: data/segthor_train_full/train, corrected labels (aorta separate), labels resampled along z to "
            f"{TARGET_MM:g} mm (nearest neighbour, as the 2.5D runs). Near an end = centre slices within "
            f"{EDGE_SLICES} slices of the organ's own top or bottom slice in that patient, both ends pooled. Dashed "
            "lines leave out neighbours past the organ's end (empty, Dice 0: 10/20/30/40% of edge neighbours at "
            "2.5/5/7.5/10 mm for every organ). Neighbours outside the scan are skipped."
        ),
    )
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"neighbour_dice_z_single.{ext}")
    plt.close(fig)
    print(f"wrote {out_dir / 'neighbour_dice_z_single.png'}")


def main():
    """Measure every patient, write the table and log, and draw the figure."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/segthor_train_full/train"))
    ap.add_argument("--out-dir", type=Path, default=Path("figures/dataset_40"))
    args = ap.parse_args()
    out_dir = args.out_dir / "neighbour_dice_z"
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()

    tables, geometry, failed = [], [], []
    for gt_path in sorted(args.data_dir.glob("Patient_*/GT.nii.gz")):
        try:
            seg, geo = resampled_labels(gt_path)
            tables.append(neighbour_dice(seg, geo["patient"]))
        except Exception as e:  # log and skip one bad patient instead of losing the run
            print(f"[skip] {gt_path}: {type(e).__name__}: {e}")
            failed.append({"patient": gt_path.parent.name, "error": f"{type(e).__name__}: {e}"})
            continue
        print(f"{geo['patient']}: {geo['orientation']} spacing {geo['spacing_mm']} shape {geo['shape']} "
              f"-> {geo['resampled_slices']} slices at {TARGET_MM} mm")
        geometry.append(geo)
    df = pd.concat(tables, ignore_index=True)
    df.to_csv(out_dir / "neighbour_dice_z.csv", index=False)

    labels = sorted(df.label.unique())
    draw(df, labels, len(geometry), out_dir)
    draw_single(df, labels, len(geometry), out_dir)

    summary = df[df.offset.abs().isin(STRIDES)].groupby(["region", "label", df.offset.abs()]).dice.median()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    log = {
        "data_dir": str(args.data_dir),
        "git_commit": commit,
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(),
        "target_mm": TARGET_MM,
        "edge_slices": EDGE_SLICES,
        "labels_drawn": [CLASSES[c] for c in labels],
        "median_dice_by_region_label_offset": {
            f"{r}/{CLASSES[c]}/±{o}": round(v, 3) for (r, c, o), v in summary.items()
        },
        "patients": geometry,
        "failed": failed,
    }
    (out_dir / "neighbour_dice_z_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps({k: log[k] for k in ("labels_drawn", "median_dice_by_region_label_offset", "failed")}, indent=2))


if __name__ == "__main__":
    main()
