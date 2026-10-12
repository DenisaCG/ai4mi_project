#!/usr/bin/env python3
"""
One slide figure introducing the 40-patient dataset: (a) the 20 added patients look like the original 20,
(b) the release also cleaned the labels of the original 20, (c) the evaluation moved from one 15/5 split to
4-fold patient-level cross-validation.

Inputs:
  figures/dataset_40/patient_labels.csv   organ volume and 3D components per patient, full 40-patient release
                                          (written by tools/dataset_40_tables.py)
  figures/dataset_40/comparison.csv       original_20 vs added_20 test per quantity (same script)
  data/segthor_part1_corrected/train/Patient_*/GT.nii.gz   the midterm labels of the first 20 patients, whose
                                          3D components are counted here (face connectivity, as components_face)

Writes figures/dataset_40/dataset_overview_slide.png/.pdf and dataset_overview_slide.csv (the counts of panel b).

Usage:
    python dataset_analysis/dataset_40_slide_figure.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import generate_binary_structure, label

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import CLASSES, COLORS, NAMES  # noqa: E402  (also puts tools/ on sys.path and sets the Agg backend)
from plot_style import apply_style, decorate, tint  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

INK, MUTED = "#333333", "#8C8C8C"
TRAIN, VAL = "#D9D4CA", "#2E5E6E"


def midterm_components(midterm_dir: Path) -> pd.DataFrame:
    """3D connected components (face connectivity) of every organ in the midterm labels.

    Args:
        midterm_dir: Folder with one Patient_* folder per patient holding GT.nii.gz.

    Returns:
        One row per patient and organ with column components_midterm.
    """
    rows = []
    structure = generate_binary_structure(3, 1)
    for gt_path in sorted(midterm_dir.glob("Patient_*/GT.nii.gz")):
        nii = nib.load(gt_path)
        codes = "".join(nib.aff2axcodes(nii.affine))
        print(f"{gt_path.parent.name}: {codes}, spacing {np.round(nii.header.get_zooms()[:3], 3).tolist()}",
              flush=True)
        seg = np.asarray(nii.dataobj)
        for k in CLASSES:
            mask = seg == k
            if mask.any():  # crop to the organ's bounding box: same count, far less work
                mask = mask[tuple(slice(i.min(), i.max() + 1) for i in np.nonzero(mask))]
            rows.append({"patient": gt_path.parent.name, "label": k,
                         "components_midterm": label(mask, structure)[1]})
    return pd.DataFrame(rows)


def panel_population(ax, labels: pd.DataFrame, comparison: pd.DataFrame, note: bool = True) -> None:
    """(a) Organ volume per patient, original 20 (filled) above added 20 (hollow), log scale.

    Args:
        ax: Axes to draw on.
        labels: patient_labels.csv.
        comparison: comparison.csv.
        note: Write the group-test note under the axes (the slide version puts it on the slide instead).
    """
    rng = np.random.default_rng(0)
    ticks = []
    for i, k in enumerate(CLASSES):
        for j, (group, filled) in enumerate((("original_20", True), ("added_20", False))):
            v = labels[(labels.label == k) & (labels.group == group)].volume_ml
            y = 2.6 * i + 0.9 * j
            ax.scatter(v, y + rng.uniform(-0.18, 0.18, len(v)), s=26, marker="o" if filled else "s",
                       facecolor=COLORS[k] if filled else "white", edgecolor=COLORS[k], lw=1.2, zorder=3)
            ax.plot([v.median()] * 2, [y - 0.32, y + 0.32], color=INK, lw=1.6, zorder=4)
        ticks.append((2.6 * i + 0.45, NAMES[k]))
    ax.set_xscale("log")
    ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000])
    ax.set_xticklabels(["20", "50", "100", "200", "500", "1000", "2000"])
    ax.set_yticks([t for t, _ in ticks])
    ax.set_yticklabels([n for _, n in ticks], fontweight="bold")
    for tick, k in zip(ax.get_yticklabels(), CLASSES):
        tick.set_color(COLORS[k])
    ax.set_ylim(2.6 * len(CLASSES) - 1.2, -0.7)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color="#E6E6E6")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("organ volume (mL, log scale)")
    n_diff, n_all = int(comparison.differs.sum()), len(comparison)
    ax.set_title("(a) Twice the patients, same population", loc="left", fontsize=13, fontweight="bold")
    if note:
        ax.text(0, -0.13, f"{n_diff} of {n_all} measured properties differ between the two groups (p < 0.05);\n"
                f"about {round(0.05 * n_all)} would by chance alone. Bar = median.", transform=ax.transAxes,
                va="top", fontsize=9.5, color="#555555")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", markerfacecolor=MUTED, markeredgecolor=MUTED, label="original 20 (midterm)"),
        Line2D([], [], marker="s", ls="", markerfacecolor="white", markeredgecolor=MUTED, label="added 20"),
    ], loc="upper right", fontsize=9.5, frameon=False, handletextpad=0.2)


def panel_labels(ax, counts: pd.DataFrame) -> None:
    """(b) Patients (of the original 20) whose organ label is more than one 3D piece, midterm vs. now.

    Args:
        ax: Axes to draw on.
        counts: One row per patient and organ with components_midterm and components_face.
    """
    x = np.arange(len(CLASSES))
    for i, k in enumerate(CLASSES):
        c = counts[counts.label == k]
        before, after = int((c.components_midterm > 1).sum()), int((c.components_face > 1).sum())
        ax.bar(i - 0.2, before, width=0.38, color=tint(COLORS[k], 0.45), edgecolor=COLORS[k], lw=1.2, zorder=3)
        ax.bar(i + 0.2, max(after, 0.04), width=0.38, color=COLORS[k], edgecolor=COLORS[k], lw=1.2, zorder=3)
        ax.text(i - 0.2, before + 0.15, str(before), ha="center", va="bottom", fontsize=10.5, color=INK)
        ax.text(i + 0.2, after + 0.15, str(after), ha="center", va="bottom", fontsize=10.5, color=INK,
                fontweight="bold")
        if before:
            ax.text(i - 0.2, before + 1.0, f"up to\n{int(c.components_midterm.max())} pieces", ha="center",
                    va="bottom", fontsize=8.5, color="#666666", linespacing=0.95)
    ax.set_xticks(x)
    ax.set_xticklabels([NAMES[k] for k in CLASSES], fontweight="bold")
    for tick, k in zip(ax.get_xticklabels(), CLASSES):
        tick.set_color(COLORS[k])
    top = int(counts.groupby("label").components_midterm.apply(lambda c: (c > 1).sum()).max()) + 3
    ax.set_ylim(0, top)
    ax.set_yticks(range(0, top + 1, 2))
    ax.set_ylabel("patients with the label\nin 2+ pieces (of 20)")
    ax.set_title("(b) Cleaner labels for the original 20", loc="left", fontsize=13, fontweight="bold")
    ax.legend(handles=[
        Rectangle((0, 0), 1, 1, facecolor=tint(MUTED, 0.45), edgecolor=MUTED, label="midterm labels"),
        Rectangle((0, 0), 1, 1, facecolor=MUTED, edgecolor=MUTED, label="new release"),
    ], loc="upper center", fontsize=9.5, frameon=False)


def panel_protocol(ax) -> None:
    """(c) Schematic: one 15/5 split of 20 patients vs. 4-fold cross-validation of 40 patients.

    Args:
        ax: Axes to draw on.
    """
    ax.set_xlim(0, 40)
    ax.set_ylim(7.6, -1.6)
    ax.axis("off")
    w = 0.86

    def row(y, n, val, label_text):
        for p in range(n):
            ax.add_patch(Rectangle((p, y), w, 0.78, facecolor=VAL if p in val else TRAIN, lw=0))
        ax.text(-0.6, y + 0.39, label_text, ha="right", va="center", fontsize=10, color=INK)

    ax.text(0, -0.75, "Midterm: one split, 20 patients", fontsize=11, fontweight="bold", color=INK)
    row(0, 20, set(range(15, 20)), "split")
    ax.text(21, 0.39, "15 train · 5 val", va="center", fontsize=10, color="#555555")
    ax.text(0, 2.05, "Now: 4-fold cross-validation, 40 patients", fontsize=11, fontweight="bold", color=INK)
    for f in range(4):
        row(2.8 + 1.05 * f, 40, set(range(10 * f, 10 * f + 10)), f"fold {f}")
    ax.text(0, 7.25, "30 train · 10 val per fold; every patient validated once; × 3 seeds = 12 runs per model",
            fontsize=10, color="#555555")
    ax.set_title("(c) A fairer evaluation", loc="left", fontsize=13, fontweight="bold")
    ax.legend(handles=[Rectangle((0, 0), 1, 1, facecolor=TRAIN, label="training patient"),
                       Rectangle((0, 0), 1, 1, facecolor=VAL, label="validation patient")],
              loc="upper right", bbox_to_anchor=(1.0, 1.02), fontsize=9.5, frameon=False, ncol=2)


def main():
    """Count the midterm components, draw the three panels and write the panel (b) table."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", type=Path, default=Path("figures/dataset_40"))
    ap.add_argument("--midterm-dir", type=Path, default=Path("data/segthor_part1_corrected/train"))
    ap.add_argument("--out-dir", type=Path, default=Path("figures/dataset_40"))
    args = ap.parse_args()
    labels = pd.read_csv(args.tables / "patient_labels.csv")
    comparison = pd.read_csv(args.tables / "comparison.csv")
    counts = midterm_components(args.midterm_dir).merge(
        labels[labels.group == "original_20"][["patient", "label", "components_face"]], on=["patient", "label"])
    assert counts.patient.nunique() == 20, f"expected the 20 original patients, got {counts.patient.nunique()}"
    counts.to_csv(args.out_dir / "dataset_overview_slide.csv", index=False)
    print(counts.groupby("label")[["components_midterm", "components_face"]].agg(["max", lambda c: (c > 1).sum()]))

    apply_style()
    fig = plt.figure(figsize=(16, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.95, 1.25], wspace=0.32)
    panel_population(fig.add_subplot(gs[0]), labels, comparison)
    panel_labels(fig.add_subplot(gs[1]), counts)
    panel_protocol(fig.add_subplot(gs[2]))
    decorate(
        fig,
        "The New Dataset: Twice the Patients, Cleaner Labels and a Fairer Evaluation",
        subtitle="SegTHOR training release, 40 patients with corrected labels (esophagus, heart, trachea, aorta), "
                 "against the 20-patient midterm data.",
        footnote_text=(
            "(a) organ volume = voxel count × voxel volume. (b) pieces = 3D connected components, voxels touching by a "
            "face; midterm labels from data/segthor_part1_corrected, new from data/segthor_train_full. (c) schematic: "
            "the actual folds are patient-level splits fixed by seed 0, not consecutive patients."
        ),
    )
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"dataset_overview_slide.{ext}")
    plt.close(fig)
    print(f"wrote {args.out_dir / 'dataset_overview_slide.png'}")

    # slide version: panels (a) and (b) only, no title (the slide has one), shaped like the deck's image area;
    # cross-validation has its own slide
    apply_style()
    fig = plt.figure(figsize=(13.4, 3.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.3)
    panel_population(fig.add_subplot(gs[0]), labels, comparison, note=False)
    panel_labels(fig.add_subplot(gs[1]), counts)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"dataset_overview_slide_panels.{ext}", dpi=300, bbox_inches="tight",
                    pad_inches=0.08)
    plt.close(fig)
    print(f"wrote {args.out_dir / 'dataset_overview_slide_panels.png'}")


if __name__ == "__main__":
    main()
