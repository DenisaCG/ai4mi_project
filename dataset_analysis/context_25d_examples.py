#!/usr/bin/env python3
"""
Esophagus slices where 2.5D (stride 4, ±10 mm) loses the most Dice against 2D, to see why.

For each patient, finds the validation fold that holds it, loads seed 0 of both experiments
(runs/<exp>_fold<k>/seed0/volumes/val), scores esophagus Dice per slice, and picks the SLICES_PER_PATIENT
slices with the largest 2D - 2.5D drop (at least MIN_GAP slices apart). Each column is one slice,
cropped around the esophagus:
  row 1  CT with the ground-truth outline (solid) and the ground truth 10 mm above and below (dashed),
         i.e. what the stride-4 neighbour channels show
  row 2  2D prediction (filled) against the ground-truth outline
  row 3  2.5D prediction (filled), same outlines plus the dashed neighbour outlines

Writes into <out-dir>/: context_25d_examples.png/.pdf and context_25d_examples.csv (chosen slices, Dice).

Usage:
    python dataset_analysis/context_25d_examples.py --patients Patient_11 Patient_33 Patient_32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import NAMES  # noqa: E402  (also puts tools/ on sys.path)
from plot_style import EARTH_LABEL_COLORS, PALETTE, apply_style, decorate  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ORGAN = 1  # esophagus
NEIGHBOUR_MM = 10.0
SLICES_PER_PATIENT = 3
MIN_GAP = 3
HALF_WIDTH_MM = 30.0
CT_WINDOW = (-160, 240)  # soft tissue, HU
DATA = "data/segthor_train_full/train/{patient}/{name}.nii.gz"
ABOVE, BELOW = PALETTE[6], PALETTE[4]


def val_fold(runs_dir: Path, experiment: str, patient: str) -> int:
    """Fold whose validation split holds the patient.

    Args:
        runs_dir: Directory with <experiment>_fold<k>/seed0.
        experiment: Experiment name.
        patient: Patient id.

    Returns:
        The fold number.
    """
    for k in range(4):
        if (runs_dir / f"{experiment}_fold{k}" / "seed0" / "volumes" / "val" / f"{patient}.nii.gz").exists():
            return k
    raise FileNotFoundError(f"{patient} not in any validation fold of {experiment}")


def slice_dice(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Dice per axial slice (NaN where both are empty).

    Args:
        gt: Boolean ground truth, (x, y, z).
        pred: Boolean prediction, (x, y, z).

    Returns:
        Dice per slice, shape (z,).
    """
    inter = np.logical_and(gt, pred).sum(axis=(0, 1))
    denom = gt.sum(axis=(0, 1)) + pred.sum(axis=(0, 1))
    with np.errstate(invalid="ignore"):
        return np.where(denom > 0, 2 * inter / np.maximum(denom, 1), np.nan)


def pick_slices(drop: np.ndarray, candidates: np.ndarray) -> list[int]:
    """Largest-drop slices, at least MIN_GAP apart.

    Args:
        drop: 2D minus 2.5D Dice per slice.
        candidates: Slice indices with ground truth.

    Returns:
        Up to SLICES_PER_PATIENT slice indices, superior first.
    """
    chosen = []
    for z in candidates[np.argsort(-drop[candidates])]:
        if all(abs(z - c) >= MIN_GAP for c in chosen):
            chosen.append(int(z))
        if len(chosen) == SLICES_PER_PATIENT:
            break
    return sorted(chosen, reverse=True)


def main():
    """Pick the slices, draw the comparison grid and write the table."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", nargs="+", default=["Patient_11", "Patient_33", "Patient_32"])
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--base", default="full_cv4_resenc_ds_unet_dice_ce")
    ap.add_argument("--context", default="full_cv4_resenc_ds_unet_dice_ce_25d_c1_s4")
    ap.add_argument("--out-dir", type=Path, default=Path("figures/context_25d_regions"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    panels, rows = [], []
    for patient in args.patients:
        fold = val_fold(args.runs_dir, args.base, patient)
        assert fold == val_fold(args.runs_dir, args.context, patient), f"{patient}: folds differ"
        gt_nii = nib.load(DATA.format(patient=patient, name="GT"))
        codes = "".join(nib.aff2axcodes(gt_nii.affine))
        assert codes == "LPS", f"{patient}: orientation {codes}"
        sx, sy, sz = (float(s) for s in gt_nii.header.get_zooms()[:3])
        gt = np.asarray(gt_nii.dataobj) == ORGAN
        ct = np.asarray(nib.load(DATA.format(patient=patient, name=patient)).dataobj)
        preds = {}
        for key, exp in (("2d", args.base), ("25d", args.context)):
            nii = nib.load(args.runs_dir / f"{exp}_fold{fold}" / "seed0" / "volumes" / "val" / f"{patient}.nii.gz")
            assert nii.shape == gt.shape, f"{patient} {exp}: shape {nii.shape} != {gt.shape}"
            preds[key] = np.asarray(nii.dataobj) == ORGAN
        d2, d25 = slice_dice(gt, preds["2d"]), slice_dice(gt, preds["25d"])
        candidates = np.flatnonzero(gt.any(axis=(0, 1)))
        k = int(round(NEIGHBOUR_MM / sz))
        print(f"{patient}: fold {fold}, spacing ({sx:.3f}, {sy:.3f}, {sz:.2f}) mm, {codes}, "
              f"neighbours at ±{k} slices, volume Dice 2D {np.nanmean(d2[candidates]):.3f} "
              f"2.5D {np.nanmean(d25[candidates]):.3f}")
        for z in pick_slices(np.nan_to_num(d2 - d25), candidates):
            above, below = min(z + k, gt.shape[2] - 1), max(z - k, 0)  # LPS: +z is superior
            ys, xs = np.nonzero(gt[:, :, z].T)
            cy, cx = int(ys.mean()), int(xs.mean())
            hy, hx = int(HALF_WIDTH_MM / sy), int(HALF_WIDTH_MM / sx)
            window = (slice(max(cy - hy, 0), cy + hy), slice(max(cx - hx, 0), cx + hx))
            panels.append(
                {
                    "title": f"{patient.replace('_', ' ')}, slice {z}",
                    "ct": ct[:, :, z].T[window],
                    "gt": gt[:, :, z].T[window],
                    "above": gt[:, :, above].T[window],
                    "below": gt[:, :, below].T[window],
                    "2d": preds["2d"][:, :, z].T[window],
                    "25d": preds["25d"][:, :, z].T[window],
                    "dice_2d": d2[z],
                    "dice_25d": d25[z],
                }
            )
            rows.append({"patient": patient, "fold": fold, "slice": z, "dice_2d": d2[z], "dice_25d": d25[z],
                         "gt_pixels": int(gt[:, :, z].sum()), "pred_2d_pixels": int(preds["2d"][:, :, z].sum()),
                         "pred_25d_pixels": int(preds["25d"][:, :, z].sum())})
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "context_25d_examples.csv", index=False)
    print(table.round(3).to_string(index=False))

    apply_style()
    color = EARTH_LABEL_COLORS[ORGAN]
    fig, axes = plt.subplots(3, len(panels), figsize=(2.3 * len(panels), 8.6))
    row_titles = ["ground truth\n(dashed: ±10 mm)", "2D", "2.5D stride 4"]
    for col, p in enumerate(panels):
        for r in range(3):
            ax = axes[r, col]
            ax.imshow(p["ct"], cmap="gray", vmin=CT_WINDOW[0], vmax=CT_WINDOW[1], interpolation="nearest")
            if r > 0:
                pred = p["2d" if r == 1 else "25d"]
                overlay = np.zeros((*pred.shape, 4))
                overlay[pred] = (*to_rgb(color), 0.6)
                ax.imshow(overlay, interpolation="nearest")
                dice = p["dice_2d" if r == 1 else "dice_25d"]
                ax.text(0.03, 0.97, f"Dice {dice:.2f}", transform=ax.transAxes, va="top", fontsize=9,
                        color="white", fontweight="bold")
            if r != 1:
                for key, c in (("above", ABOVE), ("below", BELOW)):
                    if p[key].any():
                        ax.contour(p[key], levels=[0.5], colors=[c], linewidths=1.3, linestyles="--")
            ax.contour(p["gt"], levels=[0.5], colors=["white"], linewidths=1.4)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for side in ax.spines.values():
                side.set_visible(False)
            if r == 0:
                ax.set_title(p["title"], fontsize=9.5)
            if col == 0:
                ax.set_ylabel(row_titles[r], fontsize=11, fontweight="bold")
    handles = [
        Line2D([], [], color="#333333", lw=1.4, label="ground truth, this slice"),
        Line2D([], [], color=ABOVE, lw=1.3, ls="--", label=f"ground truth {NEIGHBOUR_MM:g} mm above"),
        Line2D([], [], color=BELOW, lw=1.3, ls="--", label=f"ground truth {NEIGHBOUR_MM:g} mm below"),
        Patch(color=color, alpha=0.6, label=f"predicted {NAMES[ORGAN].lower()}"),
    ]
    # decorate() draws this as the below-plot legend (legend_below needs plotted artists)
    fig._pending_legend = (handles, [h.get_label() for h in handles], 4)

    decorate(
        fig,
        f"{NAMES[ORGAN]} Slices Where 2.5D (Stride 4) Loses the Most Dice Against 2D",
        subtitle=(
            f"Residual-encoder U-Net, seed 0, validation fold of each patient; {SLICES_PER_PATIENT} slices per patient "
            f"with the largest per-slice Dice drop, cropped ±{HALF_WIDTH_MM:g} mm around the {NAMES[ORGAN].lower()}. "
            "If the 2.5D prediction leans toward the dashed outlines, the neighbour channels are pulling it there."
        ),
        footnote_text=(
            "Data: data/segthor_train_full/train, corrected labels; predictions are the stitched validation volumes "
            f"on the original CT grid. CT window {CT_WINDOW[0]} to {CT_WINDOW[1]} HU. Axial view, anterior up, "
            "patient left on the right. Dashed outlines are the ground truth on the slices the stride-4 input sees."
        ),
    )
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"context_25d_examples.{ext}")
    plt.close(fig)
    print(f"wrote {args.out_dir / 'context_25d_examples.png'}")


if __name__ == "__main__":
    main()
