#!/usr/bin/env python3
"""
The CT slices the ENet 2.5D inputs actually contain, one row per organ.

The 3-slice model (context 1) sees z-1, z, z+1 as channels, the 5-slice model (context 2) sees z-2 ... z+2; both predict
only slice z. For each organ, z is the slice of the given patient with the largest ground-truth area of that organ. The three
slices shared by both models are framed in one colour, the two extra slices of the 5-slice model in another, and the
ground truth of the organ is outlined on every slice. Panels are cropped around the organ.

Uses the raw CT of data/segthor_train_full (original spacing), not the resampled/normalised network input.

Writes into <out-dir>/: context_25d_input_slices.png/.pdf

Usage:
    python dataset_analysis/context_25d_input_slices.py --patient Patient_01
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
from matplotlib.lines import Line2D  # noqa: E402

CT_WINDOW = (-160, 240)  # soft tissue, HU
DATA = "data/segthor_train_full/train/{patient}/{name}.nii.gz"
HALF_WIDTH_MM = {1: 30.0, 2: 80.0, 3: 30.0, 4: 40.0}  # per organ: esophagus, heart, trachea, aorta
OFFSETS = (-2, -1, 0, 1, 2)
SHARED, EXTRA = PALETTE[0], PALETTE[6]
METRICS = "metrics/cv_pooled_full_cv4_enet_dice_ce_25d_c{context}.csv"  # per patient and organ, mean over 3 seeds


def patient_scores(patient: str) -> str:
    """Mean Dice / HD95 / ASSD of the patient over the four organs, for both ENet 2.5D models.

    Args:
        patient: Patient id.

    Returns:
        One line, e.g. "3-slice: Dice 0.90, HD95 5.1 mm, ASSD 1.2 mm  |  5-slice: ...".
    """
    parts = []
    for context, label in ((1, "3-slice"), (2, "5-slice")):
        df = pd.read_csv(METRICS.format(context=context))
        df = df[df["patient"] == patient]
        assert df["class_name"].nunique() == len(NAMES), f"{patient}: {df['class_name'].tolist()}"
        m = df[["dice", "hd95", "assd"]].mean()
        parts.append(f"{label}: Dice {m['dice']:.3f}, HD95 {m['hd95']:.1f} mm, ASSD {m['assd']:.2f} mm")
    return "  |  ".join(parts)


def main():
    """Pick the slice per organ, draw the grid."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patient", default="Patient_01")
    ap.add_argument("--out-dir", type=Path, default=Path("figures/context_25d_input_slices"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gt_nii = nib.load(DATA.format(patient=args.patient, name="GT"))
    codes = "".join(nib.aff2axcodes(gt_nii.affine))
    assert codes == "LPS", f"{args.patient}: orientation {codes}"
    sx, sy, sz = (float(s) for s in gt_nii.header.get_zooms()[:3])
    print(f"{args.patient}: spacing ({sx:.3f}, {sy:.3f}, {sz:.2f}) mm, {codes}")
    gt = np.asarray(gt_nii.dataobj)
    ct = np.asarray(nib.load(DATA.format(patient=args.patient, name=args.patient)).dataobj)
    assert ct.shape == gt.shape, f"CT {ct.shape} != GT {gt.shape}"

    apply_style()
    organs = list(NAMES)
    fig, axes = plt.subplots(len(organs), len(OFFSETS), figsize=(2.3 * len(OFFSETS), 2.4 * len(organs)))
    for r, organ in enumerate(organs):
        mask = gt == organ
        z = int(np.argmax(mask.sum(axis=(0, 1))))
        ys, xs = np.nonzero(mask[:, :, z].T)
        cy, cx = int(ys.mean()), int(xs.mean())
        hy, hx = int(HALF_WIDTH_MM[organ] / sy), int(HALF_WIDTH_MM[organ] / sx)
        window = (slice(max(cy - hy, 0), cy + hy), slice(max(cx - hx, 0), cx + hx))
        for c, dz in enumerate(OFFSETS):
            ax = axes[r, c]
            zz = min(max(z + dz, 0), ct.shape[2] - 1)  # edge slices repeated, as in the dataset
            ax.imshow(ct[:, :, zz].T[window], cmap="gray", vmin=CT_WINDOW[0], vmax=CT_WINDOW[1], interpolation="nearest")
            m = mask[:, :, zz].T[window]
            if m.any():
                ax.contour(m, levels=[0.5], colors=[EARTH_LABEL_COLORS[organ]], linewidths=1.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            frame = SHARED if abs(dz) <= 1 else EXTRA
            for side in ax.spines.values():
                side.set_visible(True)
                side.set_color(frame)
                side.set_linewidth(3)
            ax.set_title(f"z{dz:+d}" if dz else f"z (target, slice {z})", fontsize=9.5)
            if c == 0:
                ax.set_ylabel(NAMES[organ], fontsize=11, fontweight="bold")

    handles = [
        Line2D([], [], color=SHARED, lw=3, label="input of both 3- and 5-slice models"),
        Line2D([], [], color=EXTRA, lw=3, label="extra input of the 5-slice model only"),
        Line2D([], [], color="#888888", lw=1.5, label="ground-truth outline of the organ"),
    ]
    fig._pending_legend = (handles, [h.get_label() for h in handles], 3)
    decorate(
        fig,
        f"{args.patient.replace('_', ' ')}: What the 3-Slice and 5-Slice ENet Inputs Contain",
        subtitle=(
            f"Mean over the four organs (CV, 3 seeds) - {patient_scores(args.patient)}. "
            f"One row per organ at its largest ground-truth slice. Neighbouring slices are "
            f"{sz:g} mm apart, so even the 5-slice input spans only ±{2 * sz:g} mm."
        ),
        footnote_text=(
            f"Raw CT from data/segthor_train_full (corrected labels), not the resampled network input. CT window "
            f"{CT_WINDOW[0]} to {CT_WINDOW[1]} HU. Axial view, anterior up, patient left on the right. Crop differs per organ."
        ),
    )
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"context_25d_input_slices.{ext}")
    plt.close(fig)
    print(f"wrote {args.out_dir / 'context_25d_input_slices.png'}")


if __name__ == "__main__":
    main()
