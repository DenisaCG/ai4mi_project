#!/usr/bin/env python3
"""
Same figure as context_25d_input_slices.py, but with the slices as the ENet 2.5D models were fed them.

Reads the validation PNGs of the fold that holds the patient (data/sliced_<hash>/val/{img,gt}), i.e. after median-spacing
resampling (0.977 x 0.977 x 2.5 mm), the fold's HU window rescaled to 0..255, and the ROI crop/pad to 288x288. The network
additionally z-scores these values (a linear shift/scale, so the picture is the same up to the gray scale). The centre slice per
organ is again the slice with the most ground-truth voxels of that organ, now counted on the resampled grid, and the
outline is the resampled ground truth.

Writes into <out-dir>/: context_25d_input_slices_preprocessed.png/.pdf

Usage:
    python dataset_analysis/context_25d_input_slices_preprocessed.py --patient Patient_14
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_25d_input_slices import EXTRA, HALF_WIDTH_MM, OFFSETS, SHARED, patient_scores  # noqa: E402
from utils import NAMES  # noqa: E402
from plot_style import EARTH_LABEL_COLORS, apply_style, decorate  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RUN = "runs/full_cv4_enet_dice_ce_25d_c1_fold{fold}/seed0/config.yaml"  # only for the fold's data root
LABEL_SCALE = 63  # slice_segthor.py writes class k as k * 63


def orient(a: np.ndarray) -> np.ndarray:
    """PNG slice -> display orientation of the raw-CT figure (anterior up, patient left on the right).

    The PNGs are stored transposed relative to that (anterior to the left, mirrored), which is a transpose.
    """
    return a.T


def fold_root(patient: str) -> tuple[int, Path]:
    """Fold whose validation split holds the patient, and its processed-data root.

    Args:
        patient: Patient id.

    Returns:
        (fold, data root).
    """
    for fold in range(4):
        root = Path(yaml.safe_load(Path(RUN.format(fold=fold)).read_text())["data"]["root"])
        if (root / "val" / "img" / f"{patient}_0000.png").exists():
            return fold, root
    raise FileNotFoundError(f"{patient} is in no validation fold")


def main():
    """Load the processed slices, pick the slice per organ, draw the grid."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patient", default="Patient_01")
    ap.add_argument("--out-dir", type=Path, default=Path("figures/context_25d_input_slices_preprocessed"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fold, root = fold_root(args.patient)
    crop = json.loads((root / "roi_crop.json").read_text())
    stats = json.loads((root / "ct_norm_stats.json").read_text())
    sx, sy, sz = crop["target_spacing"]
    files = sorted((root / "val" / "img").glob(f"{args.patient}_*.png"))
    n = len(files)
    assert [f.stem for f in files] == [f"{args.patient}_{i:04d}" for i in range(n)], "slice numbering has gaps"
    ct = np.stack([np.array(Image.open(f).convert("L")) for f in files], axis=-1)  # (H, W, z)
    gt = np.stack([np.array(Image.open(root / "val" / "gt" / f.name)) for f in files], axis=-1) // LABEL_SCALE
    print(f"{args.patient}: fold {fold}, root {root}, {n} slices of {ct.shape[:2]}, spacing ({sx:.3f}, {sy:.3f}, {sz:g}) mm, "
          f"window {stats['lo']:.0f}..{stats['hi']:.0f} HU, labels {np.unique(gt).tolist()}")

    apply_style()
    fig, axes = plt.subplots(len(NAMES), len(OFFSETS), figsize=(2.3 * len(OFFSETS), 2.4 * len(NAMES)))
    for r, organ in enumerate(NAMES):
        mask = gt == organ
        z = int(np.argmax(mask.sum(axis=(0, 1))))
        ys, xs = np.nonzero(mask[:, :, z])
        cy, cx = int(ys.mean()), int(xs.mean())
        hy, hx = int(HALF_WIDTH_MM[organ] / sy), int(HALF_WIDTH_MM[organ] / sx)
        window = (slice(max(cy - hy, 0), cy + hy), slice(max(cx - hx, 0), cx + hx))
        for c, dz in enumerate(OFFSETS):
            ax = axes[r, c]
            zz = min(max(z + dz, 0), n - 1)  # edge slices repeated, as in the dataset
            ax.imshow(orient(ct[:, :, zz][window]), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            m = orient(mask[:, :, zz][window])
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
        f"{args.patient.replace('_', ' ')}: ENet 2.5D Inputs After Pre-processing",
        subtitle=(
            f"Mean over the four organs (CV, 3 seeds) - {patient_scores(args.patient)}. "
            f"One row per organ at its largest ground-truth slice. Slices resampled to {sz:g} mm apart, so the 5-slice input "
            f"spans ±{2 * sz:g} mm."
        ),
        footnote_text=(
            f"Validation PNGs of fold {fold} ({root}): median-spacing resample ({sx:.3f} x {sy:.3f} x {sz:g} mm), fold HU window "
            f"{stats['lo']:.0f} to {stats['hi']:.0f} mapped to 0..255, ROI crop/pad to {crop['size']}x{crop['size']}; the network "
            f"also z-scores (mean {stats['mean']:.0f}, std {stats['std']:.0f} HU). Display crop differs per organ."
        ),
    )
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"context_25d_input_slices_preprocessed.{ext}")
    plt.close(fig)
    print(f"wrote {args.out_dir / 'context_25d_input_slices_preprocessed.png'}")


if __name__ == "__main__":
    main()
