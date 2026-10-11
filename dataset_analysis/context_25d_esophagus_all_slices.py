#!/usr/bin/env python3
"""
Every slice that contains esophagus for one patient, as the ENet 2.5D models were fed them (pre-processed validation PNGs).

Each panel is one axial slice (one 2.5D sample is this slice plus its neighbours), ordered inferior -> superior, with the
resampled ground truth of every organ outlined. All panels share one crop (the union of the esophagus bounding boxes plus a margin),
so the esophagus can be followed from slice to slice.

Writes into <out-dir>/: context_25d_esophagus_all_slices.png/.pdf

Usage:
    python dataset_analysis/context_25d_esophagus_all_slices.py --patient Patient_14
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_25d_input_slices_preprocessed import LABEL_SCALE, fold_root, orient  # noqa: E402
from utils import NAMES  # noqa: E402
from plot_style import EARTH_LABEL_COLORS, apply_style, decorate  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ORGAN = 1  # esophagus
COLUMNS = 10
MARGIN_MM = 12.0


def main():
    """Load the processed slices, crop around the esophagus, draw one panel per slice."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patient", default="Patient_01")
    ap.add_argument("--out-dir", type=Path, default=Path("figures/context_25d_esophagus_all_slices"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fold, root = fold_root(args.patient)
    sx, sy, sz = json.loads((root / "roi_crop.json").read_text())["target_spacing"]
    files = sorted((root / "val" / "img").glob(f"{args.patient}_*.png"))
    assert [f.stem for f in files] == [f"{args.patient}_{i:04d}" for i in range(len(files))], "slice numbering has gaps"
    ct = np.stack([np.array(Image.open(f).convert("L")) for f in files], axis=-1)
    gt = np.stack([np.array(Image.open(root / "val" / "gt" / f.name)) for f in files], axis=-1) // LABEL_SCALE
    mask = gt == ORGAN
    zs = np.flatnonzero(mask.any(axis=(0, 1)))
    assert len(zs) > 0, f"{args.patient}: no {NAMES[ORGAN]} voxels"
    assert np.all(np.diff(zs) == 1), f"{args.patient}: {NAMES[ORGAN]} slices are not contiguous: {zs.tolist()}"
    xs_any, ys_any = np.flatnonzero(mask.any(axis=(1, 2))), np.flatnonzero(mask.any(axis=(0, 2)))
    my, mx = int(MARGIN_MM / sy), int(MARGIN_MM / sx)
    rows_w = slice(max(xs_any.min() - my, 0), xs_any.max() + my)
    cols_w = slice(max(ys_any.min() - mx, 0), ys_any.max() + mx)
    print(f"{args.patient}: fold {fold}, {len(files)} slices, esophagus in {len(zs)} (z {zs[0]}..{zs[-1]})")

    apply_style()
    n_rows = math.ceil(len(zs) / COLUMNS)
    fig, axes = plt.subplots(n_rows, COLUMNS, figsize=(1.9 * COLUMNS, 1.9 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, z in zip(axes.ravel(), zs):
        ax.imshow(orient(ct[:, :, z][rows_w, cols_w]), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        for organ in NAMES:
            m = orient((gt[:, :, z] == organ)[rows_w, cols_w])
            if m.any():
                ax.contour(m, levels=[0.5], colors=[EARTH_LABEL_COLORS[organ]], linewidths=1.2)
        ax.set_title(f"z {z}", fontsize=8.5)
    handles = [Line2D([], [], color=EARTH_LABEL_COLORS[o], lw=2, label=NAMES[o]) for o in NAMES]
    fig._pending_legend = (handles, [h.get_label() for h in handles], len(handles))
    decorate(
        fig,
        f"{args.patient.replace('_', ' ')}: All {len(zs)} Slices With Esophagus",
        subtitle=(
            f"Pre-processed slices (inferior to superior, left to right, then down), {sz:g} mm apart, so the esophagus spans "
            f"{(len(zs) - 1) * sz:g} mm. Each panel is the centre of one 2.5D sample; outlines are the ground truth of all organs."
        ),
        footnote_text=(
            f"Validation PNGs of fold {fold} ({root}): median-spacing resample, HU window rescaled to 0..255, ROI crop/pad. "
            "One shared crop for all panels. Axial view, anterior up, patient left on the right."
        ),
    )
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"context_25d_esophagus_all_slices.{ext}")
    plt.close(fig)
    print(f"wrote {args.out_dir / 'context_25d_esophagus_all_slices.png'}")


if __name__ == "__main__":
    main()
