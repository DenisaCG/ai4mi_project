#!/usr/bin/env python3
"""
Label size and intensity: every patient's label 1, 2 and 3 drawn in 3D at the
same physical scale and sorted by volume, above one panel overlapping the HU
histograms of the three labels and the background. Each patient has the same
shade in both parts (light = smallest volume, dark = largest).

Reads labels.csv, patients.csv and intensity_histograms.npz written by
tools/dataset_profile.py, and the label masks from the data directory.

Usage:
    python dataset_analysis/profile_figures/f06_07_label_size_intensity.py --profile-dir figures/profile \
        --data-dir data/segthor_part1/train
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sns
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import zoom
from skimage.measure import marching_cubes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from dataset_profile import load_tables  # noqa: E402
from profile_figures.common import LABEL_COLORS, apply_ticks_style, build_arg_parser, header, out_subdir  # noqa: E402
from profile_figures.f05_label_intensity import binned_counts  # noqa: E402

LABELS = [1, 2, 3]
COLORS = {**LABEL_COLORS, "background": "#9A9A9A"}
MESH_MM = 2.0  # isotropic grid the masks are resampled to before meshing
HALF_WIDTH_MM, HALF_HEIGHT_MM = 70, 175  # shared 3D box for every shape
HU_RANGE = (-1030, 500)
BIN_HU = 10
LIGHT = np.array([-0.4, -0.6, 0.7]) / np.linalg.norm([-0.4, -0.6, 0.7])


def shades(color: str, n: int) -> list[np.ndarray]:
    """Blend a colour towards white, from lightest (index 0) to the full colour (index n-1).

    Args:
        color: Any matplotlib colour.
        n: Number of shades.

    Returns:
        n RGB arrays.
    """
    base = np.array(to_rgb(color))
    return [base + (1 - base) * t for t in np.linspace(0.78, 0.0, n)]


def resample_mask(mask: np.ndarray, zooms: np.ndarray, step_mm: float) -> np.ndarray:
    """Crop a boolean mask to its bounding box and resample it to an isotropic grid.

    Args:
        mask: Boolean mask with at least one True voxel.
        zooms: Voxel spacing (x, y, z) in mm.
        step_mm: Output voxel size in mm.

    Returns:
        Boolean mask on a step_mm grid covering the bounding box.
    """
    idx = np.argwhere(mask)
    lo, hi = idx.min(axis=0), idx.max(axis=0) + 1
    sub = mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]].astype(np.float32)
    return zoom(sub, np.asarray(zooms) / step_mm, order=1) > 0.5


def draw_shape(ax, mask: np.ndarray, color: np.ndarray) -> None:
    """Draw a resampled mask as a shaded surface centred in a shared 3D box."""
    verts, faces, _, _ = marching_cubes(np.pad(mask, 1).astype(np.float32), 0.5, spacing=(MESH_MM,) * 3)
    tri = (verts - MESH_MM - np.array(mask.shape) * MESH_MM / 2)[faces]
    tri[..., 0] *= -1  # array x runs towards patient left; flip so patient left is on the viewer's right
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
    ax.add_collection3d(
        Poly3DCollection(tri, facecolors=np.outer(0.5 + 0.5 * np.abs(normals @ LIGHT), color), linewidths=0)
    )
    ax.set_xlim(-HALF_WIDTH_MM, HALF_WIDTH_MM)
    ax.set_ylim(-HALF_WIDTH_MM, HALF_WIDTH_MM)
    ax.set_zlim(-HALF_HEIGHT_MM, HALF_HEIGHT_MM)
    ax.set_box_aspect((1, 1, HALF_HEIGHT_MM / HALF_WIDTH_MM), zoom=1.2)
    ax.set_axis_off()
    ax.view_init(elev=12, azim=-60)


def main():
    """Draw the label size and intensity figure."""
    ap = build_arg_parser(__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/segthor_part1/train"))
    args = ap.parse_args()

    tables = load_tables(args.profile_dir)
    hists = tables["histograms"]
    patients = sorted({p for p, _ in hists})
    volume = tables["labels"].set_index(["patient", "label"]).volume_ml
    order = {lab: volume.xs(lab, level="label").reindex(patients).sort_values().index for lab in LABELS}
    order["background"] = tables["patients"].set_index("patient").voxels.reindex(patients).sort_values().index

    apply_ticks_style()
    fig = plt.figure(figsize=(22, 12))
    col_w, row_h, pitch, left = 0.0465, 0.27, 0.185, 0.07  # axes overlap: 3D axes leave empty margin
    masks = {}
    for patient in patients:
        img = nib.load(str(args.data_dir / patient / "GT.nii.gz"))
        seg, zooms = np.asarray(img.dataobj), np.array(img.header.get_zooms()[:3], dtype=float)
        for lab in LABELS:
            masks[(patient, lab)] = resample_mask(seg == lab, zooms, MESH_MM)
    for r, lab in enumerate(LABELS):
        top = 1.0 - r * pitch
        for c, (patient, shade) in enumerate(zip(order[lab], shades(COLORS[lab], len(patients)))):
            ax = fig.add_axes([left + c * col_w, top - row_h, col_w, row_h], projection="3d")
            draw_shape(ax, masks[(patient, lab)], shade)
            fig.text(
                left + (c + 0.5) * col_w,
                top - row_h / 2 - 0.065,
                patient[-2:],
                ha="center",
                fontsize=10,
                color="#555555",
            )
        fig.text(0.01, top - row_h / 2, f"label {lab}", fontsize=17, fontweight="bold", va="center")

    ax = fig.add_axes([left, 0.065, 0.9, 0.3])
    for key in ["background", 3, 2, 1]:
        name = key if key == "background" else f"label {key}"
        per_patient = [binned_counts(*hists[(p, name)], *HU_RANGE, BIN_HU) for p in order[key]]
        x = per_patient[0][0]
        pct = 100 / sum(c for _, c in per_patient).sum()
        cum = np.zeros_like(x)
        for (_, counts), shade in zip(per_patient, shades(COLORS[key], len(patients))):
            ax.fill_between(x, cum, cum + counts * pct, step="mid", color=shade, alpha=0.75, lw=0)
            cum = cum + counts * pct
        ax.step(x, cum, where="mid", color=COLORS[key], lw=1.6, label=name)
    ax.set_xlim(*HU_RANGE)
    ax.set_xlabel("HU", fontsize=13)
    ax.set_ylabel("% of that group's voxels", fontsize=13)
    ax.legend(frameon=False, loc="upper center", ncol=4, fontsize=13)
    sns.despine(ax=ax)
    fig.text(
        left,
        0.39,
        "HU inside each label and the background, all patients stacked (same shading as the shapes above)",
        fontsize=15,
        fontweight="bold",
    )

    header(
        fig,
        "Label Size and Intensity",
        "Top: every patient's label at the same physical scale, sorted left to right from smallest to largest volume "
        f"(light to dark). Bottom: {BIN_HU} HU histograms as % of each group's voxels, one band per patient.",
        subtitle_y=0.95,
    )
    out = out_subdir(args.profile_dir, "06-07_label_size_intensity") / "label_size_intensity.png"
    fig.savefig(out, dpi=115)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
