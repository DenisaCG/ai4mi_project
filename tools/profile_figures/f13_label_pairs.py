#!/usr/bin/env python3
"""
Label pairs: where labels 1, 2 and 3 meet, drawn in 3D for every patient.

Each patient's three labels are drawn see-through at the same physical scale; the
part of a label's surface that lies within one voxel of another label is
painted in that pair's colour. Patients are sorted by total shared border area.

Reads label_pairs.csv written by tools/dataset_profile.py and the label masks
from the data directory.

Usage:
    python tools/profile_figures/f13_label_pairs.py --profile-dir figures/profile \
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
import pandas as pd
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import distance_transform_edt, zoom
from skimage.measure import marching_cubes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import apply_ticks_style  # noqa: E402
from profile_figures.common import LABEL_COLORS, build_arg_parser, header, out_subdir  # noqa: E402

LABELS = list(LABEL_COLORS)
PAIRS = [(1, 2), (1, 3), (2, 3)]
PAIR_COLORS = {(1, 2): "#6A3D9A", (1, 3): "#1F5FA8", (2, 3): "#2E8B2E"}
MESH_MM = 2.0
HALF_WIDTH_MM, HALF_HEIGHT_MM = 90, 170  # shared 3D box for every patient
LABEL_TINT, LABEL_ALPHA = 0.25, 0.2  # labels are drawn see-through so the contact patches stay visible
LIGHT = np.array([-0.4, -0.6, 0.7]) / np.linalg.norm([-0.4, -0.6, 0.7])


def label_grids(seg: np.ndarray, zooms: np.ndarray, labels: list[int], step_mm: float) -> dict[int, np.ndarray]:
    """Resample several labels onto one shared isotropic grid covering all of them.

    Args:
        seg: Integer label volume.
        zooms: Voxel spacing (x, y, z) in mm.
        labels: Label values to resample; at least one must be present.
        step_mm: Output voxel size in mm.

    Returns:
        {label: boolean mask}, all with the same shape and a one-voxel empty border.
    """
    idx = np.argwhere(np.isin(seg, labels))
    lo, hi = idx.min(axis=0), idx.max(axis=0) + 1
    crop = seg[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    factors = np.asarray(zooms) / step_mm
    return {lab: np.pad(zoom((crop == lab).astype(np.float32), factors, order=1) > 0.5, 1) for lab in labels}


def _draw_patient(ax, grids: dict[int, np.ndarray]) -> None:
    near_other = {lab: distance_transform_edt(~g, sampling=MESH_MM) for lab, g in grids.items()}
    centre = np.array(next(iter(grids.values())).shape) * MESH_MM / 2
    for lab, grid in grids.items():
        if not grid.any():
            continue
        verts, faces, _, _ = marching_cubes(grid.astype(np.float32), 0.5, spacing=(MESH_MM,) * 3)
        tri = verts[faces]
        base = np.array(to_rgb(LABEL_COLORS[lab]))
        rgba = np.tile(np.r_[base + (1 - base) * LABEL_TINT, LABEL_ALPHA], (len(faces), 1))
        cell = np.clip((tri.mean(axis=1) / MESH_MM).astype(int), 0, np.array(grid.shape) - 1)
        for pair in PAIRS:
            if lab in pair:
                other = pair[1] if lab == pair[0] else pair[0]
                touching = near_other[other][cell[:, 0], cell[:, 1], cell[:, 2]] <= 1.5 * MESH_MM
                rgba[touching] = np.r_[to_rgb(PAIR_COLORS[pair]), 1.0]
        tri = tri - centre
        tri[..., 0] *= -1  # array x runs towards patient left; flip so patient left is on the viewer's right
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
        rgba[:, :3] *= (0.5 + 0.5 * np.abs(normals @ LIGHT))[:, None]
        ax.add_collection3d(Poly3DCollection(tri, facecolors=rgba, linewidths=0))
    ax.set_xlim(-HALF_WIDTH_MM, HALF_WIDTH_MM)
    ax.set_ylim(-HALF_WIDTH_MM, HALF_WIDTH_MM)
    ax.set_zlim(-HALF_HEIGHT_MM, HALF_HEIGHT_MM)
    ax.set_box_aspect((1, 1, HALF_HEIGHT_MM / HALF_WIDTH_MM), zoom=1.3)
    ax.set_axis_off()
    ax.view_init(elev=8, azim=-70)


def main():
    """Draw the label pairs contact figure."""
    ap = build_arg_parser(__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/segthor_part1/train"))
    args = ap.parse_args()

    pairs = pd.read_csv(args.profile_dir / "label_pairs.csv")
    pairs = pairs[pairs["label_a"].isin(LABELS) & pairs["label_b"].isin(LABELS)]
    order = pairs.groupby("patient")["shared_face_area_mm2"].sum().sort_values().index

    apply_ticks_style()
    fig = plt.figure(figsize=(20, 10.5))
    col_w, row_h, pitch = 0.1, 0.5, 0.42
    for k, patient in enumerate(order):
        img = nib.load(str(args.data_dir / patient / "GT.nii.gz"))
        grids = label_grids(np.asarray(img.dataobj), np.array(img.header.get_zooms()[:3], dtype=float), LABELS, MESH_MM)
        row, col = divmod(k, 10)
        top = 0.97 - row * pitch
        ax = fig.add_axes([col * col_w, top - row_h, col_w, row_h], projection="3d")
        _draw_patient(ax, grids)
        fig.text((col + 0.5) * col_w, top - row_h + 0.07, patient[-2:], ha="center", fontsize=11, color="#555555")

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=np.array(to_rgb(LABEL_COLORS[lab])) * (1 - LABEL_TINT) + LABEL_TINT,
            alpha=LABEL_ALPHA * 3,
            label=f"label {lab}",
        )
        for lab in LABELS
    ] + [plt.Rectangle((0, 0), 1, 1, color=PAIR_COLORS[p], label=f"where labels {p[0]} & {p[1]} meet") for p in PAIRS]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=13)
    header(
        fig,
        "Label Pairs: Where the Labels Meet",
        "Front view, each patient's three labels drawn see-through at the same scale; surface within one voxel of another "
        "label is coloured by the pair. Sorted left to right, top to bottom, by total shared border.",
        0.945,
    )
    out = out_subdir(args.profile_dir, "13_label_pairs") / "label_pairs_contact_3d.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
