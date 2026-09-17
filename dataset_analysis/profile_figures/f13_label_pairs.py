#!/usr/bin/env python3
"""
Label pairs: where labels 1, 2 and 3 meet, drawn in 3D for every patient, with
a strip under each row of shapes giving each pair's shared border area.

Each patient's labels are drawn as faint grey outlines at the same physical
scale; the part of a label's surface that lies within one voxel of another
label is painted in a colour for that pair and direction: within a slice where the
surface faces sideways, between slices where it faces up or down. The strips split the exact shared face area the same way:
faces between left-right or front-back neighbours are within a slice, faces
between voxels on neighbouring slices are between slices. Patients are sorted
by total shared border area.

Reads label_pairs.csv written by tools/dataset_profile.py and the label masks
from the data directory.

Usage:
    python dataset_analysis/profile_figures/f13_label_pairs.py --profile-dir figures/profile \
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
from profile_figures.common import LABEL_COLORS, apply_ticks_style, build_arg_parser, header, out_subdir  # noqa: E402

LABELS = list(LABEL_COLORS)
PAIRS = [(1, 2), (1, 3), (2, 3)]
MESH_MM = 2.0
HALF_WIDTH_MM, HALF_HEIGHT_MM = 90, 170  # shared 3D box for every patient
DIRECTIONS = ("within a slice", "between slices")
# Colour-blind-safe (Okabe-Ito) colours. Contact types that share a surface get opposite hues:
# labels 1 & 2 (most shared border) blue between slices vs orange within a slice,
# labels 1 & 3 green within a slice vs bright red between slices.
CONTACT_COLORS = {
    ((1, 2), "between slices"): "#0072B2",
    ((1, 2), "within a slice"): "#E69F00",
    ((1, 3), "within a slice"): "#009E73",
    ((1, 3), "between slices"): "#E8000B",
    ((2, 3), "within a slice"): "#7B2CBF",
    ((2, 3), "between slices"): "#000000",
}
FACING_UP = 0.7  # |normal z| above this counts a surface patch as facing the neighbouring slice
SURFACE_RGBA = (0.6, 0.6, 0.6, 0.06)  # faint grey so only the contact patches stand out
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


def shared_area_by_direction(a: np.ndarray, b: np.ndarray, zooms: np.ndarray) -> dict[str, float]:
    """Shared face area between two label masks, split into within-slice and between-slice faces.

    Args:
        a: Boolean mask of the first label, shape (x, y, z).
        b: Boolean mask of the second label, same shape.
        zooms: Voxel spacing (x, y, z) in mm.

    Returns:
        {"within a slice": area of faces between x or y neighbours (mm²),
         "between slices": area of faces between z neighbours (mm²)}.
    """
    face_area = [zooms[1] * zooms[2], zooms[0] * zooms[2], zooms[0] * zooms[1]]
    per_axis = []
    for axis in range(3):
        head, tail = [slice(None)] * 3, [slice(None)] * 3
        head[axis], tail[axis] = slice(None, -1), slice(1, None)
        head, tail = tuple(head), tuple(tail)
        per_axis.append(face_area[axis] * int((a[head] & b[tail]).sum() + (b[head] & a[tail]).sum()))
    return {DIRECTIONS[0]: per_axis[0] + per_axis[1], DIRECTIONS[1]: per_axis[2]}


def _shade(pair: tuple[int, int], direction: str) -> np.ndarray:
    return np.array(to_rgb(CONTACT_COLORS[(pair, direction)]))


def _draw_patient(ax, grids: dict[int, np.ndarray]) -> None:
    near_other = {lab: distance_transform_edt(~g, sampling=MESH_MM) for lab, g in grids.items()}
    centre = np.array(next(iter(grids.values())).shape) * MESH_MM / 2
    for lab, grid in grids.items():
        if not grid.any():
            continue
        verts, faces, _, _ = marching_cubes(grid.astype(np.float32), 0.5, spacing=(MESH_MM,) * 3)
        tri = verts[faces]
        rgba = np.tile(SURFACE_RGBA, (len(faces), 1))
        cell = np.clip((tri.mean(axis=1) / MESH_MM).astype(int), 0, np.array(grid.shape) - 1)
        tri = tri - centre
        tri[..., 0] *= -1  # array x runs towards patient left; flip so patient left is on the viewer's right
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
        facing_up = np.abs(normals[:, 2]) > FACING_UP
        for pair in PAIRS:
            if lab in pair:
                other = pair[1] if lab == pair[0] else pair[0]
                touching = near_other[other][cell[:, 0], cell[:, 1], cell[:, 2]] <= 1.5 * MESH_MM
                for direction, mask in zip(DIRECTIONS, (~facing_up, facing_up)):
                    rgba[touching & mask] = np.r_[_shade(pair, direction), 1.0]
        rgba[:, :3] *= (0.5 + 0.5 * np.abs(normals @ LIGHT))[:, None]
        ax.add_collection3d(Poly3DCollection(tri, facecolors=rgba, linewidths=0))
    ax.set_xlim(-HALF_WIDTH_MM, HALF_WIDTH_MM)
    ax.set_ylim(-HALF_WIDTH_MM, HALF_WIDTH_MM)
    ax.set_zlim(-HALF_HEIGHT_MM, HALF_HEIGHT_MM)
    ax.set_box_aspect((1, 1, HALF_HEIGHT_MM / HALF_WIDTH_MM), zoom=1.35)
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

    rows = [(pair, direction) for pair in PAIRS for direction in DIRECTIONS]
    area = {}

    apply_ticks_style()
    fig = plt.figure(figsize=(20, 12.5))
    col_w, row_h, left = 0.09, 0.4, 0.07
    strip_h = 0.11
    strips = []
    for r in range(2):
        top = 1.0 - r * 0.44
        row_patients = order[r * 10 : (r + 1) * 10]
        for c, patient in enumerate(row_patients):
            img = nib.load(str(args.data_dir / patient / "GT.nii.gz"))
            seg, zooms = np.asarray(img.dataobj), np.array(img.header.get_zooms()[:3], dtype=float)
            ax = fig.add_axes([left + c * col_w, top - row_h, col_w, row_h], projection="3d")
            _draw_patient(ax, label_grids(seg, zooms, LABELS, MESH_MM))
            for a, b in PAIRS:
                split = shared_area_by_direction(seg == a, seg == b, zooms)
                for direction in DIRECTIONS:
                    area[(patient, (a, b), direction)] = split[direction]
        strips.append((top, row_patients))
    area_max = max(area.values())
    for top, row_patients in strips:
        strip = fig.add_axes([left, top - row_h - strip_h + 0.095, col_w * len(row_patients), strip_h])
        cells = np.ones((len(rows), len(row_patients), 3))
        for i, (pair, direction) in enumerate(rows):
            share = np.array([area[(p, pair, direction)] for p in row_patients]) / area_max
            cells[i] = 1 - share[:, None] * (1 - _shade(pair, direction))
        strip.imshow(cells, aspect="auto", extent=(0, len(row_patients), len(rows), 0))
        strip.set_xticks(np.arange(len(row_patients)) + 0.5, [f"patient {p[-2:]}" for p in row_patients])
        strip.set_yticks(np.arange(len(rows)) + 0.5, [f"{a} & {b}, {d}" for (a, b), d in rows], fontsize=10)
        strip.tick_params(length=0)
        strip.hlines(range(1, len(rows)), 0, len(row_patients), color="white", lw=1.5)
        strip.vlines(range(1, len(row_patients)), 0, len(rows), color="white", lw=2)
        for spine in strip.spines.values():
            spine.set_visible(False)

    key = fig.add_axes([0.6, 0.04, 0.3, 0.016])
    key.imshow(np.linspace(1, 0.25, 256)[None, :, None].repeat(3, axis=2), aspect="auto", extent=(0, area_max, 0, 1))
    key.set_yticks([])
    key.set_title("shared border area (mm²), white = none", fontsize=12)
    for spine in key.spines.values():
        spine.set_visible(False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_shade(p, d), label=f"labels {p[0]} & {p[1]}, {d}")
        for p in PAIRS
        for d in DIRECTIONS
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, 0.005), ncol=3, frameon=False, fontsize=12)
    header(
        fig,
        "Contact Between Label Pairs",
        f"One 3D render per patient (n={pairs.patient.nunique()}), sorted by total shared border area; "
        "colour = surface within 1 voxel of the other label (colour per pair and direction); "
        "strip = shared border area per pair and direction (mm²).",
        0.95,
    )
    out = out_subdir(args.profile_dir, "13_label_pairs") / "label_pairs_contact_3d.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
