#!/usr/bin/env python3
"""
Scan geometry before vs after median-spacing resampling.

Writes dataset_analysis/results/median_spacing/scan_geometry_before_after.png: one line per patient from its native value
(left, hollow dots) to its resampled value (right, filled dots).
  Top row    voxel size (in-plane pixel size, slice spacing): every scan ends at the same target.
  Bottom row matrix size (in-plane voxels, number of slices): now varies, because the physical extent stays the same.

Resampling changes the sampling grid, not the scan: field of view, scan length and organ size in mm are unchanged.
The "after" values are therefore computed from the native geometry and the target spacing, no volume is resampled:
    voxels_after = round(voxels_native * spacing_native / target)
The target is the one the resampling step uses (slice_segthor.median_target_spacing on the training patients of
the split, as in the config) and is checked against the one saved next to the built dataset.

Same look as the profile_figures/scan_geometry.py figure (crest colours of its panels, red median triangle), but only
needs matplotlib, numpy and nibabel headers: no patients.csv, pandas or seaborn.

Usage (from the repo root):
    python dataset_analysis/scan_geometry_before_after.py \
        --config configs/segthor_enet_ce_corrected_ct_window_median_spacing.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from slice_segthor import get_splits, median_target_spacing  # noqa: E402
from src.config import load_config  # noqa: E402

MEDIAN_COLOR = "#D1495B"
# crest colours of the matching panels of scan_geometry.png (pixel size, slice spacing, image width, number of slices)
COLORS = {"pixel": "#7dba91", "slice": "#59a590", "matrix": "#1c6488", "slices": "#287a8c"}
INK, MUTED = "#171717", "#555555"


def native_geometry(data_dir: Path) -> dict[str, dict]:
    """Voxel size and matrix size of every patient, from the NIfTI headers only."""
    out = {}
    for path in sorted(data_dir.glob("Patient_*")):
        img = nib.load(str(path / f"{path.name}.nii.gz"))
        dx, dy, dz = (float(z) for z in img.header.get_zooms()[:3])
        assert np.isclose(dx, dy), (path.name, dx, dy)
        out[path.name] = {"pixel_mm": dx, "slice_mm": dz, "x_vox": img.shape[0], "y_vox": img.shape[1], "z_vox": img.shape[2]}
    return out


def after_resampling(native: dict, target: tuple[float, float, float]) -> dict:
    """Same grid arithmetic as ndimage.zoom in the resampling step: extent stays, voxel size becomes the target."""
    return {"pixel_mm": target[0], "slice_mm": target[2],
            "x_vox": int(np.rint(native["x_vox"] * native["pixel_mm"] / target[0])),
            "y_vox": int(np.rint(native["y_vox"] * native["pixel_mm"] / target[1])),
            "z_vox": int(np.rint(native["z_vox"] * native["slice_mm"] / target[2]))}


def clusters(values: np.ndarray, decimals: int) -> list[tuple[float, int]]:
    """(value, number of patients) for every distinct value after rounding to `decimals`."""
    key = np.round(values, decimals)
    return [(float(values[key == k].mean()), int((key == k).sum())) for k in np.unique(key)]


def slopegraph(ax, before, after, color: str, title: str, ylabel: str, decimals: int, target: float | None = None,
               note: str | None = None, note_right: bool = False, label_extremes_only: bool = False) -> None:
    before, after = np.asarray(before, float), np.asarray(after, float)
    changed = np.abs(after - before) > 0.5 * 10.0**-decimals
    for b, a, c in zip(before, after, changed):
        ax.plot([0, 1], [b, a], color=color, lw=2.4 if c else 1.4, alpha=0.9 if c else 0.45, zorder=3 if c else 2)
    ax.axvline(0, color="#999999", ls="--", lw=1.1, zorder=1)  # native: dashed
    ax.axvline(1, color=MUTED, ls="-", lw=1.4, zorder=1)  # resampled: solid
    fmt = f"{{:.{decimals}f}}"
    for x, values, face, side in ((0, before, "white", -1), (1, after, color, 1)):
        found = clusters(values, decimals)
        for v, n in found:
            ax.scatter([x], [v], s=75 * np.sqrt(n), facecolor=face, edgecolor=color, lw=2.2, zorder=5)
            if label_extremes_only and v not in (found[0][0], found[-1][0]):  # too many values to label each one
                continue
            ax.annotate(fmt.format(v) + (f"  ×{n}" if n > 1 else ""), (x, v), xytext=(side * 15, 0),
                        textcoords="offset points", ha="right" if side < 0 else "left", va="center", fontsize=11.5, color=INK)
    if target is not None:  # the target is the median of the training patients
        ax.plot(1, target, marker="^", markersize=13, color=MEDIAN_COLOR, ls="", zorder=7, clip_on=False)
    lo, hi = min(before.min(), after.min()), max(before.max(), after.max())
    pad = 0.12 * (hi - lo) if hi > lo else 0.1 * hi
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(-0.7, 1.7)
    ax.set_xticks([0, 1], ["Native", "Resampled"], fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=15, fontweight="bold", loc="left")
    if note:
        ax.text(0.98 if note_right else 0.02, 0.98, note, transform=ax.transAxes, ha="right" if note_right else "left", va="top", fontsize=11.5, color=INK, style="italic",
                bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#CCCCCC", "alpha": 0.95}, zorder=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw(native: dict, after: dict, target: tuple, n_train: int, out: Path) -> None:
    names = sorted(native)
    col = lambda d, k: [d[n][k] for n in names]  # noqa: E731
    plt.rcParams.update({"font.size": 12.5, "axes.linewidth": 1.25, "axes.labelsize": 13, "figure.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(16, 9.5))
    slopegraph(axes[0, 0], col(native, "pixel_mm"), col(after, "pixel_mm"), COLORS["pixel"],
               "Pixel size, x = y (mm)", "mm", 3, target=target[0])
    slopegraph(axes[0, 1], col(native, "slice_mm"), col(after, "slice_mm"), COLORS["slice"],
               "Slice spacing, z (mm)", "mm", 2, target=target[2])
    slopegraph(axes[1, 0], col(native, "x_vox"), col(after, "x_vox"), COLORS["matrix"],
               "Image size in plane, x = y (voxels)", "voxels", 0,
               note="uniform grid → varies:\ncrop/pad to a fixed input size")
    slopegraph(axes[1, 1], col(native, "z_vox"), col(after, "z_vox"), COLORS["slices"],
               "Number of slices", "slices", 0,
               note="same scan length in mm,\nnew slice count", note_right=True, label_extremes_only=True)
    for ax, text in ((axes[0, 0], "1 · the fix: voxel size"), (axes[1, 0], "2 · the consequence: matrix size")):
        ax.text(-0.17, 0.5, text, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=13,
                fontweight="bold", color=MUTED)
    fig.legend(handles=[plt.Line2D([], [], marker="^", ls="", color=MEDIAN_COLOR, markersize=12,
                                   label=f"median of the {n_train} training patients = target spacing")],
               loc="lower center", frameon=False, bbox_to_anchor=(0.5, 0.045), fontsize=12.5)
    fig.suptitle("Voxel Geometry Before vs After Resampling", fontsize=22, fontweight="bold", x=0.02, ha="left", y=0.995)
    fig.text(0.02, 0.935, "Resampling gives every scan the same voxel size; the matrix size then varies with real anatomy, "
             "so we crop/pad to a fixed input.", fontsize=13.5, color="#444444", ha="left")
    fig.text(0.02, 0.012, f"One line per patient ({len(names)} scans); hollow = native, filled = resampled to "
             f"{target[0]:.3f} × {target[1]:.3f} × {target[2]:.2f} mm; bold lines = the value changed, dot size = number of "
             "patients at that value (slice counts: min and max labelled).\nField of view and scan length in mm do not change; "
             "the resampled values are computed from the native geometry, not measured on resampled volumes.",
             fontsize=10.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0.03, 0.075, 1, 0.9))
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def report(native: dict, after: dict, target: tuple, cfg: dict) -> None:
    """Print the sanity checks; assert everything the figure claims."""
    names = sorted(native)
    p = cfg["data"]["preprocess"]
    print(f"target spacing (analytic, train patients only): {np.round(target, 4).tolist()} mm")
    stats = REPO / cfg["data"]["root"] / "ct_norm_stats.json"
    if stats.exists():
        saved = json.loads(stats.read_text())["target_spacing"]
        assert np.allclose(target, saved, rtol=0, atol=1e-9), (target, saved)
        print(f"  matches the target the resampling step used ({stats.relative_to(REPO)}): {np.round(saved, 4).tolist()}")
    else:
        print(f"  WARNING: {stats} not found, could not compare with the resampling step's target")

    for key in ("pixel_mm", "slice_mm"):
        vals = {after[n][key] for n in names}
        assert len(vals) == 1, (key, vals)
        print(f"after {key}: constant = {vals.pop():.4f} across all {len(names)} patients")
    fov = np.array([native[n]["x_vox"] * native[n]["pixel_mm"] for n in names])
    length = np.array([native[n]["z_vox"] * native[n]["slice_mm"] for n in names])
    for key, label in (("x_vox", "in-plane voxels"), ("z_vox", "slices")):
        b, a = np.array([native[n][key] for n in names]), np.array([after[n][key] for n in names])
        print(f"{label}: native min/max {b.min()}/{b.max()} -> resampled min/max {a.min()}/{a.max()}")
    print(f"physical field of view {fov.min():.0f}-{fov.max():.0f} mm and scan length {length.min():.0f}-{length.max():.0f} mm "
          "are unchanged by construction")
    assert len({after[n]["x_vox"] for n in names}) > 1, "in-plane voxel count should vary after resampling"

    resampled = REPO / cfg["data"]["root"] / "resampled" / "train"  # header-only check against the real resampled volumes
    if resampled.exists():
        diff = np.array([[after[n][k] - nib.load(str(resampled / n / f"{n}.nii.gz")).shape[i]
                          for i, k in enumerate(("x_vox", "y_vox", "z_vox"))] for n in names])
        print(f"analytic vs actual resampled shape (header): max |difference| = {np.abs(diff).max()} voxels "
              f"over {len(names)} patients")
        assert np.abs(diff).max() <= 1, diff


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO / "configs/segthor_enet_ce_corrected_ct_window_median_spacing.yaml",
                    help="a config with data.preprocess.resample: median (gives the source, split and dataset)")
    ap.add_argument("--out", type=Path, default=REPO / "dataset_analysis" / "results" / "median_spacing")
    args = ap.parse_args()

    cfg = load_config(args.config)
    p = cfg["data"]["preprocess"]
    assert p.get("resample") == "median", "config has no data.preprocess.resample: median"
    src = REPO / p["source_dir"]
    native = native_geometry(src / "train")
    assert len(native) == 20, len(native)

    random.seed(p["seed"])  # same shuffle as slice_segthor.py, so the same training patients
    train_ids, _, _ = get_splits(src, p["retains"], p["fold"])
    target = median_target_spacing([(native[i]["pixel_mm"], native[i]["pixel_mm"], native[i]["slice_mm"]) for i in train_ids])
    after = {n: after_resampling(g, target) for n, g in native.items()}
    report(native, after, target, cfg)

    args.out.mkdir(parents=True, exist_ok=True)
    draw(native, after, target, len(train_ids), args.out / "scan_geometry_before_after.png")


if __name__ == "__main__":
    main()
