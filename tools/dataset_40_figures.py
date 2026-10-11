"""
Per-patient figures for the 40-patient SegTHOR set, read from the tables written by
dataset_40_tables.py. Rows are patients; circles are the original 20, squares the added 20.

Usage:
    python tools/dataset_40_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from plot_style import (
    EARTH_LABEL_COLORS,
    FOOTNOTE_COLOR,
    GRID_COLOR,
    PALETTE,
    apply_style,
    decorate,
    legend_below,
)

REPO = Path(__file__).resolve().parents[1]
MARKERS = {"original_20": "o", "added_20": "s"}
GROUP_NAMES = {"original_20": "original 20 (midterm)", "added_20": "added 20"}
ORGAN_NAMES = {1: "esophagus", 2: "heart", 3: "trachea", 4: "aorta"}
INK = "#333333"


def prepare_axes(
    fig_size: tuple[float, float], n_panels: int, patients: list[str], boundary: float
):
    fig, axes = plt.subplots(1, n_panels, figsize=fig_size, sharey=True, squeeze=False)
    axes = axes[0]
    y = {p: i for i, p in enumerate(patients)}
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
        ax.axhline(boundary, color=GRID_COLOR, linewidth=1.2, linestyle="--")
        ax.tick_params(length=0, pad=4, labelsize=8)
    axes[0].set_ylim(len(patients) - 0.5, -0.5)  # Patient_01 at the top
    axes[0].set_yticks(
        range(len(patients)), [p.replace("Patient_", "P") for p in patients]
    )
    return fig, axes, y


def scatter_groups(
    ax, df: pd.DataFrame, y: dict, column: str, color: str, **kw
) -> None:
    for group, marker in MARKERS.items():
        sub = df[df.group == group]
        ax.scatter(
            sub[column],
            [y[p] for p in sub.patient],
            marker=marker,
            s=26,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=GROUP_NAMES[group],
            zorder=3,
            **kw,
        )


def boundary_between_groups(df: pd.DataFrame, patients: list[str]) -> float:
    """Row position of the dashed line between the original and the added patients."""
    flags = [
        df.loc[df.patient == p, "group"].iloc[0] == "original_20" for p in patients
    ]
    if all(flags) or not any(flags):
        return -1.0
    return max(i for i, f in enumerate(flags) if f) + 0.5


def save(fig, path: Path, title: str, subtitle: str, footnote: str) -> None:
    decorate(fig, title, subtitle=subtitle, footnote_text=footnote)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def geometry_figure(tables: Path, out: Path) -> None:
    g = pd.read_csv(tables / "patient_geometry.csv")
    patients = sorted(g.patient)
    fig, axes, y = prepare_axes(
        (13, 10), 5, patients, boundary_between_groups(g, patients)
    )
    panels = [
        (
            "Pixel size (mm)",
            ["pixel_size_left_right_mm", "pixel_size_anterior_posterior_mm"],
            "left-right | anterior-posterior",
        ),
        (
            "Slice spacing (mm)",
            ["slice_spacing_superior_inferior_mm"],
            "superior-inferior",
        ),
        (
            "Matrix size (pixels)",
            ["matrix_left_right", "matrix_anterior_posterior"],
            "left-right | anterior-posterior",
        ),
        ("Number of slices", ["n_slices"], "superior-inferior"),
        ("Scan length (mm)", ["scan_length_mm"], "superior-inferior"),
    ]
    for ax, (title, cols, sub) in zip(axes, panels, strict=True):
        for col, color in zip(cols, (PALETTE[0], PALETTE[1]), strict=False):
            scatter_groups(ax, g, y, col, color)
        ax.set_title(f"{title}\n{sub}", fontsize=9, color=INK)
        if len(cols) == 2:
            ax.text(
                0.98,
                0.01,
                "blue: left-right\npeach: anterior-posterior",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
                color=FOOTNOTE_COLOR,
            )
    legend_below(axes[0], ncol=2)
    save(
        fig,
        out / "geometry_per_patient.png",
        "Scan geometry per patient",
        "Original 20 (circles) and added 20 (squares); dashed line separates them",
        "Pixel size and matrix size are drawn for both in-plane axes; where the markers coincide only one is visible.",
    )


def hu_figure(tables: Path, out: Path) -> None:
    h = pd.read_csv(tables / "patient_intensity.csv")
    patients = sorted(h.patient.unique())
    regions = list(dict.fromkeys(h.region))
    colors = {"all voxels": PALETTE[0]} | {
        ORGAN_NAMES[k]: EARTH_LABEL_COLORS[k] for k in ORGAN_NAMES
    }
    fig, axes, y = prepare_axes(
        (15, 10), len(regions), patients, boundary_between_groups(h, patients)
    )
    for ax, region in zip(axes, regions, strict=True):
        sub = h[h.region == region]
        color = colors[region]
        for _, r in sub.iterrows():
            ax.plot(
                [r["p0.5"], r["p99.5"]],
                [y[r.patient]] * 2,
                color=color,
                linewidth=1.0,
                alpha=0.6,
                zorder=2,
            )
        scatter_groups(ax, sub, y, "median", color)
        ax.scatter(
            sub["mean"],
            [y[p] for p in sub.patient],
            marker="|",
            s=40,
            color=INK,
            linewidth=1.0,
            zorder=4,
            label="mean",
        )
        ax.set_title(region, fontsize=10, color=INK)
        ax.set_xlabel("HU", fontsize=8)
    legend_below(axes[0], ncol=3)
    save(
        fig,
        out / "hu_per_patient.png",
        "HU distribution per patient, whole scan and inside each organ",
        "Marker = median, line = p0.5 to p99.5, tick = mean; circles original 20, squares added 20",
        "Standard deviations are in patient_intensity.csv.",
    )


def volume_figure(tables: Path, out: Path) -> None:
    lab = pd.read_csv(tables / "patient_labels.csv")
    patients = sorted(lab.patient.unique())
    fig, axes, y = prepare_axes(
        (12, 10), len(ORGAN_NAMES), patients, boundary_between_groups(lab, patients)
    )
    for ax, (k, name) in zip(axes, ORGAN_NAMES.items(), strict=True):
        sub = lab[lab.label == k]
        scatter_groups(ax, sub, y, "volume_ml", EARTH_LABEL_COLORS[k])
        ax.set_title(name, fontsize=10, color=INK)
        ax.set_xlabel("volume (mL)", fontsize=8)
        ax.set_xlim(left=0)
    legend_below(axes[0], ncol=2)
    save(
        fig,
        out / "organ_volume_per_patient.png",
        "Organ volume per patient",
        "Circles original 20, squares added 20; dashed line separates them",
        "Volume = voxel count x voxel volume of the scan.",
    )


def components_figure(tables: Path, out: Path) -> None:
    lab = pd.read_csv(tables / "patient_labels.csv")
    lab = lab[lab.label == 1]
    patients = sorted(lab.patient)
    fig, axes, y = prepare_axes(
        (9, 10), 2, patients, boundary_between_groups(lab, patients)
    )
    color = EARTH_LABEL_COLORS[1]
    scatter_groups(axes[0], lab, y, "components_face", color)
    axes[0].set_title("3D connected components\n(6-connected)", fontsize=9, color=INK)
    axes[0].set_xlim(0, lab.components_face.max() + 1)
    axes[0].set_xlabel("components per patient", fontsize=8)
    scatter_groups(axes[1], lab, y, "share_slices_2plus_components_pct", color)
    axes[1].set_title(
        "Axial slices with 2+ components\n(4-connected, of slices with esophagus)",
        fontsize=9,
        color=INK,
    )
    axes[1].set_xlabel("% of slices", fontsize=8)
    axes[1].set_xlim(left=0)
    legend_below(axes[0], ncol=2)
    save(
        fig,
        out / "esophagus_components_per_patient.png",
        "Esophagus fragmentation per patient",
        "Circles original 20, squares added 20; dashed line separates them",
        "",
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tables-dir", type=Path, default=REPO / "figures/dataset_40")
    ap.add_argument("--out-dir", type=Path, default=REPO / "figures/dataset_40")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    for fn in (geometry_figure, hu_figure, volume_figure, components_figure):
        fn(args.tables_dir, args.out_dir)


if __name__ == "__main__":
    main()
