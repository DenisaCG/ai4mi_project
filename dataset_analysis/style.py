"""Shared presentation style: reference-inspired organ colors and seven curated figures."""
import os
from pathlib import Path

COLORS = {1: "#B637CA", 2: "#087F9D", 3: "#16B79B"}
BACKGROUND = "#929292"
INK = "#171717"
NAMES = {1: "Esophagus", 2: "Heart", 3: "Trachea"}
FIGURES = ("class_distribution.png", "shape_descriptors_3d.png",
           "shape_descriptors_summary.png", "target_area_through_scan.png",
           "baseline_3d_dice_by_class.png", "baseline_dice_vs_target_size.png",
           "baseline_dice_by_organ_position.png")
OLD_FIGURES = ("original_class_frequency.png", "original_patient_volume_extent.png",
               "original_normalized_extent.png", "processed_positive_area_distribution.png",
               "processed_area_vs_z.png", "baseline_positive_dice_distribution.png",
               "baseline_dice_vs_area_scatter.png", "baseline_dice_vs_area_binned.png",
               "baseline_dice_vs_scan_z.png", "baseline_dice_vs_organ_z.png",
               "baseline_patient_performance.png")


def pyplot(output: Path):
    """One backend, font, palette, resolution and background for all figures."""
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
        "font.weight": "normal", "text.color": INK, "axes.labelcolor": INK,
        "axes.labelsize": 14, "axes.labelweight": "bold", "axes.titlesize": 17,
        "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#999999", "axes.linewidth": .7,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "xtick.color": INK, "ytick.color": INK, "legend.frameon": False,
        "legend.fontsize": 12, "lines.linewidth": 2.5,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "figure.figsize": (14, 9),
        "figure.dpi": 120, "savefig.dpi": 220})
    return plt


def frame(plt, title: str, subtitle: str, footer: str):
    """Reserve fixed header/footer bands so labels never compete with captions."""
    fig = plt.figure(figsize=(14, 9))
    fig.text(.055, .965, title, fontsize=24, weight="bold", va="top")
    fig.text(.055, .912, subtitle, fontsize=13, va="top", linespacing=1.5, color="#494949")
    fig.text(.055, .045, footer, fontsize=11, va="bottom", linespacing=1.45, color="#494949")
    fig.subplots_adjust(left=.10, right=.95, bottom=.27, top=.80, wspace=.35)
    return fig


def clean_axis(ax, grid: bool = True):
    """Only light horizontal reference lines; no box or competing color coding."""
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis="y", color="#E7E7E7", linewidth=.7)
    ax.tick_params(length=0, pad=8)


def save(fig, output: Path, name: str, plt):
    if name not in FIGURES:
        raise ValueError(f"Uncurated figure name: {name}")
    fig.savefig(output / "plots" / name, dpi=220, bbox_inches=None)
    plt.close(fig)


def remove_superseded(output: Path):
    """Delete only named v1 generated graphics inside the selected result folder."""
    for name in OLD_FIGURES:
        (output / "plots" / name).unlink(missing_ok=True)
    for name in ("esophagus_examples.png", "heart_examples.png", "trachea_examples.png"):
        (output / "examples" / name).unlink(missing_ok=True)
    (output / "tables/baseline_examples.csv").unlink(missing_ok=True)
