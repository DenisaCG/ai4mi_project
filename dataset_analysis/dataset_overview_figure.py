"""The state of the dataset and the overall performance of the upgraded baseline, sized for a slide.

    python dataset_analysis/dataset_overview_figure.py [--out figures]

Left: the 40 CT scans of the full training set (tables of the Snellius analysis in
deck/snellius_inventory/deficiencies_scur0049): slices per scan split by slice thickness, and a ring of the pooled voxel
share of the four organs among the foreground voxels, the background share in its centre. Right: the upgraded baseline
(ENet, Dice + CE, full pre-processing; 4 folds x 3 seeds, 40 validation patients) per organ: the kernel density of the
120 patient-seed values of 3D Dice and of HD95, the tick under a curve being the median, and the number of values with
HD95 above 50 mm. 3D metrics on the original CT grid, best checkpoint. Writes dataset_overview.{png,pdf} and
dataset_overview.csv (the plotted statistics).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy.stats import gaussian_kde

from utils import CLASSES, REPO, load_cv_runs, pyplot, read_csv, write_csv

sys.path.insert(0, str(REPO / "tools"))
from plot_style import BACKGROUND_COLOR, EARTH_LABEL_COLORS, GRID_COLOR, PALETTE  # noqa: E402

EXPERIMENT = "full_cv4_enet_dice_ce"
DATA_DIR = REPO / "deck/snellius_inventory/deficiencies_scur0049"
ORGANS = list(CLASSES.values())
ORGAN_COLORS = {name: EARTH_LABEL_COLORS[idx] for idx, name in CLASSES.items()}
FIG_W, FIG_H = 11.9, 3.3  # inches, the slide slot
RIDGE_H = 0.95  # curve height in row units (rows overlap slightly)
GRID_POINTS = 400
THICKNESS_COLORS = {2.5: PALETTE[0], 2.0: PALETTE[1]}
SLICE_BINS = np.arange(140, 301, 20)


def quantiles(values) -> dict[str, float]:
    """Median and 5th and 95th percentile of a sample (linear interpolation).

    Args:
        values: Sample of one metric.

    Returns:
        Dict with keys median, p5 and p95.
    """
    p5, median, p95 = np.percentile(values, [5, 50, 95])
    return {"median": float(median), "p5": float(p5), "p95": float(p95)}


def density(values, grid: np.ndarray) -> np.ndarray:
    """Gaussian kernel density of a sample on a grid, scaled to a peak of 1.

    Args:
        values: Sample (Dice values, or log10 of HD95).
        grid: Points where the density is evaluated.

    Returns:
        The density at the grid points divided by its maximum.
    """
    d = gaussian_kde(values)(grid)
    return d / d.max()


def summarise(rows: list[dict]) -> list[dict]:
    """Statistics behind the ridges, one row per organ.

    Args:
        rows: Validation rows (Dice, HD95) from load_cv_runs.

    Returns:
        Rows with n, the median, p5 and p95 of Dice and HD95.
    """
    out = []
    for organ in ORGANS:
        mine = [r for r in rows if r["organ"] == organ]
        row = {"organ": organ, "n": len(mine)}
        for metric in ("dice", "hd95"):
            row |= {
                f"{metric}_{k}": v
                for k, v in quantiles([r[metric] for r in mine]).items()
            }
        out.append(row)
    return out


def draw_ridges(ax, rows, stats, metric) -> None:
    """One filled density per organ with its median tick."""
    log = metric == "hd95"
    grid = (
        np.linspace(np.log10(1.7), np.log10(450), GRID_POINTS)
        if log
        else np.linspace(0, 1, GRID_POINTS)
    )
    for i, organ in enumerate(ORGANS):
        color = ORGAN_COLORS[organ]
        base = i + 0.45
        values = np.array([r[metric] for r in rows if r["organ"] == organ])
        y = base - RIDGE_H * density(np.log10(values) if log else values, grid)
        xs = 10**grid if log else grid
        ax.fill_between(xs, base, y, color=color, alpha=0.55, lw=0, zorder=2 + i * 0.1)
        ax.plot(xs, y, color=color, lw=1.4, zorder=3 + i * 0.1)
        med = stats[organ][f"{metric}_median"]
        ax.plot([med, med], [base, base + 0.2], color=color, lw=2.4, solid_capstyle="butt", zorder=5)  # fmt: skip
        ax.axhline(base, color=GRID_COLOR, lw=0.8, zorder=1)
    ax.set_ylim(len(ORGANS) - 0.5 + 0.45, 0.45 - RIDGE_H - 0.05)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=GRID_COLOR, lw=0.8, ls=(0, (3, 3)))
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=11, length=2, pad=1)


def geometry() -> list[dict]:
    """Slice thickness (mm) and slice count of the 40 scans."""
    return [
        {
            "thickness": round(float(r["slice_thickness_superior_inferior_mm"]), 1),
            "slices": int(r["n_slices"]),
        }
        for r in read_csv(DATA_DIR / "scan_geometry.csv")
    ]


def label_shares() -> dict[str, float]:
    """Pooled voxel share (%) of the background and of each organ over the 40 scans."""
    return {
        r["name"]: float(r["share_of_all_voxels_pct"])
        for r in read_csv(DATA_DIR / "label_shares_pooled.csv")
    }


def draw_slices(fig, scans) -> None:
    """Stacked histogram of slices per scan, one colour per slice thickness."""
    ax = fig.add_axes([0.5 / FIG_W, 0.68 / FIG_H, 2.2 / FIG_W, 2.2 / FIG_H])
    thicknesses = sorted(THICKNESS_COLORS, reverse=True)
    ax.hist(
        [[s["slices"] for s in scans if s["thickness"] == t] for t in thicknesses],
        bins=SLICE_BINS, stacked=True, histtype="stepfilled", lw=1.0,
        color=[THICKNESS_COLORS[t] for t in thicknesses],
        edgecolor="white", label=[f"{t} mm" for t in thicknesses],
    )  # fmt: skip
    ax.set_xticks([150, 200, 250, 300])
    ax.set_ylim(0, 13)
    ax.set_yticks([0, 5, 10])
    ax.grid(axis="y", color=GRID_COLOR, lw=0.8, ls=(0, (3, 3)))
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11, length=2, pad=1)
    ax.set_xlabel("slices per scan", fontsize=11, labelpad=1)
    ax.set_ylabel("scans", fontsize=11, labelpad=1)
    ax.legend(
        title="slice thickness", title_fontsize=11, fontsize=11, loc="upper right",
        frameon=False, handlelength=1, borderaxespad=0, labelspacing=0.2,
    )  # fmt: skip
    fig.text(0, 1 - 0.02 / FIG_H, "Slices per scan", fontsize=12, fontweight="bold", va="top")  # fmt: skip


def label_voxels() -> dict[str, int]:
    """Pooled voxel count of the background and of each organ over the 40 scans."""
    return {
        r["name"]: int(r["voxels"])
        for r in read_csv(DATA_DIR / "label_shares_pooled.csv")
    }


def draw_shares(fig, shares, voxels) -> None:
    """Label make-up, stacked: ring of the background share above a panel with the organ shares of the labelled voxels."""
    labelled = 100 - shares["background"]
    cx = 3.1 + 3.4 / 2  # centre of the block, inches
    bg = fig.add_axes([(cx - 0.75) / FIG_W, 1.55 / FIG_H, 1.5 / FIG_W, 1.5 / FIG_H])
    bg.pie([labelled, shares["background"]], colors=["#8C8C8C", BACKGROUND_COLOR], startangle=-90 + 1.8 * labelled,
           counterclock=False, wedgeprops={"width": 0.14, "edgecolor": "white", "lw": 0.5})  # fmt: skip
    bg.text(
        0,
        0.16,
        f"{shares['background']:.2f} %",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    bg.text(0, -0.12, "background", ha="center", va="center", fontsize=11)
    fig.text((cx + 0.8) / FIG_W, 2.3 / FIG_H, f"{labelled:.2f} %\nlabelled", ha="left", va="center", fontsize=11, color="#333333", linespacing=1.15)  # fmt: skip

    px, py, pw, ph = 3.1, 0.05, 3.4, 1.25
    fig.add_artist(FancyBboxPatch(
        (px / FIG_W, py / FIG_H), pw / FIG_W, ph / FIG_H, boxstyle="round,pad=0,rounding_size=0.01",
        transform=fig.transFigure, facecolor="#F3F3F3", edgecolor="#DDDDDD", lw=1, zorder=0,
    ))  # fmt: skip
    for x in (px, px + pw):  # lines from the labelled wedge to the panel corners
        fig.add_artist(Line2D([cx / FIG_W, x / FIG_W], [1.8 / FIG_H, (py + ph) / FIG_H], color="#CCCCCC", lw=1, transform=fig.transFigure))  # fmt: skip

    order = ["esophagus", "aorta", "heart", "trachea"]
    ring = fig.add_axes(
        [(px + 0.1) / FIG_W, (py + 0.0) / FIG_H, 1.25 / FIG_W, 1.25 / FIG_H]
    )
    values = [voxels[o] for o in order]
    total = sum(values)
    ring.pie(values, colors=[ORGAN_COLORS[o] for o in order], startangle=90, counterclock=False,
             wedgeprops={"width": 0.22, "edgecolor": "white", "lw": 1})  # fmt: skip
    ring.text(
        0,
        0.14,
        f"{total / 1e6:.1f} M",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    ring.text(0, -0.16, "labelled", ha="center", va="center", fontsize=11)
    for k, organ in enumerate(order):
        y = (py + 1.0 - k * 0.28) / FIG_H
        fig.add_artist(Rectangle(((px + 1.5) / FIG_W, y - 0.07 / FIG_H), 0.14 / FIG_W, 0.14 / FIG_H, transform=fig.transFigure, color=ORGAN_COLORS[organ]))  # fmt: skip
        fig.text((px + 1.7) / FIG_W, y, organ, fontsize=11, color=ORGAN_COLORS[organ], fontweight="bold", va="center")  # fmt: skip
        fig.text((px + 3.3) / FIG_W, y, f"{100 * voxels[organ] / total:.1f} %", fontsize=11, color=ORGAN_COLORS[organ], fontweight="bold", va="center", ha="right")  # fmt: skip
    fig.text(px / FIG_W, 1 - 0.02 / FIG_H, "Label make-up of the 40 scans", fontsize=12, fontweight="bold", va="top")  # fmt: skip


def draw_performance(fig, rows, stats) -> None:
    """Dice and HD95 density panels on the right of the figure."""
    top, height = 0.68, 2.2
    left, dice_w, hd_x, hd_w = 7.7, 1.95, 9.9, 1.9
    for metric, (x, w) in {"dice": (left, dice_w), "hd95": (hd_x, hd_w)}.items():
        ax = fig.add_axes(
            [x / FIG_W, 1 - (top + height) / FIG_H, w / FIG_W, height / FIG_H]
        )
        draw_ridges(ax, rows, stats, metric)
        if metric == "dice":
            ax.set_xlim(0, 1.02)
            ax.set_xticks([0, 0.5, 1.0], ["0", ".5", "1"])
            ax.set_yticks(
                [i + 0.45 for i in range(len(ORGANS))], [o.capitalize() for o in ORGANS]
            )
            ax.set_xlabel("3D Dice", fontsize=11, labelpad=1)
        else:
            ticks = [2, 10, 100]
            ax.set_xscale("log")
            ax.set_xlim(1.7, 450)
            ax.set_xticks(ticks, [str(t) for t in ticks])
            ax.minorticks_off()
            ax.set_yticks([])
            ax.set_xlabel("HD95 (mm)", fontsize=11, labelpad=1)
        ax.tick_params(axis="y", length=0, labelsize=12, pad=3)
        fig.text(
            x / FIG_W, 1 - (top - 0.06) / FIG_H,
            "Dice \u2191" if metric == "dice" else "HD95 \u2193",
            fontsize=12, fontweight="bold", va="bottom",
        )  # fmt: skip
    fig.text(6.9 / FIG_W, 1 - 0.02 / FIG_H, "Upgraded baseline, 120 values", fontsize=12, fontweight="bold", va="top")  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metrics-dir", type=Path, default=REPO / "metrics")
    parser.add_argument("--out", type=Path, default=REPO / "figures")
    args = parser.parse_args(argv)

    plt = pyplot()
    rows = load_cv_runs(args.metrics_dir, EXPERIMENT, keys=("dice", "hd95"))
    table = summarise(rows)
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    draw_slices(fig, geometry())
    draw_shares(fig, label_shares(), label_voxels())
    draw_performance(fig, rows, {r["organ"]: r for r in table})
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"dataset_overview.{ext}")
    write_csv(
        args.out / "dataset_overview.csv",
        [
            {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}
            for r in table
        ],
    )


if __name__ == "__main__":
    main()
