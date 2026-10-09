"""Alternative figures of the augmentation results.

    python dataset_analysis/augmentation_options.py [--runs docs/augmentation_results/per_run.csv] [--out figures]

The changes are those of augmentation_results.py (every run against the run of its control with the same fold and seed).
ridge: distribution of the Dice change over the 12 paired runs of every 12-run arm. organs: Dice of every organ along the
augmentation steps. violins: distribution of the 12 per-run organ Dice values at every step. folds: Dice of every fold along the steps (mean of its 3 seeds).
Writes augmentation_option_<name>.png for each.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde

from augmentation_results import (
    COLORS,
    ENET,
    POINTS,
    RESENC,
    RUNS,
    load_runs,
    pair_changes,
)
from utils import CLASSES, INK, REPO, pyplot
from plot_style import EARTH_LABEL_COLORS, tint
from augmentation_results import GEOMETRIC, INTENSITY, JOINT, STRONG

LABEL_NAMES = {"Joint ": "Joint", "Strong ": "Strong"}


def rows(runs: dict) -> list[dict]:
    """Label, type, group and per-run changes of every point of augmentation_results.POINTS."""
    out = []
    for label, experiment, reference, kind, _ in POINTS:
        changes = np.array(pair_changes(runs[experiment], runs[reference]))
        group = 0 if len(changes) == 4 else (2 if reference == RESENC else 1)
        out.append(
            {
                "label": LABEL_NAMES.get(label, label),
                "kind": kind,
                "group": group,
                "changes": changes,
            }
        )
    return sorted(out, key=lambda r: r["group"])


def ridge(runs: dict, out: Path) -> None:
    """Distribution of the Dice change over the paired runs of every 12-run arm."""
    plt = pyplot()
    data = [r for r in rows(runs) if r["group"] > 0]
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    grid = np.linspace(-0.01, 0.075, 300)
    step = 1.0
    for i, r in enumerate(data):
        y0 = (len(data) - 1 - i) * step
        color = COLORS[r["kind"]]
        d = r["changes"][:, 0]
        density = gaussian_kde(d, bw_method=0.5)(grid)
        density = density / density.max() * 0.95
        ax.fill_between(
            grid, y0, y0 + density, color=tint(color, 0.45), zorder=2 + i * 0.01, lw=0
        )
        ax.plot(grid, y0 + density, color=color, lw=1.6, zorder=3 + i * 0.01)
        ax.scatter(
            d,
            np.full(len(d), y0 + 0.07),
            s=16,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
        )
        ax.plot([d.mean()] * 2, [y0, y0 + 0.5], color=INK, lw=2.2, zorder=6)
        ax.text(
            0.079,
            y0 + 0.25,
            f"{d.mean():+.3f}   {(d > 0).sum()}/12".replace("-", "\N{MINUS SIGN}"),
            fontsize=11,
            va="center",
            color=INK,
            clip_on=False,
        )
    ax.axvline(0, color=INK, lw=1, ls="--", zorder=1)
    ax.set_yticks(
        [(len(data) - 1 - i) * step + 0.3 for i in range(len(data))],
        [f"{'ResEnc: ' if r['group'] == 2 else ''}{r['label']}" for r in data],
        fontsize=11,
    )
    for tick, r in zip(ax.get_yticklabels(), data):
        tick.set_color(COLORS[r["kind"]])
    ax.set_xlim(-0.01, 0.075)
    ax.set_xticks([0, 0.02, 0.04, 0.06], ["0", "+0.02", "+0.04", "+0.06"])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#D9D9D9", lw=0.7, ls=(0, (3, 3)))
    ax.set_xlabel(
        "Dice change of each run against its control (one dot per fold and seed)",
        fontsize=11,
    )
    ax.tick_params(length=0, labelsize=11)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.subplots_adjust(left=0.27, right=0.75, top=0.9, bottom=0.17)
    ax.text(
        0.079,
        len(data) - 0.15,
        "mean    runs up",
        fontsize=11,
        fontweight="bold",
        va="bottom",
        color=INK,
    )
    fig.text(
        0.75,
        0.995,
        "Paired Dice change per run",
        ha="right",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(out / "augmentation_option_ridge.png", dpi=300)


def spread(heights: dict[int, float], gap: float = 0.024) -> dict[int, float]:
    """Heights moved apart so that neighbouring labels are at least `gap` apart.

    Args:
        heights: Label height per organ.
        gap: Smallest distance between two labels.

    Returns:
        The adjusted heights, keeping the order of the input heights.
    """
    order = sorted(heights, key=heights.get)
    out = dict(heights)
    for _ in range(50):
        for lower, upper in zip(order, order[1:]):
            if out[upper] - out[lower] < gap:
                mid = (out[upper] + out[lower]) / 2
                out[lower], out[upper] = mid - gap / 2, mid + gap / 2
    return out


LADDERS = [  # panel title, experiments from the control up, step labels
    ("ENet", [ENET, f"{ENET}_augmented", f"{ENET}_strong_geometric", f"{ENET}_strong_geometric_50ep"], ["No aug.", "Joint", "Strong", "Strong,\n50 epochs"]),
    ("ResEnc U-Net", [RESENC, f"{RESENC}_augmented", f"{RESENC}_strong_geometric"], ["No aug.", "Joint", "Strong"]),
]  # fmt: skip


def organ_slopes(runs: dict, out: Path) -> None:
    """Dice of every organ along the augmentation ladder of each model (mean over the 12 runs)."""
    plt = pyplot()
    fig, axes = plt.subplots(
        1, 2, figsize=(7.0, 3.3), gridspec_kw={"width_ratios": [4, 3]}, sharey=True
    )
    for ax, (title, experiments, steps) in zip(axes, LADDERS):
        values = {
            k: [np.mean([r[organ] for r in runs[e].values()]) for e in experiments]
            for k, organ in CLASSES.items()
        }
        for end, x, offset, align in (
            (0, 0, -10, "right"),
            (-1, len(experiments) - 1, 10, "left"),
        ):
            heights = spread({k: v[end] for k, v in values.items()})
            for k, v in values.items():
                ax.annotate(
                    f"{v[end]:.2f}",
                    (x, heights[k]),
                    (offset, 0),
                    textcoords="offset points",
                    ha=align,
                    va="center",
                    fontsize=11,
                    color=EARTH_LABEL_COLORS[k],
                )
        for k, y in values.items():
            ax.plot(
                range(len(y)),
                y,
                color=EARTH_LABEL_COLORS[k],
                lw=2.4,
                marker="o",
                ms=7,
                zorder=3,
            )
        mean = [np.mean([r["dice"] for r in runs[e].values()]) for e in experiments]
        ax.plot(
            range(len(mean)),
            mean,
            color=INK,
            lw=1.4,
            ls="--",
            marker="o",
            ms=5,
            zorder=2,
        )
        ax.set_xticks(range(len(steps)), steps, fontsize=11)
        ax.set_xlim(-0.7, len(steps) - 0.3)
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color="#D9D9D9", lw=0.7, ls=(0, (3, 3)))
        ax.tick_params(length=0, labelsize=11)
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_yticks([0.7, 0.8, 0.9], [".7", ".8", ".9"])
    axes[0].set_ylim(0.65, 0.97)
    axes[0].set_ylabel("Dice (3D, mean of 12 runs)", fontsize=11)
    handles = [
        plt.Line2D([], [], color=EARTH_LABEL_COLORS[k], lw=2.4, marker="o", label=organ)
        for k, organ in CLASSES.items()
    ]
    handles.append(
        plt.Line2D(
            [], [], color=INK, lw=1.4, ls="--", marker="o", ms=5, label="mean of organs"
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        fontsize=11,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.9))
    fig.text(
        axes[1].get_position().x1,
        0.995,
        "Dice of every organ along the augmentation steps",
        ha="right",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(out / "augmentation_option_organs.png", dpi=300)


ALT = {
    GEOMETRIC: "#2A9D8F",
    INTENSITY: "#E9A23B",
    JOINT: "#E4572E",
    STRONG: "#5B4B9A",
}  # brighter palette by type
STEP_COLORS = [
    "#8A8A8A",
    ALT[JOINT],
    ALT[STRONG],
    "#2B1F52",
]  # no augmentation, joint, strong, strong for 50 epochs


def mean_of(arm: dict, key: str) -> float:
    return float(np.mean([r[key] for r in arm.values()]))


def finish(fig, name: str, title: str, right: float, out: Path) -> None:
    """Right-aligned bold title at the given figure x position, then save as augmentation_option_<name>.png."""
    fig.text(right, 0.995, title, ha="right", va="top", fontsize=14, fontweight="bold")
    fig.savefig(out / f"augmentation_option_{name}.png", dpi=300)


def step_values(runs: dict, experiments: list[str], key: str) -> list[np.ndarray]:
    """The per-run values of a metric for every experiment of a ladder."""
    return [np.array([r[key] for r in runs[e].values()]) for e in experiments]


def violins(runs: dict, out: Path) -> None:
    """Distribution of the per-run Dice of every organ at every step, one row per model."""
    plt = pyplot()
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 4.3))
    for r, (row, (title, experiments, _)) in enumerate(zip(axes, LADDERS)):
        for ax, (k, organ) in zip(row, CLASSES.items()):
            data = step_values(runs, experiments, organ)
            parts = ax.violinplot(
                data, positions=range(len(data)), showextrema=False, widths=0.85
            )
            for body, color in zip(parts["bodies"], STEP_COLORS):
                body.set_facecolor(color)
                body.set_alpha(0.85)
                body.set_edgecolor("white")
            ax.plot(
                range(len(data)), [d.mean() for d in data], color=INK, lw=1.3, zorder=3
            )
            ax.set_xticks([])
            ax.set_xlim(-0.6, 3.6)
            ax.grid(axis="x", visible=False)
            ax.grid(axis="y", color="#D9D9D9", lw=0.7, ls=(0, (3, 3)))
            ax.tick_params(length=0, labelsize=11)
            ax.spines[["top", "right", "left"]].set_visible(False)
            if r == 0:
                ax.set_title(
                    organ, fontsize=12, fontweight="bold", color=EARTH_LABEL_COLORS[k]
                )
        row[0].set_ylabel(title, fontsize=12, fontweight="bold")
    names = ["No aug.", "Joint", "Strong", "Strong, 50 epochs"]
    handles = [
        plt.Line2D([], [], color=c, lw=8, label=n) for n, c in zip(names, STEP_COLORS)
    ]
    handles.append(plt.Line2D([], [], color=INK, lw=1.3, label="mean"))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        fontsize=11,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.2,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    finish(
        fig,
        "violins",
        "Dice per run and organ along the steps",
        axes[0, -1].get_position().x1,
        out,
    )


def folds(runs: dict, out: Path) -> None:
    """Dice of every fold (mean of its 3 seeds) along the steps of each model."""
    plt = pyplot()
    fig, axes = plt.subplots(
        1, 2, figsize=(7.0, 3.5), gridspec_kw={"width_ratios": [4, 3]}, sharey=True
    )
    colors = [ALT[GEOMETRIC], ALT[INTENSITY], ALT[JOINT], ALT[STRONG]]
    for ax, (title, experiments, steps) in zip(axes, LADDERS):
        values = {
            f: [
                np.mean([r["dice"] for (fold, _), r in runs[e].items() if fold == f])
                for e in experiments
            ]
            for f in range(4)
        }
        for end, x, offset, align in (
            (0, 0, -9, "right"),
            (-1, len(experiments) - 1, 9, "left"),
        ):
            heights = spread({f: v[end] for f, v in values.items()}, 0.012)
            for f, v in values.items():
                ax.annotate(
                    f"fold {f}",
                    (x, heights[f]),
                    (offset, 0),
                    textcoords="offset points",
                    ha=align,
                    va="center",
                    fontsize=11,
                    color=colors[f],
                )
        for f, y in values.items():
            ax.plot(range(len(y)), y, color=colors[f], lw=2.4)
        ax.set_xticks(range(len(steps)), steps, fontsize=11)
        ax.set_xlim(-0.9, len(steps) - 0.1)
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color="#D9D9D9", lw=0.7, ls=(0, (3, 3)))
        ax.tick_params(length=0, labelsize=11)
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_ylabel("Dice (3D, mean of 3 seeds)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    finish(
        fig,
        "folds",
        "Dice of every fold along the augmentation steps",
        axes[1].get_position().x1,
        out,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--out", type=Path, default=REPO / "figures")
    args = parser.parse_args(argv)
    runs = load_runs(args.runs)
    ridge(runs, args.out)
    organ_slopes(runs, args.out)
    for figure in (violins, folds):
        figure(runs, args.out)


if __name__ == "__main__":
    main()
