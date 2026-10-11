"""Five figure options for the cross-validation slide, sized for the 11.9 x 3.85 in slot.

    python dataset_analysis/cv_options.py [--out figures/cv_options]

All options use the 12 runs of full_cv4_enet_dice_ce (4 folds x 3 seeds, 10 validation patients per fold, 3D Dice on the
original CT grid). A patient value is the mean over the four organs of the patient's Dice in one seed.
ecdf:     per fold the cumulative share of its 10 patients below a Dice value; one thin line per seed.
parallel: one line per patient across the three seeds, coloured by fold.
diagram:  the classic fold diagram (train and validation blocks) with the Dice of each fold on the right.
tiles:    the 40 patients as tiles grouped by fold, coloured by the patient's mean Dice over the 3 seeds.
pairs:    patient Dice of one seed against another, for the three seed pairs; dashed line = identical.
Writes cv_option_<name>.png.
"""

import argparse
import statistics
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from utils import GRID_COLOR, N_FOLDS, N_SEEDS, REPO, load_cv_runs, pyplot

from plot_style import PALETTE  # noqa: E402  (utils puts tools/ on the path)

EXPERIMENT = "full_cv4_enet_dice_ce"
FOLD_COLORS = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[4]]
CMAP = LinearSegmentedColormap.from_list("dice", ["#F4F1EA", "#5B84A8", "#2E5E6E"])
SIZE = (11.9, 3.85)


def patient_values(rows: list[dict]) -> dict[tuple[str, int], float]:
    """Mean over the four organs of a patient's Dice, per (patient, seed)."""
    cells: dict = {}
    for r in rows:
        cells.setdefault((r["patient"], r["seed"]), []).append(r["dice"])
    return {k: statistics.fmean(v) for k, v in cells.items()}


def patient_folds(rows: list[dict]) -> dict[str, int]:
    """Validation fold of every patient."""
    return {r["patient"]: r["fold"] for r in rows}


def style(ax, grid="y"):
    ax.grid(axis=grid, color=GRID_COLOR, lw=0.8, ls=(0, (3, 3)))
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11)


def option_ecdf(plt, rows, values, folds):
    fig, ax = plt.subplots(figsize=SIZE)
    for f in range(N_FOLDS):
        members = sorted(p for p, g in folds.items() if g == f)
        for s in range(N_SEEDS):
            v = np.sort([values[p, s] for p in members])
            ax.step(np.append(v, 1.0), np.append(np.arange(1, len(v) + 1) / len(v), 1.0), where="post", color=FOLD_COLORS[f], lw=2.0, alpha=0.9 if s == 0 else 0.55)  # fmt: skip
        ax.plot([], [], color=FOLD_COLORS[f], lw=3, label=f"fold {f}")
    style(ax, "both")
    ax.legend(loc="upper left", frameon=False, fontsize=12)
    ax.set_xlim(0.55, 0.95)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.5, 1], ["0", "half", "all"])
    ax.set_xlabel("patient Dice (mean of the 4 organs)", fontsize=12)
    ax.set_ylabel("share of the fold's patients", fontsize=12)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.17)
    return fig


def option_parallel(plt, rows, values, folds):
    fig, ax = plt.subplots(figsize=SIZE)
    for p, f in folds.items():
        ax.plot(
            range(N_SEEDS),
            [values[p, s] for s in range(N_SEEDS)],
            color=FOLD_COLORS[f],
            lw=1.8,
            alpha=0.8,
        )
    lowest = sorted(
        folds, key=lambda p: statistics.fmean(values[p, s] for s in range(N_SEEDS))
    )[:3]
    for p in lowest:
        ax.text(
            N_SEEDS - 0.95,
            values[p, N_SEEDS - 1],
            p.replace("Patient_", "P"),
            fontsize=11,
            va="center",
            color="#333333",
        )
    for f in range(N_FOLDS):
        ax.plot([], [], color=FOLD_COLORS[f], lw=3, label=f"fold {f}")
    ax.legend(
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.32),
        frameon=False,
        fontsize=12,
    )
    style(ax)
    ax.set_xticks(range(N_SEEDS), [f"seed {s}" for s in range(N_SEEDS)], fontsize=12)
    ax.set_xlim(-0.1, N_SEEDS - 0.7)
    ax.set_ylabel("patient Dice (mean of the 4 organs)", fontsize=12)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.25)
    return fig


def option_diagram(plt, rows, values, folds):
    fig = plt.figure(figsize=SIZE)
    ax = fig.add_axes([0.07, 0.1, 0.63, 0.82])
    for f in range(N_FOLDS):
        for b in range(4):
            val = b == f
            ax.add_patch(
                Rectangle(
                    (b * 1.02, f * 1.1),
                    1.0,
                    0.9,
                    color=FOLD_COLORS[f] if val else "#E4E4E4",
                    lw=0,
                )
            )
            ax.text(b * 1.02 + 0.5, f * 1.1 + 0.45, "validate\n10 patients" if val else "train\n10", ha="center", va="center", fontsize=11, color="white" if val else "#555555", fontweight="bold" if val else "normal", linespacing=1.1)  # fmt: skip
    ax.set_xlim(0, 4.1)
    ax.set_ylim(4.35, -0.1)
    ax.set_yticks(
        [f * 1.1 + 0.45 for f in range(N_FOLDS)],
        [f"Fold {f}" for f in range(N_FOLDS)],
        fontsize=12,
    )
    ax.set_xticks([])
    ax.tick_params(length=0)
    ax.grid(False)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.text(0.72, 0.94, "Mean Dice, 3 seeds", fontsize=12, fontweight="bold")
    for f in range(N_FOLDS):
        members = [p for p, g in folds.items() if g == f]
        seed_means = [
            statistics.fmean(values[p, s] for p in members) for s in range(N_SEEDS)
        ]
        y = 0.1 + 0.82 * (1 - (f * 1.1 + 0.45 + 0.1) / 4.45)
        fig.text(
            0.72,
            y,
            f"{statistics.fmean(seed_means):.3f}",
            fontsize=16,
            fontweight="bold",
            color=FOLD_COLORS[f],
            va="center",
        )
        fig.text(
            0.82,
            y,
            "  ".join(f"{m:.3f}" for m in seed_means),
            fontsize=11,
            va="center",
            color="#333333",
        )
    return fig


def option_tiles(plt, rows, values, folds):
    fig = plt.figure(figsize=SIZE)
    for f in range(N_FOLDS):
        ax = fig.add_axes([0.02 + f * 0.245, 0.05, 0.23, 0.78])
        members = sorted(
            (p for p, g in folds.items() if g == f),
            key=lambda p: -statistics.fmean(values[p, s] for s in range(N_SEEDS)),
        )
        for k, p in enumerate(members):
            v = statistics.fmean(values[p, s] for s in range(N_SEEDS))
            r, c = divmod(k, 2)
            ax.add_patch(
                Rectangle(
                    (c * 1.02, r * 1.02), 1.0, 1.0, color=CMAP((v - 0.55) / 0.4), lw=0
                )
            )
            ax.text(c * 1.02 + 0.5, r * 1.02 + 0.5, f"{p.replace('Patient_', 'P')}\n{v:.2f}".replace("0.", "."), ha="center", va="center", fontsize=11, color="white" if v > 0.8 else "#333333", linespacing=1.1)  # fmt: skip
        ax.set_xlim(0, 2.04)
        ax.set_ylim(5.1, 0)
        ax.axis("off")
        mean = statistics.fmean(values[p, s] for p in members for s in range(N_SEEDS))
        fig.text(
            0.02 + f * 0.245,
            0.9,
            f"Fold {f}",
            fontsize=13,
            fontweight="bold",
            color="#333333",
        )
        fig.text(
            0.02 + f * 0.245 + 0.23,
            0.9,
            f"mean {mean:.3f}",
            fontsize=12,
            ha="right",
            color="#333333",
        )
    return fig


def option_pairs(plt, rows, values, folds):
    fig, axes = plt.subplots(1, 3, figsize=SIZE)
    for ax, (a, b) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
        ax.plot([0.55, 0.95], [0.55, 0.95], color="#999999", lw=1.2, ls=(0, (4, 3)))
        for f in range(N_FOLDS):
            members = [p for p, g in folds.items() if g == f]
            ax.scatter([values[p, a] for p in members], [values[p, b] for p in members], s=46, color=FOLD_COLORS[f], edgecolor="white", lw=0.7, zorder=3, label=f"fold {f}")  # fmt: skip
        style(ax, "both")
        ax.set_xlim(0.55, 0.95)
        ax.set_ylim(0.55, 0.95)
        ax.set_aspect("equal")
        ax.set_xticks([0.6, 0.7, 0.8, 0.9], [".6", ".7", ".8", ".9"])
        ax.set_yticks([0.6, 0.7, 0.8, 0.9], [".6", ".7", ".8", ".9"])
        ax.set_xlabel(f"seed {a}", fontsize=12)
        ax.set_ylabel(f"seed {b}", fontsize=12)
    axes[0].legend(loc="upper left", frameon=False, fontsize=11, handletextpad=0.1)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.97, bottom=0.17, wspace=0.25)
    return fig


OPTIONS = {
    "ecdf": option_ecdf,
    "parallel": option_parallel,
    "diagram": option_diagram,
    "tiles": option_tiles,
    "pairs": option_pairs,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metrics-dir", type=Path, default=REPO / "metrics")
    parser.add_argument("--out", type=Path, default=REPO / "figures/cv_options")
    args = parser.parse_args(argv)
    plt = pyplot()
    rows = load_cv_runs(args.metrics_dir, EXPERIMENT)
    values, folds = patient_values(rows), patient_folds(rows)
    for name, build in OPTIONS.items():
        build(plt, rows, values, folds).savefig(args.out / f"cv_option_{name}.png")


if __name__ == "__main__":
    main()
