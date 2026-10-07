"""Dice of every cross-validation run, grouped by model and seed, to show how much the seeds differ.

    python dataset_analysis/seed_variability.py \\
        --experiments "2D baseline\\n(1 slice)=full_cv4_enet_dice_ce" "2.5D\\n(3 slices: ±1)=full_cv4_enet_dice_ce_25d_c1" \\
        [--title "..."] [--note "..."] [--out figures/context_25d]

--experiments, --title, --note and the 12-run requirement are as in architecture_progress.py, whose loading and layout code this reuses.
One panel per organ and one for the mean over the organs. Each model is a group of one column per seed: the Dice of its 4 folds as
small points (the same fold is joined across the seeds by a thin line) and the mean over the folds as a large marker.
"seed σ" above a group is the std of those three seed means (n-1); seed_variability.csv has it with the std over the folds of the
seed-averaged Dice, so the two sources of variation can be compared.

Writes seed_variability_dice.{png,pdf} and seed_variability.csv.
"""

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np
from matplotlib.patheffects import withStroke

from architecture_progress import (
    FG,
    FG_COLOR,
    ORGAN_COLOR,
    save,
    step_spec,
    tighten,
    load_steps,
)
from utils import CLASSES, INK, REPO, clean_axis, decorate, pyplot, write_csv

from plot_style import legend_below  # isort: skip -- needs the tools/ path utils adds

sys.path.append(str(REPO))  # appended, not inserted: see architecture_progress.py
from src.aggregate import CV_FOLDS, CV_SEEDS

MARKERS = "osD^v"  # one per seed, so the seeds also differ in grayscale
GROUP_GAP = 1.2  # empty columns between the groups
DEFAULT_TITLE = "Dice differs far less between seeds than between folds"


def dice_grid(step, organ: str) -> tuple[np.ndarray, list[int], list[int]]:
    """Dice of every run of the step for one organ as an array [fold, seed], with the fold and seed numbers of its axes."""
    folds = sorted({f for f, _ in step.runs})
    seeds = sorted({s for _, s in step.runs})
    key = f"eval.val_dice_{organ}"
    return np.array([[step.runs[f, s][key] for s in seeds] for f in folds], dtype=float), folds, seeds


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="+", type=step_spec, required=True, metavar="LABEL=EXPERIMENT")
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--note", default="", help="sentence appended to the subtitle (dataset, preprocessing, loss)")
    parser.add_argument("--metrics-dir", type=Path, default=REPO / "metrics")
    parser.add_argument("--out", type=Path, default=REPO / "figures/context_25d")
    args = parser.parse_args(argv)

    steps = load_steps(args.metrics_dir, args.experiments)
    first_run = next(iter(steps[0].runs.values()))
    organs = [name for name in CLASSES.values() if f"eval.val_dice_{name}" in first_run]
    panels = [*organs, FG]
    n_patients = len({row["patient"] for rows in steps[0].patient_rows.values() for row in rows})

    group_width = CV_SEEDS + GROUP_GAP
    centers = [g * group_width + (CV_SEEDS - 1) / 2 for g in range(len(steps))]
    plt = pyplot()
    fig, axes = plt.subplots(len(panels), 1, sharex=True, squeeze=False,
                             figsize=(max(9, 3.2 * len(steps) + 2), 1.9 * len(panels) + 1.4))
    table = []
    for ax, organ in zip(axes[:, 0], panels):
        color = ORGAN_COLOR.get(organ, FG_COLOR)
        for g, step in enumerate(steps):
            grid, folds, seeds = dice_grid(step, FG if organ == FG else organ)
            x = g * group_width + np.arange(len(seeds))
            for row in grid:  # one fold across the seeds
                ax.plot(x, row, color=color, alpha=0.3, lw=0.8, zorder=2)
                ax.scatter(x, row, s=14, color=color, alpha=0.5, linewidths=0, zorder=3)
            seed_means = np.nanmean(grid, axis=0)
            for xi, mean, marker in zip(x, seed_means, MARKERS):
                ax.scatter(xi, mean, s=60, marker=marker, color=color, edgecolor="white", linewidths=1.2, zorder=5)
            seed_sd = statistics.stdev(seed_means)
            fold_sd = statistics.stdev(np.nanmean(grid, axis=1))
            ax.text(centers[g], 0.97, f"seed σ {seed_sd:.3f}", transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=8.5, color=INK, zorder=6, path_effects=[withStroke(linewidth=2.5, foreground="white")])
            table.append({"step": step.name, "experiment": step.experiment, "organ": organ,
                          **{f"seed{s}_mean": m for s, m in zip(seeds, seed_means)},
                          "seed_std": seed_sd, "fold_std": fold_sd})
        ax.margins(x=0.04, y=0.2)
        ax.set_title("All organs (mean)" if organ == FG else organ.capitalize(), loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel("Dice")
        clean_axis(ax)
    bottom = axes[-1, 0]
    seeds = sorted({s for _, s in steps[0].runs})
    bottom.set_xticks([g * group_width + i for g in range(len(steps)) for i in range(len(seeds))],
                      [f"seed {s}" for _ in steps for s in seeds], fontsize=8.5)
    for center, step in zip(centers, steps):
        bottom.annotate(step.label.replace("\n", " "), (center, 0), xycoords=bottom.get_xaxis_transform(),
                        xytext=(0, -26), textcoords="offset points", ha="center", va="top", fontsize=10, color=INK)
    bottom.scatter([], [], s=14, color=INK, alpha=0.5, linewidths=0, label=f"fold ({CV_FOLDS} folds, joined across the seeds)")
    bottom.scatter([], [], s=60, marker="o", color=INK, edgecolor="white", label="mean over the folds, one marker shape per seed")
    legend_below(bottom, ncol=2)
    subtitle = f"{CV_FOLDS}-fold cross-validation × {CV_SEEDS} seeds, {n_patients} patients" + (f". {args.note}" if args.note else "")
    decorate(fig, args.title, subtitle=subtitle,
             footnote_text="Dice, 3D on the original CT grid, best checkpoint; each point is the mean over the patients of one validation fold. "
                           "seed σ: std over the 3 seeds of the fold-averaged Dice. Y-axes differ per organ.")
    tighten(fig, 0.45, -0.5)  # negative: room under the axes for the group names
    fig.subplots_adjust(hspace=0.55)

    args.out.mkdir(parents=True, exist_ok=True)
    save(plt, fig, args.out, "seed_variability_dice")
    write_csv(args.out / "seed_variability.csv", table)
    print(f"{len(steps)} steps -> {args.out}")


if __name__ == "__main__":
    main()
