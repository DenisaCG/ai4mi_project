"""Step-by-step architecture comparison from the 4-fold x 3-seed cross-validation runs in metrics/.

    python dataset_analysis/architecture_progress.py \\
        --experiments "ENet CE=full_cv4_enet_ce" "ENet Dice+CE=full_cv4_enet_dice_ce" "U-Net=full_cv4_unet_dice_ce" \\
        --compare "ENet Dice+CE" "U-Net" [--out figures/architecture]

--experiments gives the steps in order as LABEL=EXPERIMENT (a literal \\n in LABEL breaks the line); every experiment needs all 12 runs (4 folds x 3 seeds).
The numbers are those of `python -m src.aggregate`: the per-run values come from each run's summary.json (eval block),
the per-patient values from its metrics_3d.csv, averaged over the seeds as in cv_pooled_<experiment>.csv.

Writes, as .png + .pdf: architecture_progress_{dice,hd95,assd} (one panel per organ, the 12 runs of every step as
points, their mean as a marker joined across steps) and, with --compare A B, architecture_per_patient_<A>_vs_<B>
(Dice of every patient in A and B, grouped by fold). Also architecture_progress.csv with the plotted means and the
number of undefined (NaN) patient values per step and organ.
"""

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.patheffects import withStroke

from utils import (
    CLASSES,
    COLORS,
    INK,
    MUTED,
    REPO,
    clean_axis,
    decorate,
    pyplot,
    write_csv,
)

from plot_style import legend_below  # isort: skip -- needs the tools/ path utils adds

sys.path.append(
    str(REPO)
)  # appended, not inserted: the repo root has its own utils.py that must not shadow ours
from src.aggregate import (
    CV_FOLDS,
    CV_METRICS,
    CV_SEEDS,
    cv_runs,
    load_runs,
    nan_free_mean,
    pool_patients,
    read_patient_rows,
)

ORGAN_COLOR = {name: COLORS[label] for label, name in CLASSES.items()}
FOLD_GAP = 0.6  # extra rows between folds in the per-patient figure


@dataclass
class Step:
    label: str
    experiment: str
    runs: dict[tuple[int, int], dict]  # (fold, seed) -> flattened summary.json
    patient_rows: dict[
        tuple[int, int], list[dict]
    ]  # (fold, seed) -> metrics_3d.csv rows

    @property
    def name(self) -> str:
        """The label on one line, for titles, legends, file names and --compare."""
        return " ".join(self.label.split())


def step_spec(text: str) -> tuple[str, str]:
    """Splits LABEL=EXPERIMENT; a literal \\n in the label is a line break in the figure."""
    label, sep, experiment = text.rpartition("=")
    if not (sep and label and experiment):
        raise argparse.ArgumentTypeError(f"{text!r} is not LABEL=EXPERIMENT")
    return label.replace("\\n", "\n"), experiment


def load_steps(metrics_dir: Path, specs: list[tuple[str, str]]) -> list[Step]:
    """The steps in the given order.

    Args:
        metrics_dir: Directory holding `<experiment>_fold<k>/seed<s>/`.
        specs: (label, experiment) pairs.

    Returns:
        One Step per spec.

    Raises:
        SystemExit: If an experiment does not have all folds x seeds.
    """
    groups = cv_runs(load_runs(metrics_dir, ""))
    steps = []
    for label, experiment in specs:
        found = groups.get(experiment, {})
        if len(found) != CV_FOLDS * CV_SEEDS:
            raise SystemExit(
                f"{experiment}: {len(found)}/{CV_FOLDS * CV_SEEDS} runs found; only complete experiments are plotted"
            )
        rows = {key: read_patient_rows(metrics_dir, run) for key, run in found.items()}
        steps.append(Step(label, experiment, found, rows))
    return steps


def run_values(step: Step, key: str, organ: str) -> np.ndarray:
    """The metric of every run of the step for one organ; NaN where undefined."""
    return np.array(
        [run[f"eval.val_{key}_{organ}"] for run in step.runs.values()], dtype=float
    )


def nan_patients(step: Step, key: str, organ: str) -> int:
    """Patient values of the metric that are undefined, summed over the runs (as counted by src.aggregate)."""
    return sum(
        math.isnan(row[key])
        for rows in step.patient_rows.values()
        for row in rows
        if row["class_name"] == organ
    )


def tighten(fig, top_in: float, bottom_in: float) -> None:
    """Wins back white space that decorate() reserves above the first panel and between the axes and the legend."""
    h = fig.get_size_inches()[1]
    pars = fig.subplotpars
    fig.subplots_adjust(top=pars.top + top_in / h, bottom=pars.bottom - bottom_in / h)


def save(plt, fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def progress_figure(plt, steps, organs, metric, n_patients, out) -> None:
    """One metric: a panel per organ, the runs of each step as points and their mean joined across steps."""
    label, key, digits = metric
    x = np.arange(len(steps))
    jitter = np.random.default_rng(0).uniform(
        -0.14, 0.14, (len(steps), len(steps[0].runs))
    )
    fig, axes = plt.subplots(
        len(organs), 1, sharex=True, figsize=(9, 1.9 * len(organs) + 1.2), squeeze=False
    )
    any_nan = False
    for ax, organ in zip(axes[:, 0], organs):
        color = ORGAN_COLOR[organ]
        values = np.array([run_values(s, key, organ) for s in steps])
        means = [nan_free_mean(list(v)) for v in values]
        ax.scatter(
            x[:, None] + jitter,
            values,
            s=14,
            color=color,
            alpha=0.45,
            linewidths=0,
            zorder=3,
        )
        ax.plot(
            x, means, "-o", color=color, lw=1.8, ms=8, mec="white", mew=1.2, zorder=4
        )
        for xi, mean, step in zip(x, means, steps):
            if not math.isnan(mean):
                ax.annotate(
                    f"{mean:.{digits}f}",
                    (xi, mean),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                    color=INK,
                    zorder=5,
                    path_effects=[withStroke(linewidth=3, foreground="white")],
                )
            n_nan = nan_patients(step, key, organ)
            if n_nan:
                any_nan = True
                ax.text(
                    xi,
                    0.03,
                    f"{n_nan} NaN",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=MUTED,
                )
        ax.margins(x=0.07, y=0.2)
        ax.set_title(organ.capitalize(), loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel(label)
        clean_axis(ax)
    bottom = axes[-1, 0]
    bottom.set_xticks(x, [s.label for s in steps])
    bottom.scatter(
        [],
        [],
        s=14,
        color=INK,
        alpha=0.45,
        linewidths=0,
        label=f"run ({CV_FOLDS} folds × {CV_SEEDS} seeds)",
    )
    bottom.plot(
        [],
        [],
        "-o",
        color=INK,
        lw=1.8,
        ms=8,
        mec="white",
        mew=1.2,
        label=f"mean of the {len(steps[0].runs)} runs",
    )
    legend_below(bottom, ncol=2)
    footnote = (
        f"{label}, 3D on the original CT grid, best checkpoint; each run's value is the mean over the patients "
        "of its validation fold. Y-axes differ per organ."
    )
    if any_nan:
        footnote += " k NaN: patient values undefined (empty prediction or ground truth), left out of the means."
    decorate(
        fig,
        f"{label.split()[0]} per organ at each architecture step",
        subtitle=f"{CV_FOLDS}-fold cross-validation × {CV_SEEDS} seeds, {n_patients} patients",
        footnote_text=footnote,
    )
    tighten(fig, 0.45, 0.3)
    fig.subplots_adjust(hspace=0.55)
    save(plt, fig, out, f"architecture_progress_{key}")


def patient_figure(plt, a: Step, b: Step, organs, out: Path) -> None:
    """Dice of every patient in step a (open marker) and step b (filled), one row per patient, grouped by fold."""
    pooled_a = {
        (r["patient"], r["class_name"]): r for r in pool_patients(a.patient_rows)
    }
    pooled_b = {
        (r["patient"], r["class_name"]): r for r in pool_patients(b.patient_rows)
    }
    patients = sorted({(r["fold"], r["patient"]) for r in pooled_a.values()})
    y = np.array([i + FOLD_GAP * fold for i, (fold, _) in enumerate(patients)])
    fig, axes = plt.subplots(
        1,
        len(organs),
        sharey=True,
        squeeze=False,
        figsize=(3.1 * len(organs) + 1.3, 0.17 * (y[-1] + 1) + 1.9),
    )
    for ax, organ in zip(axes[0], organs):
        color = ORGAN_COLOR[organ]
        dice_a = np.array([pooled_a[p, organ]["dice"] for _, p in patients])
        dice_b = np.array([pooled_b[p, organ]["dice"] for _, p in patients])
        ax.hlines(y, dice_a, dice_b, color=color, alpha=0.5, lw=1.6, zorder=2)
        ax.scatter(
            dice_a,
            y,
            s=22,
            facecolor="white",
            edgecolor=color,
            linewidths=1.2,
            zorder=3,
        )
        ax.scatter(dice_b, y, s=22, color=color, zorder=4)
        ax.set_title(organ.capitalize(), loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Dice")
        clean_axis(ax, "x")
    first = axes[0, 0]
    first.set_yticks(y, [p for _, p in patients], fontsize=7)
    first.invert_yaxis()
    for fold in range(CV_FOLDS):
        rows = [yi for yi, (f, _) in zip(y, patients) if f == fold]
        first.annotate(
            f"fold {fold}",
            (0, np.mean(rows)),
            xycoords=first.get_yaxis_transform(),
            xytext=(-58, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=INK,
        )
    first.scatter(
        [], [], s=22, facecolor="white", edgecolor=INK, linewidths=1.2, label=a.name
    )
    first.scatter([], [], s=22, color=INK, label=b.name)
    legend_below(first, ncol=2)
    decorate(
        fig,
        f"Per-patient Dice, {a.name} and {b.name}",
        subtitle=f"{len(patients)} patients, one row each, grouped by fold ({CV_FOLDS}-fold cross-validation)",
        footnote_text=f"Each patient's 3D Dice (best checkpoint) averaged over {CV_SEEDS} seeds.",
    )
    tighten(fig, 0.45, 0.3)
    slug = [re.sub(r"\W+", "_", s.name).strip("_").lower() for s in (a, b)]
    save(plt, fig, out, f"architecture_per_patient_{slug[0]}_vs_{slug[1]}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        type=step_spec,
        required=True,
        metavar="LABEL=EXPERIMENT",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("A", "B"),
        help="labels of two steps for the per-patient figure",
    )
    parser.add_argument("--metrics-dir", type=Path, default=REPO / "metrics")
    parser.add_argument("--out", type=Path, default=REPO / "figures/architecture")
    args = parser.parse_args(argv)

    steps = load_steps(args.metrics_dir, args.experiments)
    by_label = {s.name: s for s in steps}
    if args.compare and not set(args.compare) <= set(by_label):
        raise SystemExit(
            f"--compare {args.compare} must be labels among {list(by_label)}"
        )
    first_run = next(iter(steps[0].runs.values()))
    organs = [name for name in CLASSES.values() if f"eval.val_dice_{name}" in first_run]
    n_patients = len(
        {row["patient"] for rows in steps[0].patient_rows.values() for row in rows}
    )

    args.out.mkdir(parents=True, exist_ok=True)
    plt = pyplot()
    table = []
    for metric in CV_METRICS:
        _, key, _ = metric
        progress_figure(plt, steps, organs, metric, n_patients, args.out)
        for step in steps:
            for organ in organs:
                values = run_values(step, key, organ)
                table.append(
                    {
                        "metric": key,
                        "step": step.name,
                        "experiment": step.experiment,
                        "organ": organ,
                        "n_runs": int(np.isfinite(values).sum()),
                        "n_nan": nan_patients(step, key, organ),
                        "mean": nan_free_mean(list(values)),
                    }
                )
    write_csv(args.out / "architecture_progress.csv", table)
    if args.compare:
        patient_figure(
            plt, by_label[args.compare[0]], by_label[args.compare[1]], organs, args.out
        )
    print(f"{len(steps)} steps -> {args.out}")


if __name__ == "__main__":
    main()
