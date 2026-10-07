"""Step-by-step architecture comparison from the 4-fold x 3-seed cross-validation runs in metrics/.

    python dataset_analysis/architecture_progress.py \\
        --experiments "ENet CE=full_cv4_enet_ce" "ENet Dice+CE=full_cv4_enet_dice_ce" "U-Net=full_cv4_unet_dice_ce" \\
        --sections "sequence:ENet CE,ENet Dice+CE,U-Net" "U-Net + one change:U-Net + attention gates" \\
        --compare "ENet Dice+CE" "U-Net" [--out figures/architecture] [--title "{metric} ..."] [--note "..."]

--experiments gives the steps in order as LABEL=EXPERIMENT (a literal \\n in LABEL breaks the line); every experiment needs all 12 runs (4 folds x 3 seeds).
--sections groups the steps as HEADING:LABEL,LABEL,... (labels without line breaks); without it all steps form one section.
The first section is the sequence: its means are joined by a line. In every later section the steps are not joined, and a
dashed line per organ marks the mean of the last step of the first section, which is what those steps are compared with.
The numbers are those of `python -m src.aggregate`: the per-run values come from each run's summary.json (eval block),
the per-patient values from its metrics_3d.csv, averaged over the seeds as in cv_pooled_<experiment>.csv.

Writes, as .png + .pdf: architecture_progress_{dice,hd95,assd} (one panel per organ and one for the mean over the organs, the 12 runs of every step as
points, their mean as a marker) and, with --compare A B, architecture_per_patient_<A>_vs_<B>
(Dice of every patient in A and B, grouped by fold). Also architecture_progress.csv with the plotted means and the
number of undefined (NaN) patient values per step and organ.
"""

import argparse
import itertools
import math
import re
import sys
import textwrap
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
FG = "fg"  # the eval key of the mean over the organs of a run
FG_COLOR = "#555555"  # neutral, not an organ colour
FOLD_GAP = 0.6  # extra rows between folds in the per-patient figure
SECTION_GAP = 0.7  # extra x distance between sections in the progress figures
DEFAULT_TITLE = "{metric} per organ at each architecture step"


def flat(text: str) -> str:
    """A step label on one line, with a literal \\n or a line break read as a space."""
    return " ".join(text.replace("\\n", " ").split())


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
        return flat(self.label)


def step_spec(text: str) -> tuple[str, str]:
    """Splits LABEL=EXPERIMENT; a literal \\n in the label is a line break in the figure."""
    label, sep, experiment = text.rpartition("=")
    if not (sep and label and experiment):
        raise argparse.ArgumentTypeError(f"{text!r} is not LABEL=EXPERIMENT")
    return label.replace("\\n", "\n"), experiment


def section_spec(text: str) -> tuple[str, list[str]]:
    """Splits HEADING:LABEL,LABEL,... into the heading and the one-line step labels."""
    heading, sep, labels = text.partition(":")
    if not (sep and labels):
        raise argparse.ArgumentTypeError(f"{text!r} is not HEADING:LABEL,LABEL")
    return heading.strip(), [flat(label) for label in labels.split(",")]


def group_steps(
    sections: list[tuple[str, list[str]]] | None, steps: list[Step]
) -> list[tuple[str, list[Step]]]:
    """The steps grouped into the sections, or one unnamed section with every step.

    Raises:
        SystemExit: If a section names an unknown step or a step is in no section.
    """
    if not sections:
        return [("", steps)]
    by_name = {s.name: s for s in steps}
    named = [name for _, names in sections for name in names]
    if unknown := set(named) - set(by_name):
        raise SystemExit(
            f"--sections names unknown steps {sorted(unknown)}; known: {list(by_name)}"
        )
    if unused := set(by_name) - set(named):
        raise SystemExit(f"steps {sorted(unused)} are in no section")
    return [(heading, [by_name[n] for n in names]) for heading, names in sections]


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
    """Undefined values of the metric, summed over the runs, as counted by src.aggregate.

    For an organ these are patient values; for the fg mean, organ values missing from a run's mean.
    """
    if organ == FG:
        return sum(
            math.isnan(run[f"eval.val_{key}_{name}"])
            for run in step.runs.values()
            for name in CLASSES.values()
            if f"eval.val_{key}_{name}" in run
        )
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


def progress_figure(plt, sections, organs, metric, n_patients, out, title=DEFAULT_TITLE, note="") -> None:
    """One metric: a panel per organ, the runs of each step as points and their mean as a marker.

    The means of the first section are joined by a line. Each later section starts after a vertical separator, its
    means are not joined, and a dashed line marks the last step of the first section.
    """
    label, key, digits = metric
    panels = [*organs, FG]
    steps = [s for _, group in sections for s in group]
    positions, start = [], 0.0
    for _, group in sections:
        positions.append(start + np.arange(len(group)))
        start = positions[-1][-1] + 1 + SECTION_GAP
    x = np.concatenate(positions)
    reference = (
        len(sections[0][1]) - 1
    )  # index of the step the later sections are compared with
    jitter = np.random.default_rng(0).uniform(
        -0.14, 0.14, (len(steps), len(steps[0].runs))
    )
    fig, axes = plt.subplots(
        len(panels),
        1,
        sharex=True,
        figsize=(max(9, 1.6 * (x[-1] + 1) + 1.4), 1.9 * len(panels) + 1.2),
        squeeze=False,
    )
    any_nan = False
    for ax, organ in zip(axes[:, 0], panels):
        color = ORGAN_COLOR.get(organ, FG_COLOR)
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
        first = 0
        for i, xs in enumerate(positions):
            group_means = means[first : first + len(xs)]
            first += len(xs)
            ax.plot(
                xs,
                group_means,
                "-o" if i == 0 else "o",
                color=color,
                lw=1.8,
                ms=8,
                mec="white",
                mew=1.2,
                zorder=4,
            )
            if i:
                ax.hlines(
                    means[reference],
                    xs[0] - 0.45,
                    xs[-1] + 0.45,
                    color=color,
                    linestyles="--",
                    linewidth=1,
                    zorder=2,
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
                    zorder=6,
                    path_effects=[withStroke(linewidth=2.5, foreground="white")],
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
        for before, after in itertools.pairwise(positions):
            ax.axvline(
                (before[-1] + after[0]) / 2, color=MUTED, linewidth=0.8, alpha=0.6
            )
        ax.margins(x=0.04, y=0.2)
        ax.set_title(
            "All organs (mean)" if organ == FG else organ.capitalize(),
            loc="left",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_ylabel(label)
        clean_axis(ax)
    top, bottom = axes[0, 0], axes[-1, 0]
    for (heading, _), xs in zip(sections, positions):
        if heading:
            top.annotate(
                heading,
                ((xs[0] + xs[-1]) / 2, 1),
                xycoords=top.get_xaxis_transform(),
                xytext=(0, 26),  # above the organ title, on a row of its own
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                color=INK,
            )
    bottom.set_xticks(
        x,
        [
            "\n".join(textwrap.fill(line, 16) for line in s.label.split("\n"))
            for s in steps
        ],
        fontsize=10,
    )
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
    if len(sections) > 1:
        bottom.plot(
            [], [], "--", color=INK, lw=1, label=f"mean of {steps[reference].name}"
        )
    legend_below(bottom, ncol=3)
    footnote = (
        f"{label}, 3D on the original CT grid, best checkpoint; each run's value is the mean over the patients "
        f"of its validation fold. All organs = mean of the {len(organs)} organs per run. Y-axes differ per organ."
    )
    if any_nan:
        footnote += " k NaN: patient values undefined (empty prediction or ground truth), left out of the means."
    decorate(
        fig,
        title.format(metric=label.split()[0]),
        subtitle=f"{CV_FOLDS}-fold cross-validation × {CV_SEEDS} seeds, {n_patients} patients" + (f". {note}" if note else ""),
        footnote_text=footnote,
    )
    tighten(fig, 0.45, 0.3)
    fig.subplots_adjust(hspace=0.55)
    save(plt, fig, out, f"architecture_progress_{key}")


def patient_figure(plt, a: Step, b: Step, organs, out: Path, note: str = "") -> None:
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
        subtitle=f"{len(patients)} patients, one row each, grouped by fold ({CV_FOLDS}-fold cross-validation)" + (f". {note}" if note else ""),
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
    parser.add_argument(
        "--sections",
        nargs="+",
        type=section_spec,
        metavar="HEADING:LABEL,LABEL",
        help="group the steps into sections; the first is the sequence, the others are compared with its last step",
    )
    parser.add_argument("--metrics-dir", type=Path, default=REPO / "metrics")
    parser.add_argument("--out", type=Path, default=REPO / "figures/architecture")
    parser.add_argument("--title", default=DEFAULT_TITLE, help="progress figure title; {metric} is the metric name")
    parser.add_argument("--note", default="", help="sentence appended to every figure subtitle (dataset, preprocessing, loss)")
    args = parser.parse_args(argv)

    steps = load_steps(args.metrics_dir, args.experiments)
    by_label = {s.name: s for s in steps}
    sections = group_steps(args.sections, steps)
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
        progress_figure(plt, sections, organs, metric, n_patients, args.out, args.title, args.note)
        for heading, group in sections:
            for step, organ in itertools.product(group, [*organs, FG]):
                values = run_values(step, key, organ)
                table.append(
                    {
                        "metric": key,
                        "section": heading,
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
            plt, by_label[args.compare[0]], by_label[args.compare[1]], organs, args.out, args.note
        )
    print(f"{len(steps)} steps -> {args.out}")


if __name__ == "__main__":
    main()
