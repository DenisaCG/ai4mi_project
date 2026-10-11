"""Loss variants on fold 0 of the 4-fold CV (ENet, full preprocessing), from run folders in metrics/ that need not be in this repo.

    python dataset_analysis/fold0_loss_variants.py \\
        --groups "Dice + CE\\n(baseline)=metrics/full_cv4_enet_dice_ce_fold0" \\
                 "Weighted CE\\n+ Dice=metrics/full_cv4_enet_weighted_dice_ce_preprocessing_fold0" ... \\
        [--out figures/fold0_loss_variants] [--title "{metric} ..."] [--note "..."]

--groups gives the variants in order as LABEL=DIR (a literal \\n in LABEL breaks the line), DIR holding seed<s>/summary.json
(+ epochs.csv). Groups may have different seed counts; the count is printed under each label and nothing is dropped.

Writes, as .png + .pdf: fold0_loss_variants_{dice,hd95,assd} (one panel per organ and one for the mean over the organs, one
point per seed, their mean as a marker; same layout as architecture_progress.py), fold0_loss_variants_curves (validation Dice per
epoch) and fold0_loss_variants.csv with the per-seed values.
"""

import argparse
import csv
import json
import math
import sys
import textwrap
from pathlib import Path

import numpy as np
from matplotlib.patheffects import withStroke

from utils import CLASSES, INK, MUTED, REPO, clean_axis, decorate, pyplot, write_csv

from plot_style import PALETTE, legend_below  # isort: skip -- needs the tools/ path utils adds

sys.path.append(
    str(REPO)
)  # appended, not inserted: the repo root has its own utils.py that must not shadow ours
from architecture_progress import FG, FG_COLOR, ORGAN_COLOR, save, tighten
from src.aggregate import CV_METRICS, nan_free_mean

DEFAULT_TITLE = "{metric} per organ for the loss variants on fold 0"


def group_spec(text: str) -> tuple[str, Path]:
    """Splits LABEL=DIR; a literal \\n in the label is a line break in the figure."""
    label, sep, folder = text.rpartition("=")
    if not (sep and label and folder):
        raise argparse.ArgumentTypeError(f"{text!r} is not LABEL=DIR")
    return label.replace("\\n", "\n"), Path(folder)


def load_group(folder: Path) -> dict[int, dict]:
    """{seed: {"eval": summary.json eval block, "epochs": epochs.csv rows}} of one run folder."""
    runs = {}
    for summary in sorted(folder.glob("seed*/summary.json")):
        seed = int(summary.parent.name.removeprefix("seed"))
        with open(summary.parent / "epochs.csv") as f:
            epochs = [{k: float(v) for k, v in row.items() if k.startswith("val_dice_")} for row in csv.DictReader(f)]
        runs[seed] = {"eval": json.loads(summary.read_text())["eval"], "epochs": epochs}
    if not runs:
        raise SystemExit(f"{folder}: no seed*/summary.json found")
    return runs


def values(runs: dict[int, dict], key: str, organ: str) -> np.ndarray:
    return np.array([r["eval"][f"val_{key}_{organ}"] for r in runs.values()], dtype=float)


def metric_figure(plt, groups, organs, metric, out, title, note) -> None:
    """One metric: a panel per organ, the seeds of each variant as points and their mean as a marker."""
    label, key, digits = metric
    panels = [*organs, FG]
    x = np.arange(len(groups))
    n_max = max(len(runs) for _, runs in groups)
    jitter = np.random.default_rng(0).uniform(-0.12, 0.12, (len(groups), n_max))
    fig, axes = plt.subplots(
        len(panels), 1, sharex=True, figsize=(max(8, 1.9 * len(groups) + 2), 1.9 * len(panels) + 1.2), squeeze=False
    )
    for ax, organ in zip(axes[:, 0], panels):
        color = ORGAN_COLOR.get(organ, FG_COLOR)
        vals = [values(runs, key, organ) for _, runs in groups]
        for i, v in enumerate(vals):
            ax.scatter(x[i] + jitter[i, : len(v)], v, s=22, color=color, alpha=0.5, linewidths=0, zorder=3)
            mean = nan_free_mean(list(v))
            ax.plot(x[i], mean, "o", color=color, ms=9, mec="white", mew=1.2, zorder=4)
            if not math.isnan(mean):
                ax.annotate(
                    f"{mean:.{digits}f}", (x[i], mean), xytext=(0, 10), textcoords="offset points", ha="center",
                    fontsize=9, fontweight="bold", color=INK, zorder=6,
                    path_effects=[withStroke(linewidth=2.5, foreground="white")],
                )
        ax.margins(x=0.08, y=0.25)
        ax.set_title("All organs (mean)" if organ == FG else organ.capitalize(), loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel(label)
        clean_axis(ax)
    bottom = axes[-1, 0]
    bottom.set_xticks(
        x,
        [
            "\n".join(textwrap.fill(line, 18) for line in lab.split("\n")) + f"\n(n = {len(runs)} seeds)"
            for (lab, runs) in groups
        ],
        fontsize=10,
    )
    bottom.scatter([], [], s=22, color=INK, alpha=0.5, linewidths=0, label="run (one seed)")
    bottom.plot([], [], "o", color=INK, ms=9, mec="white", mew=1.2, label="mean of the seeds")
    legend_below(bottom, ncol=2)
    decorate(
        fig,
        title.format(metric=label.split()[0]),
        subtitle="Fold 0 of the 4-fold cross-validation (10 validation patients), ENet" + (f". {note}" if note else ""),
        footnote_text=(
            f"{label}, 3D on the original CT grid, best checkpoint; each run's value is the mean over the 10 patients of fold 0. "
            f"All organs = mean of the {len(organs)} organs per run. Y-axes differ per organ. One fold only: a few seeds, "
            "so differences of the size of the seed spread are not conclusive."
        ),
    )
    tighten(fig, 0.45, 0.3)
    fig.subplots_adjust(hspace=0.55)
    save(plt, fig, out, f"fold0_loss_variants_{key}")


def curves_figure(plt, groups, out, note) -> None:
    """Validation Dice per epoch: the mean over the seeds of each variant, one thin line per seed."""
    panels = (("val_dice_fg", "All organs (mean)"), ("val_dice_esophagus", "Esophagus"))
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 4.4), sharex=True)
    for ax, (column, heading) in zip(axes, panels):
        for i, (lab, runs) in enumerate(groups):
            color = PALETTE[i % len(PALETTE)]
            curves = [[row[column] for row in r["epochs"]] for r in runs.values()]
            n = min(len(c) for c in curves)
            for c in curves:
                ax.plot(range(len(c)), c, color=color, lw=0.7, alpha=0.35)
            ax.plot(range(n), np.mean([c[:n] for c in curves], axis=0), color=color, lw=2.2, label=" ".join(lab.split()))
        ax.set_title(heading, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("epoch")
        ax.set_ylabel("Dice")
        clean_axis(ax)
    legend_below(axes[0], ncol=2)
    decorate(
        fig,
        "Validation Dice per epoch for the loss variants on fold 0",
        subtitle="Thick line = mean over the seeds, thin lines = single seeds" + (f". {note}" if note else ""),
        footnote_text="2D validation Dice of the 10 fold-0 patients after every epoch (not the 3D evaluation of the best checkpoint).",
    )
    save(plt, fig, out, "fold0_loss_variants_curves")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", nargs="+", type=group_spec, required=True, metavar="LABEL=DIR")
    parser.add_argument("--out", type=Path, default=REPO / "figures/fold0_loss_variants")
    parser.add_argument("--title", default=DEFAULT_TITLE, help="metric figure title; {metric} is the metric name")
    parser.add_argument("--note", default="", help="sentence appended to every figure subtitle")
    args = parser.parse_args(argv)

    groups = [(label, load_group(folder)) for label, folder in args.groups]
    first = next(iter(groups[0][1].values()))["eval"]
    organs = [name for name in CLASSES.values() if f"val_dice_{name}" in first]
    args.out.mkdir(parents=True, exist_ok=True)
    plt = pyplot()
    table = []
    for metric in CV_METRICS:
        _, key, _ = metric
        metric_figure(plt, groups, organs, metric, args.out, args.title, args.note)
        for (label, runs), folder in zip(groups, (f for _, f in args.groups)):
            for organ in [*organs, FG]:
                v = values(runs, key, organ)
                table.append(
                    {
                        "metric": key,
                        "variant": " ".join(label.split()),
                        "folder": str(folder),
                        "organ": organ,
                        "n_seeds": len(v),
                        "mean": nan_free_mean(list(v)),
                        **{f"seed{s}": float(x) for s, x in zip(runs, v)},
                    }
                )
    curves_figure(plt, groups, args.out, args.note)
    write_csv(args.out / "fold0_loss_variants.csv", table)
    print(f"{len(groups)} variants -> {args.out}")


if __name__ == "__main__":
    main()
