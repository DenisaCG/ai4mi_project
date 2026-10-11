"""Compare the 50-epoch ResEnc DS U-Net CV runs: boundary loss (none / ramped / fixed 0.08) x input (2D / 2.5D).

All runs: strong geometric augmentation, seed 0, 4-fold CV on the 40 patients (10 val per fold). The folds are shared, so each run is also
compared with the 2D control fold by fold. Reads runs/<experiment>_fold<k>/seed0/{eval/metrics_3d.csv,epochs.csv}; each point is one fold.

    python dataset_analysis/resenc_50ep_comparison.py [--out figures/resenc_50ep_comparison]

Writes, as PNG + PDF: 3D Dice, HD95 and ASSD per organ, the paired change against the control, and the validation curves, plus a per-fold CSV.
3D numbers use the best (val-selected) checkpoint; the curves are the 2D validation Dice of every epoch.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from matplotlib.patheffects import withStroke

from utils import CLASSES, INK, REPO, clean_axis, decorate, pyplot, write_csv
from plot_style import EARTH, legend_below

sys.path.append(str(REPO))
from architecture_progress import FG, FG_COLOR, save, tighten

BASE = "full_cv4_resenc_ds_unet_dice_ce_"
FOLDS = range(4)
ORGANS = list(CLASSES.values())
# label, experiment, colour (the loss), marker (the input: circle 2D, diamond 2.5D)
RUNS = [
    ("2D · no boundary (control)", BASE + "strong_geometric_50ep", FG_COLOR, "o"),
    ("2D · ramped boundary", BASE + "strong_geometric_50ep_boundary", EARTH[0], "o"),
    ("2D · fixed boundary 0.08", BASE + "strong_geometric_50ep_boundary_w008", EARTH[1], "o"),
    ("2.5D · ramped boundary", BASE + "boundary_25d_c1_strong_geometric_50ep", EARTH[0], "D"),
    ("2.5D · fixed boundary 0.08", BASE + "25d_c1_strong_geometric_50ep_boundary_w008", EARTH[1], "D"),
]
METRICS = {  # key: (axis label, digits, higher is better)
    "dice": ("3D Dice", 3, True),
    "hd95": ("3D HD95 (mm)", 1, False),
    "assd": ("3D ASSD (mm)", 2, False),
}
SUBTITLE = "ResEnc DS U-Net · strong geometric augmentation · 50 epochs · seed 0 · 4-fold CV, 10 val patients per fold"


def load(experiment: str) -> dict:
    """Per fold: patient-averaged 3D metric per organ (best checkpoint) and the 2D validation Dice per epoch."""
    out = {"3d": [], "curve": []}
    for fold in FOLDS:
        run = REPO / "runs" / f"{experiment}_fold{fold}" / "seed0"
        with (run / "eval" / "metrics_3d.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        per_organ = {}
        for organ in ORGANS:
            sel = [r for r in rows if r["class_name"] == organ]
            per_organ[organ] = {k: float(np.nanmean([float(r[k]) for r in sel])) for k in METRICS}
        per_organ[FG] = {k: float(np.mean([per_organ[o][k] for o in ORGANS])) for k in METRICS}
        out["3d"].append(per_organ)
        with (run / "epochs.csv").open(newline="") as f:
            ep = list(csv.DictReader(f))
        out["curve"].append({m: np.array([float(r[f"val_dice_{m}"]) for r in ep]) for m in (*ORGANS, FG)})
    return out


def value(data, run, fold, organ, key):
    return data[run]["3d"][fold][organ][key]


def metric_figure(plt, data, key, out):
    label, digits, higher = METRICS[key]
    panels = [*ORGANS, FG]
    fig, axes = plt.subplots(len(panels), 1, sharex=True, figsize=(11, 1.95 * len(panels) + 1.9), squeeze=False)
    for ax, organ in zip(axes[:, 0], panels):
        for i, (name, exp, color, marker) in enumerate(RUNS):
            v = np.array([value(data, exp, f, organ, key) for f in FOLDS])
            ax.scatter(i + (np.arange(4) - 1.5) * 0.06, v, s=24, color=color, alpha=0.45, marker=marker, edgecolors="none", zorder=3)
            mean, sd = v.mean(), v.std(ddof=1)
            ax.errorbar(i, mean, yerr=sd, fmt=marker, color=color, markersize=9, markeredgecolor="white", markeredgewidth=1.3,
                        capsize=6, elinewidth=2, zorder=5)
            ax.annotate(f"{mean:.{digits}f} ± {sd:.{digits}f}", (i, max(mean + sd, v.max())), xytext=(0, 7), textcoords="offset points", ha="center",
                        fontsize=8.5, fontweight="bold", color=INK, path_effects=[withStroke(linewidth=2.8, foreground="white")], zorder=7)
        ax.set_title("All organs (mean)" if organ == FG else organ.capitalize(), loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel(label)
        ax.margins(x=0.1, y=0.45)
        clean_axis(ax)
    bottom = axes[-1, 0]
    bottom.set_xticks(range(len(RUNS)), [n.replace(" · ", "\n") for n, *_ in RUNS], fontsize=9)
    bottom.scatter([], [], s=25, color="#999999", alpha=0.6, label="One fold")
    bottom.plot([], [], "o", color=INK, ms=8, label="Mean ± SD over the 4 folds")
    for color, name in ((FG_COLOR, "No boundary"), (EARTH[0], "Ramped boundary"), (EARTH[1], "Fixed boundary 0.08")):
        bottom.plot([], [], "s", color=color, ms=8, label=name)
    bottom.plot([], [], "o", color="#999999", ms=8, label="2D input")
    bottom.plot([], [], "D", color="#999999", ms=7, label="2.5D input (3 slices)")
    legend_below(bottom, ncol=4)
    decorate(fig, f"{label.split(' (')[0]} per organ: boundary loss and 2.5D input", subtitle=SUBTITLE
             + f" · {'higher' if higher else 'lower'} is better",
             footnote_text=f"{label} on the original CT grid at the best (val-selected) epoch; each point is a patient-averaged fold. Large marker = mean and SD over "
                           "the 4 folds. Colour = loss, shape = input. One seed and 10 patients per fold: differences of a few thousandths of Dice are within noise. "
                           "Y-axes differ by organ.")
    tighten(fig, 0.5, 0.0)
    fig.subplots_adjust(hspace=0.6)
    save(plt, fig, out, f"resenc_50ep_{key}")


def paired_figure(plt, data, out):
    """Change against the 2D control on the same fold: Dice up and HD95 down is better."""
    control = RUNS[0][1]
    others = RUNS[1:]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, (key, sign, title, digits) in zip(axes, (("dice", 1, "Dice change", 3), ("hd95", -1, "HD95 reduction (mm)", 1))):
        for i, (name, exp, color, marker) in enumerate(others):
            d = np.array([sign * (value(data, exp, f, FG, key) - value(data, control, f, FG, key)) for f in FOLDS])
            ax.scatter(d, np.full(4, i) + (np.arange(4) - 1.5) * 0.05, s=26, color=color, alpha=0.5, marker=marker, edgecolors="none", zorder=3)
            ax.plot(d.mean(), i, marker, color=color, ms=11, markeredgecolor="white", markeredgewidth=1.3, zorder=5)
            ax.annotate(f"{d.mean():+.{digits}f}", (d.mean(), i), xytext=(0, 14), textcoords="offset points", ha="center", fontsize=9,
                        fontweight="bold", color=INK, path_effects=[withStroke(linewidth=2.8, foreground="white")], zorder=7)
        ax.axvline(0, color=INK, lw=1.2, ls="--", zorder=2)
        ax.set_title(title + " vs the 2D control (higher is better)", loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Dice (mean over organs)" if key == "dice" else "mm (mean over organs)")
        ax.set_yticks(range(len(others)), [n.replace(" · ", "\n") for n, *_ in others], fontsize=9)
        ax.invert_yaxis()
        ax.margins(x=0.15, y=0.18)
        clean_axis(ax, grid_axis="x")
    axes[0].scatter([], [], s=25, color="#999999", alpha=0.6, label="One fold (run minus control, same fold)")
    axes[0].plot([], [], "o", color=INK, ms=9, label="Mean over the 4 folds")
    legend_below(axes[0], ncol=2)
    decorate(fig, "Paired change against the 2D control without boundary loss", subtitle=SUBTITLE,
             footnote_text="Each point is one fold's 3D metric of the run minus the control's on the same 10 validation patients (best checkpoint, mean over the four "
                           "organs); HD95 is shown as a reduction so right is better on both. Colour = loss, shape = input (circle 2D, diamond 2.5D).")
    tighten(fig, 0.35, 0.0)
    save(plt, fig, out, "resenc_50ep_paired_vs_control")


def curves_figure(plt, data, out):
    metrics = [(FG, "All organs"), ("esophagus", "Esophagus")]
    fig, axes = plt.subplots(len(FOLDS), len(metrics), figsize=(12, 2.9 * len(FOLDS)), squeeze=False, sharex=True)
    for fold in FOLDS:
        for col, (m, mlabel) in enumerate(metrics):
            ax = axes[fold, col]
            for name, exp, color, marker in RUNS:
                ax.plot(np.arange(1, 51), data[exp]["curve"][fold][m], color=color, lw=1.8, ls="-" if marker == "o" else (0, (4, 2)), label=name)
            ax.set_title(f"Fold {fold} · {mlabel}", loc="left", fontsize=10)
            ax.set_ylabel("2D val Dice")
            ax.set_ylim(0.84 if m == FG else 0.62, None)
            if fold == len(FOLDS) - 1:
                ax.set_xlabel("Epoch")
            clean_axis(ax)
    legend_below(axes[-1, 0], ncol=3)
    decorate(fig, "Validation Dice during training", subtitle=SUBTITLE + " · solid = 2D, dashed = 2.5D",
             footnote_text="2D validation Dice per epoch (256 x 256 slices), not the 3D evaluation. Constant learning rate, so the last epoch is noisy; the 3D numbers use the best epoch.")
    save(plt, fig, out, "resenc_50ep_curves")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO / "figures" / "resenc_50ep_comparison")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    data = {exp: load(exp) for _, exp, _, _ in RUNS}
    plt = pyplot()
    for key in METRICS:
        metric_figure(plt, data, key, args.out)
    paired_figure(plt, data, args.out)
    curves_figure(plt, data, args.out)
    rows = [{"run": name, "experiment": exp, "fold": f, "organ": organ, **{k: value(data, exp, f, organ, k) for k in METRICS}}
            for name, exp, _, _ in RUNS for f in FOLDS for organ in (*ORGANS, FG)]
    write_csv(args.out / "resenc_50ep_comparison.csv", rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
