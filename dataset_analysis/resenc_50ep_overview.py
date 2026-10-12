"""Mean Dice, HD95 and ASSD of every 50-epoch CV model, one row per model, in the look of architecture_options.py (box option).

    python dataset_analysis/resenc_50ep_overview.py [--out figures/resenc_50ep_comparison]

Per model the runs are all (fold, seed) pairs found in runs/<experiment>_fold<k>/seed<s>/eval/metrics_3d.csv: 12 runs (4 folds x 3 seeds) for
most models, 4 for the newest one (seed 0 only). Per run: the mean over the 10 validation patients of each organ, then the mean over the four
organs, 3D on the original CT grid at the best (val-selected) checkpoint. Box = run means, points shaped by fold, star = mean, dashed line = the
ResEnc DS control. Writes resenc_50ep_overview.{png,pdf} and a CSV of the run means.
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from utils import CLASSES, INK, REPO, pyplot, tint, write_csv

ORGANS = list(CLASSES.values())
BASE = "full_cv4_resenc_ds_unet_dice_ce_"
# (row label, experiment, colour); one hue per family: ENet blues, ResEnc greens, ResEnc + boundary warm
MODELS = [
    ("ENet · CE, no preprocessing", "full_cv4_enet_ce_50ep", "#5B8FC4"),
    ("ENet · Dice+CE, preprocessed", "full_cv4_enet_dice_ce_50ep", "#2F5D8C"),
    ("ResEnc DS + strong geom.\ncontrol", BASE + "strong_geometric_50ep", "#1F5C4F"),
    ("+ ramped boundary", BASE + "strong_geometric_50ep_boundary", "#A8452A"),
    ("+ fixed boundary 0.08", BASE + "strong_geometric_50ep_boundary_w008", "#C8702A"),
    ("+ 2.5D ±1, ramped boundary", BASE + "boundary_25d_c1_strong_geometric_50ep", "#D2574B"),
    ("+ 2.5D ±1, fixed boundary 0.08", BASE + "25d_c1_strong_geometric_50ep_boundary_w008", "#E0A030"),
]
KEYS = [e for _, e, _ in MODELS]
COLOR_OF = {e: c for _, e, c in MODELS}
FAMILY_STARTS = {KEYS[2], KEYS[3]}
CONTROL = KEYS[2]
FOLD_MARKERS = "osD^"
PANELS = [  # column, title, axis label, digits, (lo, hi)
    ("dice", "Dice, mean of\nfour organs", "Dice (3D)", 3, (0.68, 0.97)),
    ("hd95", "HD95 (mm), mean of\nfour organs", "HD95 (mm, lower is better)", 1, (4, 36)),
    ("assd", "ASSD (mm), mean of\nfour organs", "ASSD (mm, lower is better)", 2, (1, 5.4)),
]


def load_runs(runs_dir: Path) -> pd.DataFrame:
    """One row per model and (fold, seed) run: the three metrics, each averaged over patients and then over the four organs."""
    rows = []
    for _, experiment, _ in MODELS:
        for path in sorted(runs_dir.glob(f"{experiment}_fold[0-9]/seed*/eval/metrics_3d.csv")):
            m = re.search(r"_fold(\d)/seed(\d+)/", str(path))
            df = pd.read_csv(path)
            df = df[df["class_name"].isin(ORGANS)]
            per_organ = df.groupby("class_name")[["dice", "hd95", "assd"]].mean()
            rows.append({"arch": experiment, "fold": int(m.group(1)), "seed": int(m.group(2)), **per_organ.mean().to_dict()})
    return pd.DataFrame(rows)


def draw(plt, runs: pd.DataFrame):
    n = len(MODELS)
    names = []
    for label, e, _ in MODELS:
        first, _, tag = label.partition("\n")
        names.append(f"{first}\n({tag + ', ' if tag else ''}{(runs['arch'] == e).sum()} runs)")
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 6.4), gridspec_kw={"wspace": 0.06, "width_ratios": [1, 1, 1]})
    means = runs.groupby("arch")[["dice", "hd95", "assd"]].mean()
    for j, (ax, (col, title, xlabel, digits, (lo, hi))) in enumerate(zip(axs, PANELS)):
        ax.set_ylim(n - 0.5, -0.5)
        for i, key in enumerate(KEYS):
            ax.axhspan(i - 0.5, i + 0.5, color=tint(COLOR_OF[key], 0.12), lw=0, zorder=0)
            if key in FAMILY_STARTS:
                ax.axhline(i - 0.5, color="#555555", lw=1.1, zorder=1)
        sns.boxplot(data=runs, x=col, y="arch", hue="arch", order=KEYS, hue_order=KEYS, palette=COLOR_OF, orient="h", ax=ax, width=0.6,
                    fliersize=0, linewidth=1.2, legend=False, zorder=2, boxprops={"alpha": 0.55}, medianprops={"color": INK, "lw": 2})
        for fold, marker in enumerate(FOLD_MARKERS):
            sns.stripplot(data=runs[runs["fold"] == fold], x=col, y="arch", hue="arch", order=KEYS, hue_order=KEYS, palette=COLOR_OF, orient="h",
                          ax=ax, size=5, jitter=0.14, marker=marker, edgecolor="white", linewidth=0.5, legend=False, zorder=4)
        ax.axvline(means.loc[CONTROL, col], color="#555555", lw=1.2, ls="--", zorder=1)
        for i, key in enumerate(KEYS):
            ax.scatter(means.loc[key, col], i, marker="*", s=170, color="white", edgecolor=INK, lw=1.0, zorder=6)
            ax.text(hi - (hi - lo) * 0.015, i - 0.32, f"{means.loc[key, col]:.{digits}f}", ha="right", va="center", fontsize=10,
                    fontweight="bold", color=COLOR_OF[key], zorder=7)
        ax.set_xlim(lo, hi)
        ax.grid(axis="x", which="major", color="#C8C8C8", lw=0.8)
        ax.grid(axis="y", visible=False)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(title, fontsize=12, fontweight="bold")
        if j == 0:
            ax.set_yticks(range(n), names, fontsize=10.5)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.text(0.5, -0.1, xlabel, transform=ax.transAxes, ha="center", va="top", fontsize=11)
    handles = [plt.Line2D([], [], marker=m, color="#666666", ls="", markersize=7, label=f"fold {k}") for k, m in enumerate(FOLD_MARKERS)]
    handles += [plt.Line2D([], [], marker="*", color="white", mec=INK, ls="", markersize=12, label="mean"),
                plt.Line2D([], [], color="#555555", ls="--", lw=1.2, label="control mean")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=11, frameon=False, handletextpad=0.3, columnspacing=1.0)
    fig.subplots_adjust(left=0.215, right=0.985, top=0.83, bottom=0.2)
    fig.text(0.012, 0.52, "50-epoch model", rotation=90, ha="left", va="center", fontsize=12)
    fig.text(0.985, 0.99, "Run means of the 50-epoch models: Dice, HD95 and ASSD", ha="right", va="top", fontsize=14, fontweight="bold")
    fig.text(0.985, 0.935, "One point per run (fold x seed); 3D on the original CT grid at the best checkpoint; ResEnc rows use strong geometric augmentation", ha="right", va="top",
             fontsize=10, color="#555555")
    return fig


def draw_columns(plt, runs: pd.DataFrame):
    """A wide Dice box panel (zoomed to the models) with the mean HD95 and ASSD of every model as text columns on the right."""
    keys, models = KEYS[2:], MODELS[2:]  # the ENet rows are far left and would compress the ResEnc spread; see resenc_50ep_overview
    runs = runs[runs["arch"].isin(keys)]
    n = len(models)
    names = []
    for label, e, _ in models:
        first, _, tag = label.partition("\n")
        names.append(f"{first}\n({tag + ', ' if tag else ''}{(runs['arch'] == e).sum()} runs)")
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 5.6), gridspec_kw={"wspace": 0.08, "width_ratios": [4.2, 0.8, 0.8]})
    means = runs.groupby("arch")[["dice", "hd95", "assd"]].mean()
    ax, lo, hi = axs[0], 0.862, 0.925
    ax.set_ylim(n - 0.5, -0.5)
    for i, key in enumerate(keys):
        ax.axhspan(i - 0.5, i + 0.5, color=tint(COLOR_OF[key], 0.12), lw=0, zorder=0)
        if key in FAMILY_STARTS:
            ax.axhline(i - 0.5, color="#555555", lw=1.1, zorder=1)
    sns.boxplot(data=runs, x="dice", y="arch", hue="arch", order=keys, hue_order=keys, palette=COLOR_OF, orient="h", ax=ax, width=0.6,
                fliersize=0, linewidth=1.2, legend=False, zorder=2, boxprops={"alpha": 0.55}, medianprops={"color": INK, "lw": 2})
    for fold, marker in enumerate(FOLD_MARKERS):
        sns.stripplot(data=runs[runs["fold"] == fold], x="dice", y="arch", hue="arch", order=keys, hue_order=keys, palette=COLOR_OF, orient="h",
                      ax=ax, size=6, jitter=0.14, marker=marker, edgecolor="white", linewidth=0.5, legend=False, zorder=4)
    ax.axvline(means.loc[CONTROL, "dice"], color="#555555", lw=1.2, ls="--", zorder=1)
    for i, key in enumerate(keys):
        ax.scatter(means.loc[key, "dice"], i, marker="*", s=190, color="white", edgecolor=INK, lw=1.0, zorder=6)
        ax.text(hi - (hi - lo) * 0.01, i, f"{means.loc[key, 'dice']:.3f}", ha="right", va="center", fontsize=11, fontweight="bold",
                color=COLOR_OF[key], zorder=7)
    ax.set_xlim(lo, hi)
    ax.set_xticks(np.round(np.arange(0.87, 0.921, 0.01), 2))
    ax.set_xticks(np.round(np.arange(0.865, 0.925, 0.005), 2), minor=True)
    ax.grid(axis="x", which="major", color="#C8C8C8", lw=0.8)
    ax.grid(axis="x", which="minor", color="#E4E4E4", lw=0.6, ls=(0, (2, 2)))
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="x", which="minor", length=2)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Dice, mean of four organs", fontsize=12, fontweight="bold")
    ax.set_yticks(range(n), names, fontsize=10.5)
    ax.text(0.5, -0.1, "Dice (3D, original CT grid), one point per run", transform=ax.transAxes, ha="center", va="top", fontsize=11)
    for other, (col, title, digits) in zip(axs[1:], (("hd95", "HD95\n(mm)", 1), ("assd", "ASSD\n(mm)", 2))):
        other.set_xlim(0, 1)
        other.set_ylim(n - 0.5, -0.5)
        for i, key in enumerate(keys):
            other.text(0.5, i, f"{means.loc[key, col]:.{digits}f}", ha="center", va="center", fontsize=11, fontweight="bold", color=COLOR_OF[key])
        other.set_title(title, fontsize=12, fontweight="bold")
        other.axis("off")
    handles = [plt.Line2D([], [], marker=m, color="#666666", ls="", markersize=7, label=f"fold {k}") for k, m in enumerate(FOLD_MARKERS)]
    handles += [plt.Line2D([], [], marker="*", color="white", mec=INK, ls="", markersize=12, label="mean"),
                plt.Line2D([], [], color="#555555", ls="--", lw=1.2, label="control mean")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=11, frameon=False, handletextpad=0.3, columnspacing=1.0)
    fig.subplots_adjust(left=0.25, right=0.985, top=0.76, bottom=0.2)
    fig.text(0.012, 0.52, "50-epoch model", rotation=90, ha="left", va="center", fontsize=12)
    fig.text(0.985, 0.99, "Run means of the ResEnc 50-epoch models", ha="right", va="top", fontsize=14, fontweight="bold")
    fig.text(0.985, 0.935, "HD95 and ASSD: mean over runs and the four organs (lower is better); 3D at the best checkpoint", ha="right", va="top",
             fontsize=10, color="#555555")
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=REPO / "runs")
    parser.add_argument("--out", type=Path, default=REPO / "figures" / "resenc_50ep_comparison")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.runs)
    plt = pyplot()
    fig = draw(plt, runs)
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"resenc_50ep_overview.{ext}")
    fig = draw_columns(plt, runs)
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"resenc_50ep_overview_columns.{ext}")
    write_csv(args.out / "resenc_50ep_overview.csv", [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()} for r in runs.to_dict("records")])
    print(runs.groupby("arch")[["dice", "hd95", "assd"]].agg(["mean", "count"]).round(3).loc[KEYS].to_string())


if __name__ == "__main__":
    main()
