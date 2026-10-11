"""Two seaborn-style figure options for the architecture slide, sized for the 7.9 x 4.85 in slot.

    python dataset_analysis/architecture_options.py [--out figures]

All options use the same runs as architecture_overview.py (4 folds x 3 seeds, 3D Dice on the original CT grid). Every
architecture has its own colour (one hue per family: ENet blues, U-Net block oranges and reds, ResEnc greens; the frozen-encoder U-Net is in the U-Net block), its own
tinted row, and a line separates the families.
box:     per architecture the 12 run means (mean over the fold's 10 validation patients) as a box plot with every run as a
         point whose shape is its fold, for the mean over the four organs; the columns on the right give the esophagus Dice
         and the HD95, each averaged over the runs.
paired:  per patient (averaged over seeds), Dice of the architecture minus Dice of the upgraded baseline, as a box plot with
         the patients as points whose shape is their fold; the dashed line is no difference. The columns on the right give
         the median difference and the number of the 40 patients with a higher Dice than the baseline.
Writes architecture_option_<name>.png.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from architecture_overview import ARCHITECTURES, ESOPHAGUS
from utils import (
    CLASSES,
    INK,
    REPO,
    load_arch_runs,
    pyplot,
    tint,
)

ORGANS = list(CLASSES.values())
KEYS = [experiment for _, experiment in ARCHITECTURES]
DISPLAY = [
    label for label, _ in ARCHITECTURES
]  # not unique, so rows are keyed by experiment
SIZE = (7.9, 4.85)

# One colour per architecture, one hue per family (ENet blues, U-Net block oranges and reds, ResEnc greens; the frozen-encoder U-Net is in the U-Net block).
COLORS_BY_EXPERIMENT = {
    "full_cv4_enet_dice_ce": "#2F5D8C",
    "full_cv4_enet_dice_ce_25d_c1": "#5B8FC4",
    "full_cv4_enet_dice_ce_25d_c2": "#8FB4D8",
    "full_cv4_unet_dice_ce": "#C8702A",
    "full_cv4_attention_unet_dice_ce": "#E0A030",
    "full_cv4_ds_unet_dice_ce": "#A8452A",
    "full_cv4_resenc_unet_dice_ce": "#3E8F6E",
    "full_cv4_resenc_ds_unet_dice_ce": "#1F5C4F",
    "full_cv4_resenc_ds_unet_dice_ce_25d_c1": "#7DBE9A",
    "full_cv4_dino_unet_dice_ce": "#D2574B",
}
COLOR_OF = COLORS_BY_EXPERIMENT
FAMILY_STARTS = {KEYS[i] for i in (3, 7)}
FOLD_MARKERS = "osD^"


def load_frame(metrics_dir: Path) -> pd.DataFrame:
    """Validation metrics of every run, patient and organ, with the figure label of its architecture."""
    frames = []
    for _, experiment in ARCHITECTURES:
        rows, _ = load_arch_runs(metrics_dir, experiment)
        frames.append(pd.DataFrame(rows).assign(arch=experiment))
    return pd.concat(frames, ignore_index=True)


def run_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per architecture and run: Dice over the four organs and of the esophagus."""
    per_organ = df.groupby(["arch", "fold", "seed", "organ"], as_index=False)[
        "dice"
    ].mean()
    wide = per_organ.pivot_table(
        index=["arch", "fold", "seed"], columns="organ", values="dice"
    )
    wide["mean"] = wide[ORGANS].mean(axis=1)
    return wide.reset_index()


def patient_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per architecture and patient (Dice averaged over the seeds): four-organ mean and esophagus."""
    per_organ = df.groupby(["arch", "fold", "patient", "organ"], as_index=False)[
        "dice"
    ].mean()
    wide = per_organ.pivot_table(
        index=["arch", "fold", "patient"], columns="organ", values="dice"
    )
    wide["mean"] = wide[ORGANS].mean(axis=1)
    return wide.reset_index()


def row_decor(ax, order: list[str]) -> None:
    """Tint every row in its architecture's colour and draw a line where a new family starts."""
    for i, label in enumerate(order):
        ax.axhspan(i - 0.5, i + 0.5, color=tint(COLOR_OF[label], 0.12), lw=0, zorder=0)
        if label in FAMILY_STARTS and i > 0:
            ax.axhline(i - 0.5, color="#555555", lw=1.1, zorder=1)


def style(
    ax,
    lo: float,
    hi: float,
    step: float,
    title: str,
    names: list[str] | None,
    color: str = INK,
) -> None:
    """Shared look: limits, dashed grid at the ticks, half-step minor lines, row names (given only for the first panel)."""
    ax.set_xlim(lo, hi)
    ticks = np.round(np.arange(lo, hi + 1e-9, step), 3)
    ax.set_xticks(ticks[1:-1])
    ax.set_xticks(np.round(np.arange(lo, hi + 1e-9, step / 2), 3), minor=True)
    ax.grid(axis="x", which="major", color="#C8C8C8", lw=0.8)
    ax.grid(axis="x", which="minor", color="#E4E4E4", lw=0.6, ls=(0, (2, 2)))
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="x", which="minor", length=2)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=12, fontweight="bold", color=color)
    if names is None:
        ax.tick_params(axis="y", labelleft=False)
    else:
        ax.set_yticks(range(len(names)), names, fontsize=11)


def text_column(ax, order: list[str], texts: list[str], title: str) -> None:
    """A column of one text per row, in the colour of its architecture."""
    ax.set_xlim(0, 1)
    ax.set_ylim(len(order) - 0.5, -0.5)
    for i, (label, text) in enumerate(zip(order, texts)):
        ax.text(
            0.5,
            i,
            text,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=COLOR_OF[label],
        )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")


def finish(
    fig,
    axs,
    title: str,
    xlabel: str,
    handles=None,
    left: float = 0.27,
    right: float = 0.985,
) -> None:
    """Title right-aligned to the last panel, rotated row label, shared x label and legend."""
    bottom = 0.2
    fig.subplots_adjust(left=left, right=right, top=0.81, bottom=bottom)
    fig.supxlabel(xlabel, fontsize=12, y=bottom - 0.1, x=0.5 * (left + right))
    fig.text(
        0.012, 0.52, "Architecture", rotation=90, ha="left", va="center", fontsize=12
    )
    fig.text(
        axs[-1].get_position().x1,
        0.99,
        title,
        ha="right",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=11, frameon=False,
                   handletextpad=0.3, columnspacing=1.0)  # fmt: skip


def fold_handles(plt) -> list:
    """Legend entries for the four fold marker shapes."""
    return [
        plt.Line2D(
            [], [], marker=m, color="#666666", ls="", markersize=7, label=f"fold {k}"
        )
        for k, m in enumerate(FOLD_MARKERS)
    ]


def box_panel(
    ax, data: pd.DataFrame, value: str, order: list[str], size: float, jitter: float
) -> None:
    """Horizontal box plot of one value per architecture, each in its colour, with the points shaped by fold."""
    sns.boxplot(data=data, x=value, y="arch", hue="arch", order=order, hue_order=order, palette=COLOR_OF, orient="h",
                ax=ax, width=0.6, fliersize=0, linewidth=1.2, legend=False, zorder=2,
                boxprops={"alpha": 0.55}, medianprops={"color": INK, "lw": 2})  # fmt: skip
    for fold, marker in enumerate(FOLD_MARKERS):
        sns.stripplot(data=data[data["fold"] == fold], x=value, y="arch", hue="arch", order=order, hue_order=order,
                      palette=COLOR_OF, orient="h", ax=ax, size=size, jitter=jitter, marker=marker, edgecolor="white",
                      linewidth=0.5, legend=False, zorder=4)  # fmt: skip


def option_box(df: pd.DataFrame, plt):
    """Box plot of the 12 run means (mean of the four organs) per architecture."""
    runs = run_table(df)
    dist = df.groupby(["arch", "fold", "seed"])["hd95"].mean().groupby("arch").mean()
    eso = runs.groupby("arch")["esophagus"].mean()
    fig, axs = plt.subplots(
        1,
        3,
        figsize=SIZE,
        gridspec_kw={"wspace": 0.08, "width_ratios": [3.6, 0.9, 0.9]},
    )
    ax = axs[0]
    ax.set_ylim(len(KEYS) - 0.5, -0.5)
    row_decor(ax, KEYS)
    box_panel(ax, runs, "mean", KEYS, 5, 0.14)
    style(ax, 0.76, 0.92, 0.04, "Dice, mean of four organs", DISPLAY)
    for other in axs[1:]:
        other.set_ylim(len(KEYS) - 0.5, -0.5)
    text_column(axs[1], KEYS, [f"{eso[a]:.3f}" for a in KEYS], "Esophagus\nDice")
    text_column(axs[2], KEYS, [f"{dist[a]:.1f}" for a in KEYS], "HD95\n(mm)")
    finish(
        fig,
        axs,
        "Run means by architecture",
        "Dice (3D, original CT grid), one point per run",
        fold_handles(plt),
        left=0.3,
    )
    return fig


def option_paired(df: pd.DataFrame, plt):
    """Per-patient difference to the upgraded baseline."""
    pats = patient_table(df)
    base = pats[pats["arch"] == KEYS[0]].set_index("patient")
    diff = pats.copy()
    for col in ("mean", "esophagus"):
        diff[col] = pats[col].to_numpy() - base.loc[pats["patient"], col].to_numpy()
    diff = diff[diff["arch"] != KEYS[0]]
    order = KEYS[1:]
    fig, axs = plt.subplots(
        1,
        4,
        figsize=SIZE,
        gridspec_kw={"wspace": 0.08, "width_ratios": [2.3, 2.3, 0.8, 0.8]},
    )
    for ax, organ, title, (lo, hi), step in zip(
        axs[:2], ("mean", "esophagus"), ("Dice difference,\nmean of four organs", "Dice difference,\nesophagus"), ((-0.2, 0.2), (-0.3, 0.3)), (0.1, 0.1),
    ):  # fmt: skip
        ax.set_ylim(len(order) - 0.5, -0.5)
        row_decor(ax, order)
        box_panel(ax, diff, organ, order, 3.8, 0.18)
        ax.axvline(0, color=INK, lw=1.3, ls="--", zorder=3)
        style(
            ax,
            lo,
            hi,
            step,
            title,
            DISPLAY[1:] if organ == "mean" else None,
            INK if organ == "mean" else ESOPHAGUS,
        )
    median = diff.groupby("arch")["mean"].median()
    better = diff.groupby("arch")["mean"].apply(lambda v: int((v > 0).sum()))
    for ax in axs[2:]:
        ax.set_ylim(len(order) - 0.5, -0.5)
    text_column(axs[2], order, [f"{median[a]:+.3f}" for a in order], "Median\nΔ")
    text_column(axs[3], order, [f"{better[a]}" for a in order], "Better\nof 40")
    finish(
        fig,
        axs,
        "Per-patient change from the upgraded baseline",
        "Dice minus baseline Dice, one point per patient",
        fold_handles(plt),
    )
    return fig


OPTIONS = {"box": option_box, "paired": option_paired}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metrics-dir", type=Path, default=REPO / "metrics")
    parser.add_argument("--out", type=Path, default=REPO / "figures")
    args = parser.parse_args(argv)
    np.random.seed(0)  # seaborn jitters the points with the global generator
    plt = pyplot()
    df = load_frame(args.metrics_dir)
    for name, build in OPTIONS.items():
        fig = build(df, plt)
        fig.savefig(args.out / f"architecture_option_{name}.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
