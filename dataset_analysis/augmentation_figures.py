#!/usr/bin/env python3
"""
Augmentation selection figures, from the collected 3D validation results (docs/augmentation_results/per_run.csv,
written by tools/collect_augmentation_results.py from metrics/<exp>_fold*/seed*/summary.json).

Every change is paired: an arm is compared with its reference on the same fold and seed (seed-0-only arms with
the seed-0 runs of the reference). Writes into <out-dir>/:
  augmentation_screening.png/.pdf   every single transform on Improved ENet: mean change in foreground Dice,
                                    one dot per fold.
  augmentation_combined.png/.pdf    (a) rotation, scaling and shift alone, their summed gains and the joint
                                    recipe on Improved ENet; (b) the joint recipe's gain on each model.
  augmentation_path.png/.pdf        the selection path: the candidates tried at each step, which one was kept,
                                    for Improved ENet and ResEnc + DS U-Net.
  augmentation_organs.png/.pdf      change in Dice per organ for every multi-transform recipe and model.
  augmentation_dice_hd95.png/.pdf   change in Dice against change in HD95 for every arm on Improved ENet.
  augmentation_figures.csv          every paired change drawn, with its reference and number of runs.

All metrics are 3D (on the original CT grid), Full40 corrected SegTHOR, 4 patient-level folds.

Usage:
    python dataset_analysis/augmentation_figures.py --out-dir figures/augmentation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import CLASSES, COLORS, NAMES  # noqa: E402  (also puts tools/ on sys.path and sets the Agg backend)
from plot_style import EARTH, apply_style, decorate, legend_below, tint  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

PER_RUN = Path("docs/augmentation_results/per_run.csv")
ENET = "full_cv4_enet_dice_ce"
UNET = "full_cv4_resenc_ds_unet_dice_ce"
CHOSEN, REJECTED, JOINT = EARTH[1], "#9A9A9A", EARTH[0]
INK = "#333333"
# (label, experiment suffix, kept in the joint recipe)
SINGLES = [
    ("Rotation", "rotation", True),
    ("Scaling", "scaling", True),
    ("Shift", "shift", True),
    ("Shear", "shear", False),
    ("Gaussian noise", "gaussian_noise", False),
    ("Gaussian blur", "blur", False),
    ("Brightness / contrast", "brightness_contrast", False),
    ("Gamma", "gamma", False),
]
# (row label, control experiment, joint experiment)
MODELS = [
    ("BASE ENet (CE only)", "full_cv4_enet_ce", "full_cv4_enet_ce_augmented"),
    ("Improved ENet", ENET, f"{ENET}_augmented"),
    ("ResEnc + DS U-Net", UNET, f"{UNET}_augmented"),
]
# selection path per model: steps of (candidate label, experiment, compared with, kept)
PATHS = [
    (
        "Improved ENet",
        ENET,
        [
            [("Joint: rotation + scaling + shift, p 0.5", f"{ENET}_augmented", ENET, True)],
            [
                ("Strong geometric: ±15°, 0.8–1.2, p 0.7", f"{ENET}_strong_geometric", f"{ENET}_augmented", True),
                ("Joint + elastic", f"{ENET}_rot_scale_shift_elastic", f"{ENET}_augmented", False),
                ("nnU-Net rates (p 0.2), seed 0", f"{ENET}_nnunet_rot_scale_shift", f"{ENET}_augmented", False),
                ("nnU-Net rates + noise, seed 0", f"{ENET}_nnunet_rot_scale_shift_noise", f"{ENET}_augmented",
                 False),
            ],
            [
                ("Strong geometric, 50 epochs", f"{ENET}_strong_geometric_50ep", f"{ENET}_strong_geometric", True),
                ("Strong + elastic", f"{ENET}_strong_geometric_elastic", f"{ENET}_strong_geometric", False),
                ("Extreme geometric: ±20°, 0.7–1.3, p 0.9", f"{ENET}_extreme_geometric", f"{ENET}_strong_geometric",
                 False),
            ],
        ],
    ),
    (
        "ResEnc + DS U-Net",
        UNET,
        [
            [("Joint: rotation + scaling + shift, p 0.5", f"{UNET}_augmented", UNET, True)],
            [("Strong geometric: ±15°, 0.8–1.2, p 0.7", f"{UNET}_strong_geometric", f"{UNET}_augmented", True)],
        ],
    ),
]
# (row label, experiment, control) for the per-organ heatmap
RECIPES = [
    ("BASE ENet · joint", "full_cv4_enet_ce_augmented", "full_cv4_enet_ce"),
    ("Improved ENet · joint", f"{ENET}_augmented", ENET),
    ("Improved ENet · strong", f"{ENET}_strong_geometric", ENET),
    ("Improved ENet · strong, 50 ep", f"{ENET}_strong_geometric_50ep", ENET),
    ("ResEnc U-Net · joint", f"{UNET}_augmented", UNET),
    ("ResEnc U-Net · strong", f"{UNET}_strong_geometric", UNET),
]
# (label, experiment, kind) for the Dice vs HD95 plot, all against the Improved ENet control
TRADEOFF = [(label, f"{ENET}_{suffix}", "kept" if kept else "single") for label, suffix, kept in SINGLES] + [
    ("Joint", f"{ENET}_augmented", "recipe"),
    ("Joint + elastic", f"{ENET}_rot_scale_shift_elastic", "variant"),
    ("Strong", f"{ENET}_strong_geometric", "recipe"),
    ("Strong + elastic", f"{ENET}_strong_geometric_elastic", "variant"),
    ("Extreme", f"{ENET}_extreme_geometric", "variant"),
    ("Strong, 50 ep", f"{ENET}_strong_geometric_50ep", "recipe"),
]
# label placement (points, alignment) in the Dice vs HD95 plot where the default would collide
LABEL_OFFSETS = {
    "Brightness / contrast": (0, -15, "center"),
    "Gaussian blur": (6, -15, "left"),
    "Shear": (7, 4, "left"),
    "Strong": (-9, -15, "right"),
    "Strong + elastic": (-9, 5, "right"),
    "Extreme": (9, -13, "left"),
}
RECORDS: list[dict] = []  # every paired change drawn, written to augmentation_figures.csv


def paired(runs: pd.DataFrame, exp: str, ref: str, metric: str = "dice_fg") -> pd.Series:
    """Change of one experiment against its reference, paired by fold and seed.

    Args:
        runs: per_run.csv.
        exp: Experiment.
        ref: Reference experiment.
        metric: Column to compare.

    Returns:
        One change per (fold, seed) present in both.
    """
    a = runs[runs.experiment == exp].set_index(["fold", "seed"])[metric]
    b = runs[runs.experiment == ref].set_index(["fold", "seed"])[metric]
    d = (a - b).dropna()
    assert len(d), f"{exp} vs {ref}: no shared fold and seed"
    RECORDS.append({"figure": None, "experiment": exp, "reference": ref, "metric": metric, "n_pairs": len(d),
                    "mean_change": d.mean(), "runs_better": int((d < 0).sum() if metric.startswith(("hd95", "assd"))
                                                                else (d > 0).sum())})
    return d


def tag(figure: str) -> None:
    """Label the RECORDS added since the last call with the figure they belong to."""
    for r in RECORDS:
        r["figure"] = r["figure"] or figure


def save(fig, out_dir: Path, name: str) -> None:
    """Write the figure as png and pdf and close it."""
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    tag(name)
    print(f"wrote {out_dir / name}.png")


def legend(fig, handles: list) -> None:
    """Queue a below-plot legend from explicit handles; decorate() draws it."""
    fig._pending_legend = (handles, [h.get_label() for h in handles], len(handles))


def draw_screening(runs: pd.DataFrame, out_dir: Path) -> None:
    """Bars of mean change per single transform, with the four fold changes as dots.

    Args:
        runs: per_run.csv.
        out_dir: Folder to write the figure into.
    """
    apply_style()
    control = runs[(runs.experiment == ENET) & (runs.seed == 0)].dice_fg.mean()
    rows = []
    for label, suffix, kept in SINGLES:
        d = paired(runs, f"{ENET}_{suffix}", ENET)
        assert len(d) == 4 and set(d.index.get_level_values("seed")) == {0}, f"{suffix}: expected 4 seed-0 folds"
        rows.append((label, kept, d, runs[runs.experiment == f"{ENET}_{suffix}"].dice_fg.mean()))
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.axvline(0, color=INK, lw=1, zorder=2)
    ax.grid(axis="x", color="#E6E6E6")
    ax.grid(axis="y", visible=False)
    columns = ax.get_yaxis_transform()  # x in axes coordinates, y in data
    for i, (label, kept, d, mean_dice) in enumerate(rows):
        color = CHOSEN if kept else REJECTED
        ax.barh(i, d.mean(), height=0.62, color=tint(color, 0.55 if kept else 0.45), edgecolor=color, lw=1.2,
                zorder=3)
        ax.scatter(d, np.full(len(d), i), s=24, color=color, edgecolor="white", lw=0.6, zorder=4)
        better = int((d > 0).sum())
        ax.text(1.1, i, f"{mean_dice:.4f}", transform=columns, va="center", ha="right", fontsize=10.5, color=INK)
        ax.text(1.22, i, f"{better}/4", transform=columns, va="center", ha="right", fontsize=10.5, color=INK,
                fontweight="bold" if better == 4 else "normal")
    ax.text(1.1, -0.6, "mean\nDice", transform=columns, ha="right", fontsize=10, fontweight="bold", color=INK,
            linespacing=0.95)
    ax.text(1.22, -0.6, "folds\nbetter", transform=columns, ha="right", fontsize=10, fontweight="bold", color=INK,
            linespacing=0.95)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.4, -1.0)
    ax.set_xlim(-0.012, 0.028)
    ax.set_xlabel("change in foreground Dice vs. no augmentation (same fold, seed 0)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    legend(fig, [
        Patch(facecolor=tint(CHOSEN, 0.55), edgecolor=CHOSEN, label="kept for the joint recipe"),
        Patch(facecolor=tint(REJECTED, 0.45), edgecolor=REJECTED, label="left out"),
        Line2D([], [], marker="o", ls="", color="#555555", markersize=5, label="one fold"),
    ])
    decorate(
        fig,
        "Only Rotation, Scaling and Shift Improve Every Fold; Intensity Transforms Do Not Help",
        subtitle=(
            "Each transform added alone to Improved ENet (Dice + CE, full preprocessing), 4 folds, seed 0. "
            f"Bar = mean change against the seed-0 control of the same fold (control mean Dice {control:.4f}); "
            "dots = the four folds."
        ),
        footnote_text=(
            "3D foreground Dice (mean over esophagus, heart, trachea, aorta) on the original CT grid; Full40 corrected "
            "SegTHOR, patient-level folds of 30 train / 10 validation patients. Source: docs/augmentation_results/"
            "per_run.csv (from metrics/<experiment>_fold*/seed0/summary.json)."
        ),
    )
    fig.subplots_adjust(right=0.8)
    save(fig, out_dir, "augmentation_screening")


def draw_combined(runs: pd.DataFrame, out_dir: Path) -> None:
    """(a) singles vs their sum vs the joint recipe on Improved ENet; (b) joint gain per model.

    Args:
        runs: per_run.csv.
        out_dir: Folder to write the figure into.
    """
    apply_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 5.8), width_ratios=[1, 1.05])

    singles = {label: paired(runs, f"{ENET}_{suffix}", ENET) for label, suffix, kept in SINGLES if kept}
    joint_seed0 = paired(runs, f"{ENET}_augmented", ENET).xs(0, level="seed")
    bars = [(label, d.mean(), d.to_numpy(), tint(CHOSEN, 0.55), CHOSEN) for label, d in singles.items()]
    bars.append(("Sum of the\nthree alone", sum(d.mean() for d in singles.values()), None, "white", CHOSEN))
    bars.append(("Joint recipe\n(all three)", joint_seed0.mean(), joint_seed0.to_numpy(), tint(JOINT, 0.55), JOINT))
    for i, (label, mean, dots, face, edge) in enumerate(bars):
        left.bar(i, mean, width=0.62, color=face, edgecolor=edge, lw=1.2, hatch="//" if dots is None else None,
                 zorder=3)
        if dots is not None:
            left.scatter(np.full(len(dots), i), dots, s=24, color=edge, edgecolor="white", lw=0.6, zorder=4)
        top = mean if dots is None else max(dots.max(), mean)
        left.text(i, top + 0.001, f"+{mean:.4f}", ha="center", va="bottom", fontsize=10.5,
                  fontweight="bold" if i == len(bars) - 1 else "normal")
    left.set_xticks(range(len(bars)))
    left.set_xticklabels([b[0] for b in bars])
    left.set_ylabel("change in foreground Dice")
    left.set_ylim(0, 0.034)
    left.set_title("(a) Improved ENet, seed 0: the gains do not add up", loc="left", fontsize=13,
                   fontweight="bold")

    labels_right = right.get_yaxis_transform()
    for i, (label, ctrl, joint) in enumerate(MODELS):
        d = paired(runs, joint, ctrl)
        base = runs[runs.experiment == ctrl].dice_fg.mean()
        jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(d))
        right.scatter(d, i + jitter, s=24, color=tint(JOINT, 0.55), edgecolor="white", lw=0.5, zorder=3)
        right.plot(d.mean(), i, "D", color=JOINT, ms=9, mec="white", mew=1, zorder=4)
        right.text(1.03, i, f"+{d.mean():.4f}\n{int((d > 0).sum())}/{len(d)} runs better\nno aug {base:.3f}",
                   transform=labels_right, va="center", ha="left", fontsize=9.5, color=INK, linespacing=1.15)
    right.axvline(0, color=INK, lw=1)
    right.set_yticks(range(len(MODELS)))
    right.set_yticklabels([m[0] for m in MODELS])
    right.set_ylim(len(MODELS) - 0.5, -0.5)
    right.set_xlim(-0.004, 0.056)
    right.grid(axis="x", color="#E6E6E6")
    right.grid(axis="y", visible=False)
    right.tick_params(axis="y", length=0)
    right.spines["left"].set_visible(False)
    right.set_xlabel("change in foreground Dice, joint recipe vs. no augmentation")
    right.set_title("(b) Joint recipe per model, 4 folds × 3 seeds", loc="left", fontsize=13, fontweight="bold")

    legend(fig, [
        Line2D([], [], marker="o", ls="", color=CHOSEN, markersize=5, label="one fold (a)"),
        Line2D([], [], marker="o", ls="", color=tint(JOINT, 0.55), markersize=5, label="one run (b)"),
        Line2D([], [], marker="D", ls="", color=JOINT, markersize=8, label="mean of 12 runs (b)"),
    ])
    decorate(
        fig,
        "Combining Rotation, Scaling and Shift Helps Every Model, but Less Than the Separate Gains Suggest; "
        "Stronger Models Gain Less",
        subtitle=(
            "Joint recipe: rotation ±10°, scaling 0.9–1.1 and shift up to 10%, each with p = 0.5. Every change is "
            "paired against the same fold and seed without augmentation."
        ),
        footnote_text=(
            "3D foreground Dice on the original CT grid; Full40 corrected SegTHOR, 4 patient-level folds. (a) uses "
            "seed 0 only, because the single transforms were screened on seed 0; (b) uses 4 folds × 3 seeds per "
            "model. Source: docs/augmentation_results/per_run.csv."
        ),
    )
    fig.subplots_adjust(right=0.87, wspace=0.42)
    save(fig, out_dir, "augmentation_combined")


def draw_path(runs: pd.DataFrame, out_dir: Path) -> None:
    """Selection path: per step the candidates tried (dot at mean Dice), the kept one joined to the next step.

    Args:
        runs: per_run.csv.
        out_dir: Folder to write the figure into.
    """
    apply_style()
    heights = [1 + sum(len(step) for step in steps) for _, _, steps in PATHS]
    fig, axes = plt.subplots(len(PATHS), 1, figsize=(12.5, 9.5), height_ratios=heights, sharex=True)
    for ax, (model, control, steps) in zip(axes, PATHS):
        rows = [("No augmentation", control, None, True)] + [c for step in steps for c in step]
        y = {exp: i for i, (_, exp, _, _) in enumerate(rows)}
        mean = {exp: runs[runs.experiment == exp].dice_fg.mean() for _, exp, _, _ in rows}
        kept = [exp for _, exp, _, k in rows if k]
        ax.plot([mean[e] for e in kept], [y[e] for e in kept], color=CHOSEN, lw=2, zorder=2)
        boundary = 0.5
        for step in steps:
            boundary += len(step)
            ax.axhline(boundary - len(step), color="#E2E2E2", lw=0.8, zorder=0)
        for label, exp, ref, k in rows:
            n = int((runs.experiment == exp).sum())
            color = CHOSEN if k else REJECTED
            ax.plot(mean[exp], y[exp], "o", ms=10 if k else 8, color=color, mec="white", mew=1.2, zorder=4)
            if ref is None:
                note = f"{mean[exp]:.4f}, {n} runs"
            else:
                d = paired(runs, exp, ref)
                note = f"{mean[exp]:.4f}   {d.mean():+.4f} vs. step before, {int((d > 0).sum())}/{len(d)} better"
            ax.text(mean[exp] + 0.0012, y[exp], note, va="center", fontsize=9.5,
                    color=INK if k else "#7A7A7A", fontweight="bold" if k and ref else "normal")
            if ref is not None and not k:
                ax.plot([mean[ref], mean[exp]], [y[ref], y[exp]], color=REJECTED, lw=1, ls=":", zorder=1)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows])
        for tick, (_, _, _, k) in zip(ax.get_yticklabels(), rows):
            tick.set_color(INK if k else "#7A7A7A")
            tick.set_fontweight("bold" if k else "normal")
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.grid(axis="x", color="#EEEEEE")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_title(model, loc="left", fontsize=13, fontweight="bold")
    axes[-1].set_xlim(0.83, 0.93)
    axes[-1].set_xlabel("mean foreground Dice (4 folds × 3 seeds unless marked seed 0)")
    legend(fig, [
        Line2D([], [], marker="o", color=CHOSEN, lw=2, markersize=8, label="kept: next step builds on it"),
        Line2D([], [], marker="o", color=REJECTED, lw=1, ls=":", markersize=7, label="tried and dropped"),
    ])
    decorate(
        fig,
        "Stronger Geometric Augmentation Kept Paying Off; Elastic Deformation, nnU-Net Rates, Noise and Extreme Settings "
        "Added Little",
        subtitle=(
            "Each step tries candidates on top of the recipe kept at the step before, compared on the same folds and "
            "seeds. Joint + elastic gained +0.003 Dice but raised HD95 by 1.7 mm against the joint recipe. The final best "
            "model is ResEnc + DS U-Net with strong geometric augmentation, trained 50 epochs."
        ),
        footnote_text=(
            "Strong geometric: rotation ±15°, scaling 0.8–1.2, shift 10%, p 0.7. nnU-Net rates: rotation ±10° and "
            "scaling 0.7–1.4 at p 0.2, shift 2%; their changes use the 4 seed-0 runs of the joint recipe. 3D "
            "foreground Dice; Full40 corrected SegTHOR. Source: docs/augmentation_results/per_run.csv; final choice: "
            "docs/final_training.md."
        ),
    )
    save(fig, out_dir, "augmentation_path")


def draw_organs(runs: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap of the paired change in Dice per organ for every multi-transform recipe and model.

    Args:
        runs: per_run.csv.
        out_dir: Folder to write the figure into.
    """
    apply_style()
    columns = [(f"dice_{CLASSES[c]}", NAMES[c], COLORS[c]) for c in CLASSES] + [("dice_fg", "Foreground", INK)]
    values = np.array([[paired(runs, exp, ctrl, col).mean() for col, _, _ in columns] for _, exp, ctrl in RECIPES])
    better = [[int((paired(runs, exp, ctrl, col) > 0).sum()) for col, _, _ in columns] for _, exp, ctrl in RECIPES]
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("gain", ["#FFFFFF", tint(CHOSEN, 0.5), CHOSEN])
    vmax = np.abs(values).max()
    ax.imshow(np.clip(values, 0, None), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(len(RECIPES)):
        for j in range(len(columns)):
            v = values[i, j]
            ax.text(j, i, f"{v:+.3f}\n{better[i][j]}/12", ha="center", va="center", fontsize=10,
                    color="white" if v > 0.6 * vmax else INK, fontweight="bold" if j == len(columns) - 1 else "normal")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([c[1] for c in columns], fontweight="bold")
    for tick, (_, _, color) in zip(ax.get_xticklabels(), columns):
        tick.set_color(color)
    ax.set_yticks(range(len(RECIPES)))
    ax.set_yticklabels([r[0] for r in RECIPES])
    ax.tick_params(length=0)
    ax.grid(False)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.axvline(len(columns) - 1.5, color="white", lw=4)
    for b in (0.5, 3.5):
        ax.axhline(b, color="white", lw=4)
    decorate(
        fig,
        "Augmentation Helps the Esophagus Most and the Heart Least, on Every Model",
        subtitle=(
            "Change in 3D Dice per organ against the same model without augmentation, paired by fold and seed; "
            "mean of 12 runs, with how many of the 12 improved."
        ),
        footnote_text=(
            "Joint: rotation ±10°, scaling 0.9–1.1, shift 10%, p 0.5. Strong: rotation ±15°, scaling 0.8–1.2, shift "
            "10%, p 0.7. Darker = larger gain; a decrease would be white. Full40 corrected SegTHOR, 4 folds × 3 seeds. "
            "Source: docs/augmentation_results/per_run.csv."
        ),
    )
    save(fig, out_dir, "augmentation_organs")


def draw_dice_hd95(runs: pd.DataFrame, out_dir: Path) -> None:
    """Change in Dice against change in HD95 (as reduction) for every arm on Improved ENet, ±1 standard error.

    Args:
        runs: per_run.csv.
        out_dir: Folder to write the figure into.
    """
    apply_style()
    style = {"single": (REJECTED, "o"), "kept": (CHOSEN, "o"), "recipe": (JOINT, "D"), "variant": (REJECTED, "D")}
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axhline(0, color=INK, lw=1, zorder=1)
    ax.axvline(0, color=INK, lw=1, zorder=1)
    ax.grid(axis="x", color="#E6E6E6")
    for label, exp, kind in TRADEOFF:
        dice = paired(runs, exp, ENET)
        hd = -paired(runs, exp, ENET, "hd95_fg")  # positive = HD95 went down = better
        color, marker = style[kind]
        se = lambda s: s.std(ddof=1) / np.sqrt(len(s))  # noqa: E731
        ax.errorbar(dice.mean(), hd.mean(), xerr=se(dice), yerr=se(hd), fmt=marker, color=color,
                    ms=9 if kind in ("recipe", "kept") else 7, mec="white", mew=1, elinewidth=1, capsize=0,
                    ecolor=tint(color, 0.5), zorder=3)
        dx, dy, ha = LABEL_OFFSETS.get(label, (7, 5, "left"))
        ax.annotate(label, (dice.mean(), hd.mean()), xytext=(dx, dy), textcoords="offset points", ha=ha,
                    fontsize=9.5, color=INK if kind in ("recipe", "kept") else "#6E6E6E",
                    fontweight="bold" if kind == "recipe" else "normal")
    ax.set_xlabel("change in foreground Dice  (right = better)")
    ax.set_ylabel("reduction in foreground HD95, mm  (up = better)")
    ax.text(0.99, 0.98, "better on both", transform=ax.transAxes, ha="right", va="top", fontsize=10,
            color=CHOSEN, style="italic")
    legend(fig, [
        Line2D([], [], marker="o", ls="", color=CHOSEN, markersize=8, label="single transform, kept"),
        Line2D([], [], marker="o", ls="", color=REJECTED, markersize=7, label="single transform, dropped"),
        Line2D([], [], marker="D", ls="", color=JOINT, markersize=8, label="recipe kept"),
        Line2D([], [], marker="D", ls="", color=REJECTED, markersize=7, label="recipe variant dropped"),
    ])
    decorate(
        fig,
        "Dice and HD95 Mostly Improve Together; Rotation Alone Worsens HD95, and Elastic Erases the Joint Recipe's HD95 Gain",
        subtitle=(
            "Every arm on Improved ENet against no augmentation, paired by fold and seed: single transforms 4 runs "
            "(seed 0), recipes 12 runs. Bars = ±1 standard error of the paired change."
        ),
        footnote_text=(
            "HD95 is the 95th-percentile surface distance in mm, averaged over the four organs; a reduction is an "
            "improvement. HD95 is outlier-prone, which is why it was not used to choose transforms. Source: "
            "docs/augmentation_results/per_run.csv."
        ),
    )
    save(fig, out_dir, "augmentation_dice_hd95")


def main():
    """Read the collected results, draw every figure and write the table of drawn changes."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-run", type=Path, default=PER_RUN)
    ap.add_argument("--out-dir", type=Path, default=Path("figures/augmentation"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(args.per_run)
    print(f"{len(runs)} runs from {args.per_run}, {runs.experiment.nunique()} experiments")
    draw_screening(runs, args.out_dir)
    draw_combined(runs, args.out_dir)
    draw_path(runs, args.out_dir)
    draw_organs(runs, args.out_dir)
    draw_dice_hd95(runs, args.out_dir)
    table = pd.DataFrame(RECORDS).drop_duplicates(subset=["figure", "experiment", "reference", "metric"])
    table.to_csv(args.out_dir / "augmentation_figures.csv", index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
