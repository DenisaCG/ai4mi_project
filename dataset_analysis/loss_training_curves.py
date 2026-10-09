"""Training curves of the post-midterm loss and optimiser changes, sized for a slide.

    python dataset_analysis/loss_training_curves.py [--out figures]

Validation Dice logged after every epoch of training: per patient from counts summed over its slices, on the 256 x 256 model
grid (not the original CT grid of the evaluation). Both panels use seed 0 only, the one seed every variant has for all 4 folds
(4 runs per line, mean over the folds).
Left panel: the ResEnc deep-supervision U-Net with Dice + CE (the reference) and with a boundary loss, a Tversky loss, a
slice-presence head and AdamW. Right panel: ENet with Dice + CE (the reference) and with a constant-weight boundary loss, a
boundary loss whose weight rises over training and a class-weighted CE. Lines are the 3-epoch moving average of the fold mean, from epoch 3. The reference runs are read from metrics/, the variants from the
folders pulled from Snellius (deck/snellius_inventory/loss_runs).
Writes loss_training_curves.{png,pdf} and loss_training_curves.csv (one row per panel, variant and epoch).
"""

import argparse
from pathlib import Path

import numpy as np

from utils import INK, N_FOLDS, REPO, pyplot, read_csv, write_csv
from plot_style import EARTH  # tools/ is on the path once utils is imported

SMOOTH = 3
BRICK_LIGHT = "#D98A74"
# panel title, seeds, then (row label, experiment, folder holding the runs, colour); the first row is the reference
PANELS = [
    (
        "ResEnc U-Net, seed 0 (4 runs)",
        (0,),
        [
            (
                "Dice + CE (reference)",
                "full_cv4_resenc_ds_unet_dice_ce",
                "metrics",
                INK,
            ),
            (
                "+ boundary loss",
                "full_cv4_resenc_ds_unet_dice_ce_boundary",
                "loss_runs",
                EARTH[0],
            ),
            (
                "Tversky loss + CE",
                "full_cv4_resenc_ds_unet_tversky_ce",
                "loss_runs",
                EARTH[1],
            ),
            (
                "+ slice-presence head",
                "full_cv4_resenc_ds_unet_dice_ce_presence",
                "loss_runs",
                EARTH[2],
            ),
            (
                "AdamW optimiser",
                "full_cv4_resenc_ds_unet_dice_ce_adamw",
                "loss_runs",
                EARTH[3],
            ),
        ],
    ),
    (
        "ENet, seed 0 (4 runs)",
        (0,),
        [
            ("Dice + CE (reference)", "full_cv4_enet_dice_ce", "metrics", INK),
            (
                "+ boundary loss",
                "full_cv4_enet_dice_ce_boundary_preprocessing",
                "loss_runs",
                EARTH[0],
            ),
            (
                "Boundary loss, rising",
                "full_cv4_enet_dice_ce_boundary_v2",
                "loss_runs",
                BRICK_LIGHT,
            ),
            (
                "Class-weighted CE",
                "full_cv4_enet_weighted_dice_ce_preprocessing",
                "loss_runs",
                EARTH[2],
            ),
        ],
    ),
]
VARIANTS = [row for _, _, rows in PANELS for row in rows]


def smooth(values: np.ndarray, window: int = SMOOTH) -> np.ndarray:
    """Centred moving average along the last axis; the window shrinks at both ends.

    Args:
        values: Array whose last axis is the epoch.
        window: Odd window length in epochs.

    Returns:
        Array of the same shape as `values`.
    """
    half = window // 2
    n = values.shape[-1]
    return np.stack(
        [values[..., max(0, i - half) : i + half + 1].mean(axis=-1) for i in range(n)],
        axis=-1,
    )


def load_curves(
    root: Path, experiment: str, seeds: tuple[int, ...]
) -> tuple[np.ndarray, list[int]]:
    """Validation Dice per epoch and best epoch of every run that exists.

    Args:
        root: Folder holding `<experiment>_fold<k>/seed<s>/epochs.csv`.
        experiment: Experiment name without the fold suffix.
        seeds: Seeds to read; a missing fold/seed pair is skipped.

    Returns:
        Array of shape (runs, epochs) with `val_dice_fg`, and the 1-based epoch flagged as best in every run.
    """
    curves, best = [], []
    for fold in range(N_FOLDS):
        for seed in seeds:
            path = root / f"{experiment}_fold{fold}" / f"seed{seed}" / "epochs.csv"
            if not path.exists():
                continue
            rows = read_csv(path)
            curves.append([float(r["val_dice_fg"]) for r in rows])
            best.append(
                max(int(r["epoch"]) for r in rows if r["is_best"] == "True") + 1
            )
    return np.array(curves), best


def collect(metrics_dir: Path, runs_dir: Path) -> list[dict]:
    """Curves of every variant of both panels, in figure order."""
    table = []
    for panel, (title, seeds, rows) in enumerate(PANELS):
        for label, experiment, folder, color in rows:
            curves, best = load_curves(
                metrics_dir if folder == "metrics" else runs_dir, experiment, seeds
            )
            table.append(
                {
                    "panel": title,
                    "index": panel,
                    "label": label,
                    "experiment": experiment,
                    "color": color,
                    "curves": curves,
                    "best_epochs": best,
                }
            )
    return table


def draw_panel(ax, rows: list[dict], ylim: tuple[float, float], title: str) -> None:
    """Smoothed run-mean curves with the reference dotted, shaded towards it, and names at the line ends."""
    epochs = np.arange(1, rows[0]["curves"].shape[1] + 1)
    ref = smooth(rows[0]["curves"].mean(axis=0))
    ends = {}
    for row in rows[1:]:
        line = smooth(row["curves"].mean(axis=0))
        ax.plot(
            epochs, line, color=row["color"], lw=2.6, zorder=3, solid_capstyle="round"
        )
        ends[row["label"]] = (line[-1], row["color"])
    ax.plot(
        epochs, ref, color=INK, lw=3, ls=(0, (1, 1.6)), zorder=4, dash_capstyle="round"
    )
    ends[rows[0]["label"]] = (ref[-1], INK)
    spacing = 0.09 * (ylim[1] - ylim[0])
    last = ylim[1]
    for label, (end, color) in sorted(ends.items(), key=lambda kv: -kv[1][0]):
        last = (
            min(end, last - spacing)
            if last < ylim[1]
            else min(end, ylim[1] - spacing / 2)
        )
        ax.text(
            25.6,
            last,
            label,
            fontsize=11,
            va="center",
            color=color,
            fontweight="bold" if color == INK else "normal",
        )
    ax.set_xlim(3, 25)
    ax.set_ylim(*ylim)
    ax.set_xticks(range(3, 26, 2))
    ticks = np.arange(ylim[0], ylim[1] + 1e-9, 0.005)
    ax.set_yticks(ticks, [f"{v:.3f}" for v in ticks])
    ax.grid(color="#DADADA", lw=0.7, ls=(0, (3, 3)))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=6)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metrics-dir", type=Path, default=REPO / "metrics")
    parser.add_argument(
        "--runs-dir", type=Path, default=REPO / "deck/snellius_inventory/loss_runs"
    )
    parser.add_argument("--out", type=Path, default=REPO / "figures")
    args = parser.parse_args(argv)

    table = collect(args.metrics_dir, args.runs_dir)
    plt = pyplot()
    fig = plt.figure(figsize=(11.9, 3.0))
    ax_res = fig.add_axes((0.075, 0.24, 0.285, 0.6))
    ax_enet = fig.add_axes((0.57, 0.24, 0.265, 0.6))
    draw_panel(
        ax_res, [r for r in table if r["index"] == 0], (0.82, 0.87), PANELS[0][0]
    )
    draw_panel(
        ax_enet, [r for r in table if r["index"] == 1], (0.78, 0.835), PANELS[1][0]
    )
    for ax in (ax_res, ax_enet):
        ax.set_xlabel("Epoch", fontsize=12)
    ax_res.set_ylabel("Validation Dice, 256 × 256", fontsize=12)
    fig.text(
        0.5,
        0.995,
        "Training curves of the loss and optimiser variants",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"loss_training_curves.{ext}")

    write_csv(
        args.out / "loss_training_curves.csv",
        [
            {
                "panel": r["panel"],
                "variant": r["label"],
                "experiment": r["experiment"],
                "runs": len(r["curves"]),
                "epoch": int(e),
                "val_dice_mean_of_runs": round(float(v), 4),
            }
            for r in table
            for e, v in zip(range(1, 26), r["curves"].mean(axis=0))
        ],
    )


if __name__ == "__main__":
    main()
