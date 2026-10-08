"""Training curves of the post-midterm loss and optimiser changes, sized for a slide.

    python dataset_analysis/loss_training_curves.py [--out figures]

Validation Dice of the ResEnc deep-supervision U-Net with Dice + CE (the reference) and with a boundary loss, a Tversky loss, a
slice-presence head and AdamW, logged after every epoch of training: per patient from counts summed over its slices, on the
256 x 256 model grid (not the original CT grid of the evaluation), seed 0, mean over the 4 folds. Lines are the 3-epoch moving
average of the fold mean from epoch 3. The reference runs are read from metrics/, the
variants from the folders pulled from Snellius (deck/snellius_inventory/loss_runs).
Writes loss_training_curves.{png,pdf} and loss_training_curves.csv (one row per variant and epoch).
"""

import argparse
import statistics
from pathlib import Path

import numpy as np

from utils import INK, N_FOLDS, REPO, pyplot, read_csv, write_csv
from plot_style import EARTH  # tools/ is on the path once utils is imported

SMOOTH = 3
# (row label, experiment, folder holding the runs, colour)
VARIANTS = [
    ("Dice + CE (reference)", "full_cv4_resenc_ds_unet_dice_ce", "metrics", INK),
    (
        "+ boundary loss",
        "full_cv4_resenc_ds_unet_dice_ce_boundary",
        "loss_runs",
        EARTH[0],
    ),
    ("Tversky loss + CE", "full_cv4_resenc_ds_unet_tversky_ce", "loss_runs", EARTH[1]),
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
]


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


def load_curves(root: Path, experiment: str) -> tuple[np.ndarray, list[int]]:
    """Validation Dice per epoch and best epoch of seed 0 of every fold.

    Args:
        root: Folder holding `<experiment>_fold<k>/seed0/epochs.csv`.
        experiment: Experiment name without the fold suffix.

    Returns:
        Array of shape (folds, epochs) with `val_dice_fg`, and the 1-based epoch flagged as best in every fold.
    """
    curves, best = [], []
    for fold in range(N_FOLDS):
        rows = read_csv(root / f"{experiment}_fold{fold}" / "seed0" / "epochs.csv")
        curves.append([float(r["val_dice_fg"]) for r in rows])
        best.append(max(int(r["epoch"]) for r in rows if r["is_best"] == "True") + 1)
    return np.array(curves), best


def collect(metrics_dir: Path, runs_dir: Path) -> list[dict]:
    """Curves and best-checkpoint summary of every variant, in figure order."""
    table = []
    for label, experiment, folder, color in VARIANTS:
        curves, best = load_curves(
            metrics_dir if folder == "metrics" else runs_dir, experiment
        )
        table.append(
            {
                "label": label,
                "experiment": experiment,
                "color": color,
                "curves": curves,
                "best_epochs": best,
                "best_score": statistics.fmean(curves.max(axis=1)),
                "best_epoch": statistics.fmean(best),
            }
        )
    return table


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
    fig, ax = plt.subplots(
        figsize=(11.9, 3.0),
        gridspec_kw=dict(left=0.075, right=0.795, top=0.88, bottom=0.24),
    )
    epochs = np.arange(1, table[0]["curves"].shape[1] + 1)
    ref = smooth(table[0]["curves"].mean(axis=0))
    ends = {}
    for row in table[1:]:
        line = smooth(row["curves"].mean(axis=0))
        ax.plot(
            epochs, line, color=row["color"], lw=2.8, zorder=3, solid_capstyle="round"
        )
        ends[row["label"]] = (line[-1], row["color"])
    ax.plot(
        epochs,
        ref,
        color=INK,
        lw=3.2,
        ls=(0, (1, 1.6)),
        zorder=4,
        dash_capstyle="round",
    )
    ends[table[0]["label"]] = (ref[-1], INK)
    last = -1.0
    for label, (end, color) in sorted(ends.items(), key=lambda kv: kv[1][0]):
        last = max(end, last + 0.0052)
        ax.text(
            25.6,
            last,
            label,
            fontsize=12,
            va="center",
            color=color,
            fontweight="bold" if color == INK else "normal",
        )
    ax.set_xlim(3, 25)
    ax.set_ylim(0.82, 0.87)
    ax.set_xticks(range(3, 26))
    ax.set_yticks(np.arange(0.82, 0.8701, 0.005))
    ax.set_yticklabels([f"{v:.3f}" for v in np.arange(0.82, 0.8701, 0.005)])
    ax.grid(color="#DADADA", lw=0.7, ls=(0, (3, 3)))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Dice, 256 × 256", fontsize=12)
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
                "variant": r["label"],
                "experiment": r["experiment"],
                "epoch": int(e),
                "val_dice_2d_mean_of_folds": round(float(v), 4),
            }
            for r in table
            for e, v in zip(epochs, r["curves"].mean(axis=0))
        ],
    )


if __name__ == "__main__":
    main()
