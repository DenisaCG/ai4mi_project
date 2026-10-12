"""Paired per-patient change of the upgraded ENet's loss variants against its Dice + CE reference, as split violins.

    python dataset_analysis/loss_violins.py [--out figures]

Top row: Dice gain, bottom row: HD95 reduction (so up always means better), for every validation patient (40) and organ of the
four folds, seed 0 only (the one seed every variant has for all folds). Each variant is compared with the upgraded ENet
(Dice + CE, full pre-processing) on the same patients: a class-weighted CE, a Tversky loss in place of the Dice term, a
distance-weighted error term, and a boundary loss with weight 0.08, with weight 1.0 and with a weight that rises over training.
A violin is the density of the 40 changes of one organ, blue above zero (the variant is better on those patients) and red
below; the box is the quartiles, the diamond the mean, and the number under the organ name is how many of the 40 patients
improved. HD95 reductions beyond +-30 mm are drawn as triangles on the edge. The reference runs are read from metrics/, the
variants from the folders pulled from Snellius (~/snellius_inventory/loss_runs). Drawn at the size of its slide slot.
Writes loss_violins.{png,pdf} and loss_violins.csv (one row per variant, organ and metric).
"""

import argparse
from pathlib import Path

import numpy as np

from utils import INPUTS, CLASSES, INK, N_FOLDS, REPO, pyplot, read_csv, write_csv

SLOT_WIDTH, SLOT_HEIGHT = (
    11.4,
    3.95,
)  # inches, the figure slot of the slide (no rescaling, so 11 pt stays 11 pt)
BLUE, RED = "#3B7FB8", "#C8473F"
ORGANS = list(CLASSES.values())
ABBREV = {"esophagus": "Es", "heart": "He", "trachea": "Tr", "aorta": "Ao"}
# metric -> (sign that makes a positive value better, axis label, plotted limit)
METRICS = {
    "dice": (1, "Dice gain", 0.12),
    "hd95": (-1, "HD95 reduction (mm)", 30.0),
}
# (model, reference experiment, folder), then (row label, experiment) of the variants
MODELS = [
    (
        "ENet",
        ("full_cv4_enet_dice_ce", "metrics"),
        [
            ("Class-weighted CE", "full_cv4_enet_weighted_dice_ce_preprocessing"),
            ("Tversky + CE", "full_cv4_enet_tversky_ce"),
            ("Distance-weighted", "full_cv4_enet_dice_ce_hd"),
            ("Boundary 0.08", "full_cv4_enet_dice_ce_boundary_preprocessing"),
            ("Boundary 1.0", "full_cv4_enet_dice_ce_boundary_w1"),
            ("Boundary, rising", "full_cv4_enet_dice_ce_boundary_v2"),
        ],
    ),
]


def load_patient_metrics(
    root: Path, experiment: str
) -> dict[tuple[str, str], dict[str, float]]:
    """3D metrics of every validation patient and organ of seed 0, over the folds.

    Args:
        root: Folder holding `<experiment>_fold<k>/seed0/metrics_3d.csv`.
        experiment: Experiment name without the fold suffix.

    Returns:
        Dice and HD95 keyed by (patient, organ).
    """
    out = {}
    for fold in range(N_FOLDS):
        for r in read_csv(
            root / f"{experiment}_fold{fold}" / "seed0" / "metrics_3d.csv"
        ):
            if r["split"] == "val":
                out[r["patient"], r["class_name"]] = {k: float(r[k]) for k in METRICS}
    return out


def gains(reference: dict, variant: dict, organ: str, metric: str) -> np.ndarray:
    """Per-patient change of an organ, positive where the variant is better than the reference."""
    sign = METRICS[metric][0]
    return np.array(
        [
            (variant[k][metric] - reference[k][metric]) * sign
            for k in sorted(reference)
            if k[1] == organ
        ]
    )


def violin_profile(
    values: np.ndarray, points: int = 120
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian kernel density over the range of the values, scaled to a maximum of 1.

    Args:
        values: Sample, at least two distinct values.
        points: Number of grid points.

    Returns:
        The grid and the density on it.
    """
    spread = min(
        values.std(ddof=1),
        (np.percentile(values, 75) - np.percentile(values, 25)) / 1.34,
    )
    bandwidth = 0.9 * max(spread, 1e-9) * len(values) ** -0.2
    grid = np.linspace(values.min(), values.max(), points)
    dens = np.exp(-0.5 * ((grid[:, None] - values[None, :]) / bandwidth) ** 2).sum(
        axis=1
    )
    return grid, dens / dens.max()


def collect(metrics_dir: Path, runs_dir: Path) -> list[dict]:
    """Per-patient changes of every variant, organ and metric, in figure order."""
    table = []
    for model, (reference, _), variants in MODELS:
        ref = load_patient_metrics(metrics_dir, reference)
        for label, experiment in variants:
            var = load_patient_metrics(runs_dir, experiment)
            for metric in METRICS:
                for organ in ORGANS:
                    table.append(
                        {
                            "model": model,
                            "variant": label,
                            "experiment": experiment,
                            "metric": metric,
                            "organ": organ,
                            "gain": gains(ref, var, organ, metric),
                        }
                    )
    return table


def draw_violin(ax, position: int, values: np.ndarray, limit: float) -> None:
    """One split violin: blue above zero, red below, with box, mean and clipped values marked."""
    shown = np.clip(values, -limit, limit)
    grid, dens = violin_profile(shown)
    half = 0.42 * dens
    for mask, color in ((grid >= 0, BLUE), (grid <= 0, RED)):
        edge = np.clip(grid, 0, None) if color == BLUE else np.clip(grid, None, 0)
        ax.fill_betweenx(
            edge,
            position - half,
            position + half,
            where=mask,
            color=color,
            alpha=0.35,
            lw=0,
        )
    ax.plot(position - half, grid, color=INK, lw=1, alpha=0.7)
    ax.plot(position + half, grid, color=INK, lw=1, alpha=0.7)
    ax.boxplot(
        values,
        positions=[position],
        widths=0.13,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color="white", lw=1.6),
        boxprops=dict(facecolor=INK, edgecolor=INK),
        whiskerprops=dict(color=INK),
        capprops=dict(lw=0),
        zorder=4,
    )
    mean = values.mean()
    ax.scatter(
        [position],
        [np.clip(mean, -limit, limit)],
        marker="D",
        s=40,
        color=BLUE if mean > 0 else RED,
        edgecolor="white",
        lw=0.9,
        zorder=6,
    )
    for edge_value, marker in ((limit, "^"), (-limit, "v")):
        count = int(
            (values >= limit).sum() if edge_value > 0 else (values <= -limit).sum()
        )
        if count:
            ax.scatter(
                [position],
                [edge_value],
                marker=marker,
                s=24,
                color=BLUE if edge_value > 0 else RED,
                zorder=5,
                clip_on=False,
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metrics-dir", type=Path, default=REPO / "metrics")
    parser.add_argument(
        "--runs-dir", type=Path, default=INPUTS / "snellius_inventory/loss_runs"
    )
    parser.add_argument("--out", type=Path, default=REPO / "figures")
    args = parser.parse_args(argv)

    table = collect(args.metrics_dir, args.runs_dir)
    variants = [(m, v) for m, _, vs in MODELS for v, _ in vs]
    plt = pyplot()
    fig, axs = plt.subplots(
        2,
        len(variants),
        figsize=(SLOT_WIDTH, SLOT_HEIGHT),
        sharey="row",
        gridspec_kw=dict(
            left=0.065, right=0.995, top=0.85, bottom=0.2, wspace=0.08, hspace=0.95
        ),
    )
    for row, (metric, (_, label, limit)) in enumerate(METRICS.items()):
        for col, (model, variant) in enumerate(variants):
            ax = axs[row, col]
            cells = {
                r["organ"]: r["gain"]
                for r in table
                if r["model"] == model
                and r["variant"] == variant
                and r["metric"] == metric
            }
            for j, organ in enumerate(ORGANS):
                draw_violin(ax, j, cells[organ], limit)
            ax.axhline(0, color=INK, lw=1.2, ls="--")
            ax.set_ylim(-limit, limit)
            ax.set_xlim(-0.6, 3.6)
            ax.set_xticks(
                range(4),
                [f"{ABBREV[o]}\n{int((cells[o] > 0).sum())}" for o in ORGANS],
                fontsize=11,
            )
            ax.tick_params(labelsize=11)
            ax.grid(axis="y", color="#E4E4E4", lw=0.6, ls=(0, (3, 3)))
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(variant, fontsize=11.5, fontweight="bold", pad=6)
        axs[row, 0].set_ylabel(label, fontsize=12)
    fig.text(
        0.5,
        0.012,
        "40 patients vs the upgraded ENet (seed 0); blue: better, red: worse; diamond: mean; number: patients improved (of 40)",
        ha="center",
        fontsize=10.5,
        color="#555555",
    )
    fig.text(
        0.5,
        0.995,
        "Upgraded ENet: change from Dice + CE for every patient, by organ",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"loss_violins.{ext}")

    write_csv(
        args.out / "loss_violins.csv",
        [
            {
                "model": r["model"],
                "variant": r["variant"],
                "metric": r["metric"],
                "organ": r["organ"],
                "patients": len(r["gain"]),
                "mean_gain": round(float(r["gain"].mean()), 4),
                "median_gain": round(float(np.median(r["gain"])), 4),
                "patients_improved": int((r["gain"] > 0).sum()),
            }
            for r in table
        ],
    )


if __name__ == "__main__":
    main()
