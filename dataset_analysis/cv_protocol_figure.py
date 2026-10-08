"""Per-patient Dice of every cross-validation run, one column per fold, sized for a slide.

    python dataset_analysis/cv_protocol_figure.py [--experiment full_cv4_enet_dice_ce] [--out figures]

One column per validation fold and one row per patient, sorted by mean Dice. Per organ (colour) the large dot is the mean over the 3
seeds and the three small markers (circle, square, diamond) are the individual seeds; the black tick is the mean over the 4 organs and
the dashed line is the fold mean of that tick (3D Dice, original CT grid, best checkpoint).
Writes cv_protocol.{png,pdf} and cv_protocol_runs.csv (one row per fold, seed, patient and organ).
"""

import argparse
import statistics
import sys
from pathlib import Path

from utils import CLASSES, INK, N_FOLDS, REPO, load_cv_runs, pyplot, write_csv

sys.path.insert(0, str(REPO / "tools"))
from plot_style import EARTH_LABEL_COLORS, apply_style  # noqa: E402

MARKERS = "osD"  # one per seed
SEED_OFFSET = 0.17  # vertical spacing of the seed markers within a patient row
ORGAN_COLORS = {name: EARTH_LABEL_COLORS[idx] for idx, name in CLASSES.items()}


def organ_means(rows: list[dict]) -> dict[tuple[str, str], float]:
    """Mean Dice over the seeds for every (patient, organ)."""
    cells: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        cells.setdefault((r["patient"], r["organ"]), []).append(r["dice"])
    return {key: statistics.fmean(v) for key, v in cells.items()}


def fold_panel(
    ax, rows: list[dict], means: dict[tuple[str, str], float], fold: int
) -> None:
    """Draw the patients of one fold, best patient on top."""
    patients = sorted({r["patient"] for r in rows})
    mean_over_organs = {
        p: statistics.fmean(means[p, o] for o in CLASSES.values()) for p in patients
    }
    patients.sort(key=lambda p: mean_over_organs[p], reverse=True)
    top = len(patients) - 1
    ax.axvline(
        statistics.fmean(mean_over_organs.values()),
        color="#666666",
        lw=1.2,
        ls="--",
        zorder=0,
    )
    for k, patient in enumerate(patients):
        y = top - k
        for r in rows:
            if r["patient"] == patient:
                ax.scatter(
                    r["dice"], y + (r["seed"] - 1) * SEED_OFFSET, marker=MARKERS[r["seed"]], s=11,
                    color=ORGAN_COLORS[r["organ"]], edgecolor="none", zorder=2,
                )  # fmt: skip
        for organ in CLASSES.values():
            ax.scatter(
                means[patient, organ],
                y,
                s=42,
                color=ORGAN_COLORS[organ],
                edgecolor="white",
                lw=0.7,
                zorder=3,
            )
        ax.scatter(
            mean_over_organs[patient], y, marker="|", s=210, color=INK, lw=1.8, zorder=4
        )
    ax.set_yticks(
        range(len(patients)),
        [p.replace("Patient_", "P") for p in reversed(patients)],
        fontsize=11,
    )
    ax.set_ylim(-0.6, top + 0.6)
    ax.set_title(f"Fold {fold}", fontsize=14, fontweight="bold")
    ax.set_xticks(
        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        [".3", ".4", ".5", ".6", ".7", ".8", ".9", "1"],
    )
    ax.set_xticks([x / 100 for x in range(25, 100, 5)], minor=True)
    ax.grid(axis="x", which="major", color="#C8C8C8", lw=0.8)
    ax.grid(axis="x", which="minor", color="#E4E4E4", lw=0.6, ls=(0, (2, 2)))
    ax.grid(axis="y", color="#E4E4E4", lw=0.6, ls=(0, (2, 2)))
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="x", which="minor", length=2)
    ax.set_xlim(0.28, 1.0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--experiment", default="full_cv4_enet_dice_ce")
    parser.add_argument("--metrics-dir", type=Path, default=REPO / "metrics")
    parser.add_argument("--out", type=Path, default=REPO / "figures")
    args = parser.parse_args(argv)

    plt = pyplot()
    apply_style()
    plt.rcParams["axes.grid"] = False
    rows = load_cv_runs(args.metrics_dir, args.experiment)
    means = organ_means(rows)

    fig, axs = plt.subplots(1, N_FOLDS, figsize=(11.9, 3.85), sharex=True)
    for fold, ax in enumerate(axs):
        fold_panel(ax, [r for r in rows if r["fold"] == fold], means, fold)
    handles = [
        plt.Line2D([], [], marker="o", color=c, ls="", markersize=8, label=o)
        for o, c in ORGAN_COLORS.items()
    ]
    handles += [
        plt.Line2D(
            [], [], marker=m, color="#777777", ls="", markersize=6, label=f"seed {i}"
        )
        for i, m in enumerate(MARKERS)
    ]
    handles += [
        plt.Line2D(
            [],
            [],
            marker="|",
            color=INK,
            ls="",
            markersize=11,
            markeredgewidth=1.8,
            label="mean over organs",
        ),
        plt.Line2D([], [], color="#666666", ls="--", lw=1.2, label="fold mean"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=9,
        fontsize=11,
        frameon=False,
        handletextpad=0.2,
        columnspacing=1.0,
    )
    fig.supxlabel("Dice (3D, original CT grid)", fontsize=12, y=0.13)
    axs[0].set_ylabel("Patient, best to worst Dice", fontsize=12)
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    fig.text(
        axs[-1].get_position().x1,
        0.99,
        "Per-patient Dice across folds and seeds",
        ha="right",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"cv_protocol.{ext}")

    write_csv(
        args.out / "cv_protocol_runs.csv",
        [
            {**r, "dice": round(r["dice"], 4)}
            for r in sorted(
                rows, key=lambda r: (r["fold"], r["patient"], r["organ"], r["seed"])
            )
        ],
    )


if __name__ == "__main__":
    main()
