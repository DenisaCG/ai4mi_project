"""The final model against nnU-Net 3D on the same validation patients.

    python dataset_analysis/final_model_comparison.py [--data-dir ~/results_for_figures] [--out figures]

final_model_comparison.{png,pdf}, the slide: a scatter with the nnU-Net 3D Dice of a patient and organ on the x
axis and the final model's on the y axis, so a point above the dashed diagonal is a case the final model segments better
(the red shaded side is nnU-Net's, the blue side the final model's). Points are coloured by organ, a white star with the organ's outline marks each organ's mean, and the mean Dice and HD95
over the organs of both models are written in the panel; no legend, the organ colours are those of the other figures. The final model is the full pipeline (3-seed average, test-time
augmentation, largest component of heart and trachea) and its score uses the best epoch of every run; nnU-Net is its final
checkpoint without test-time augmentation. All are 3D on the original CT grid, over the 40 validation patients of the 4 folds.
Writes also final_model_comparison.csv.
"""

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from utils import INPUTS, REPO, pyplot, read_csv, write_csv

sys.path.insert(0, str(REPO / "tools"))
from plot_style import EARTH, EARTH_LABEL_COLORS  # noqa: E402

DATA_DIR = INPUTS / "results_for_figures"
ABLATION = "final_pipeline_ablation"
FULL_PIPELINE = "seeds_tta_lcc"
MODEL = "Final model"
# (label, model key in all_models_per_patient_3d_metrics.csv) of the nnU-Net models on the x axes
REFERENCES = [("nnU-Net 3D", "nnunet_3d_fullres_no_tta")]
ORGANS = ("esophagus", "heart", "trachea", "aorta")
ORGAN_COLORS = dict(zip(ORGANS, [EARTH_LABEL_COLORS[k] for k in (1, 2, 3, 4)]))
TEXT = "#3A3A3A"
LIMITS = (0.55, 1.0)


def patient_organ(rows: list[dict]) -> dict:
    """Mean over the seeds of every metric of a patient and organ.

    Args:
        rows: Rows with patient, class_name, dice, hd95 and assd.

    Returns:
        (patient, organ) to a dict of the three metrics.
    """
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m in ("dice", "hd95", "assd"):
            cells[r["patient"], r["class_name"]][m].append(float(r[m]))
    return {k: {m: statistics.fmean(v) for m, v in c.items()} for k, c in cells.items()}


def collect(data_dir: Path) -> dict:
    """The per patient and organ scores of the final model and the nnU-Net models.

    Args:
        data_dir: Folder holding all_models_per_patient_3d_metrics.csv and the final_pipeline_ablation folder.

    Returns:
        Label (MODEL and the labels of REFERENCES) to the output of patient_organ().
    """
    keys = {key: label for label, key in REFERENCES}
    by_model = defaultdict(list)
    for r in read_csv(data_dir / "all_models_per_patient_3d_metrics.csv"):
        if r["model"] in keys:
            by_model[keys[r["model"]]].append(r)
    by_model[MODEL] = [
        r
        for fold in range(4)
        for r in read_csv(data_dir / ABLATION / f"fold{fold}.csv")
        if r["variant"] == FULL_PIPELINE
    ]
    return {label: patient_organ(rows) for label, rows in by_model.items()}


def overall(scores: dict, metric: str) -> float:
    """Mean over the organs of the mean over the patients of a metric."""
    return statistics.fmean(
        statistics.fmean(v[metric] for (_, o), v in scores.items() if o == organ)
        for organ in ORGANS
    )


def figure(data: dict, plt):
    """The slide figure: the final model against each nnU-Net model.

    Args:
        data: Output of collect().
        plt: matplotlib.pyplot.

    Returns:
        The matplotlib figure.
    """
    model = data[MODEL]
    lo, hi = LIMITS
    fig, axs = plt.subplots(1, len(REFERENCES), figsize=(4.4, 4.4), squeeze=False)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.9, bottom=0.115)
    for ax, (label, _) in zip(axs[0], REFERENCES):
        ref = data[label]
        ax.fill_between([lo, hi], [lo, hi], lo, color="#F4E3DF", lw=0, zorder=0)
        ax.fill_between([lo, hi], [lo, hi], hi, color="#E3ECF1", lw=0, zorder=0)
        ax.plot([lo, hi], [lo, hi], color="#444444", ls="--", lw=1)
        for organ in ORGANS:
            keys = [k for k in ref if k[1] == organ]
            x = [ref[k]["dice"] for k in keys]
            y = [model[k]["dice"] for k in keys]
            ax.scatter(
                x, y, s=16, color=ORGAN_COLORS[organ], alpha=0.75, lw=0, zorder=3
            )
            ax.scatter(
                statistics.fmean(x), statistics.fmean(y), s=170, marker="*",
                color="white", edgecolor=ORGAN_COLORS[organ], lw=1.7, zorder=5,
            )  # fmt: skip
        ax.text(
            0.05, 0.95,
            f"Dice  {overall(model, 'dice'):.3f} vs {overall(ref, 'dice'):.3f}\n"
            f"HD95  {overall(model, 'hd95'):.1f} vs {overall(ref, 'hd95'):.1f} mm",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5, fontweight="bold", color=TEXT,
        )  # fmt: skip
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{label}, Dice", fontsize=10, color=TEXT)
        ax.grid(color="#D9D9D9", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ticks = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    axs[0][0].set_ylabel(f"{MODEL}, Dice", fontsize=10, color=TEXT)
    fig.text(
        0.99, 0.985, "Final model against nnU-Net 3D",
        ha="right", va="top", fontsize=14, fontweight="bold", color=EARTH[1],
    )  # fmt: skip
    return fig


def table(data: dict) -> list[dict]:
    """One row per model with its overall means and the change of the final model's to it."""
    model = data[MODEL]
    return [
        {
            "model": label,
            "dice": round(overall(data[label], "dice"), 4),
            "hd95": round(overall(data[label], "hd95"), 2),
            "assd": round(overall(data[label], "assd"), 2),
            "final_minus_model_dice": round(
                overall(model, "dice") - overall(data[label], "dice"), 4
            ),
            "final_minus_model_hd95": round(
                overall(model, "hd95") - overall(data[label], "hd95"), 2
            ),
        }
        for label in [MODEL] + [r[0] for r in REFERENCES]
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=REPO / "figures")
    args = parser.parse_args(argv)

    plt = pyplot()
    data = collect(args.data_dir)
    fig = figure(data, plt)
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"final_model_comparison.{ext}")
    write_csv(args.out / "final_model_comparison.csv", table(data))


if __name__ == "__main__":
    main()
