"""Every result of the final model's comparison on one figure: per-organ Dice, mean Dice, HD95 and ASSD.

    python dataset_analysis/all_results_table.py [--data-dir deck/results_for_figures] [--out figures]

all_results_table.{png,pdf}: one row per model or pipeline step, tinted by group (our three models, the steps of the
final model's prediction pipeline, nnU-Net). The dots give the Dice of the four organs on one axis, a star the mean of
the four, and the dashed line the full pipeline's mean; the mean Dice, HD95 and ASSD sit to the right, with the change
of HD95 to the full pipeline in a pill (teal: lower, brick: higher). Our single-run scores use the best epoch of every
run (about 0.005 optimistic), nnU-Net its final checkpoint. All are 3D on the original CT grid, over the 40 validation
patients of the 4 folds.
Writes also all_results_table.csv.
"""

import argparse
import csv
import sys
from pathlib import Path

from final_model_comparison import ORGAN_COLORS
from postprocessing_figure import BETTER, WORSE, change_text
from utils import REPO, pyplot, read_csv, write_csv

sys.path.insert(0, str(REPO / "tools"))
from plot_style import EARTH, PALETTE, tint  # noqa: E402

DATA_DIR = REPO / "deck/results_for_figures"
ABLATION = "final_pipeline_ablation"
ORGANS = ("esophagus", "heart", "trachea", "aorta")
# (group, then (label, source, key)): source "models" = all_models_per_patient_3d_metrics.csv summary, "pipeline" = ablation
GROUPS = [
    (
        "Our models, single runs",
        [
            ("Base ENet", "models", "1_base_enet_ce_50ep"),
            ("Improved ENet", "models", "2_improved_enet_dice_ce_50ep"),
            ("Final model", "models", "3_final_resenc_ds_strong_geo_50ep"),
        ],
    ),
    (
        "Final model, prediction pipeline",
        [
            ("+ test-time augmentation", "pipeline", "single_tta"),
            ("+ largest component", "pipeline", "single_lcc"),
            ("+ 3-seed average", "pipeline", "seeds"),
            ("+ seed average, largest component", "pipeline", "seeds_lcc"),
            ("+ seed average, test-time aug.", "pipeline", "seeds_tta"),
            ("Full pipeline", "pipeline", "seeds_tta_lcc"),
        ],
    ),
    (
        "nnU-Net",
        [
            ("nnU-Net 2D", "models", "nnunet_2d_no_tta"),
            ("nnU-Net 2D, mirroring TTA", "models", "nnunet_2d_tta"),
            ("nnU-Net 3D", "models", "nnunet_3d_fullres_no_tta"),
            ("nnU-Net 3D, mirroring TTA", "models", "nnunet_3d_fullres_tta"),
        ],
    ),
]
COLUMNS = [(o.capitalize(), o, "dice") for o in ORGANS] + [
    ("Dice", "OVERALL", "dice"),
    ("HD95 (mm)", "OVERALL", "hd95"),
    ("ASSD (mm)", "OVERALL", "assd"),
]
NEUTRAL = "#888888"  # a change of HD95 below 0.05 mm
# (fill, line) colours of the rows of each group, by the name of the group in GROUPS.
BOX_COLORS = {
    "Our models, single runs": ("#AAAAAA", "#555555"),
    "Final model, prediction pipeline": (PALETTE[5], EARTH[0]),
    "nnU-Net": (PALETTE[0], "#3F6A8F"),
}
DICE_LIM = (0.45, 1.0)
FULL = "Full pipeline"
TEXT = "#1A1A1A"


def pipeline_scores(data_dir: Path) -> dict:
    """The scores of the prediction pipeline variants of the final model.

    Args:
        data_dir: Folder holding final_pipeline_ablation/summary_per_organ.csv and summary_overall.csv.

    Returns:
        Variant to {(organ or "OVERALL", metric): value}.
    """
    out = {}
    with (data_dir / ABLATION / "summary_per_organ.csv").open(newline="") as f:
        rows = list(csv.reader(f))
    metrics, organs = rows[0][1:], rows[1][1:]
    for row in rows[3:]:
        out[row[0]] = {
            (o, m): float(v) for m, o, v in zip(metrics, organs, row[1:], strict=True)
        }
    with (data_dir / ABLATION / "summary_overall.csv").open(newline="") as f:
        rows = list(csv.reader(f))
    for row in rows[3:]:
        for m, i in (("dice", 1), ("hd95", 3), ("assd", 5)):
            out[row[0]][("OVERALL", m)] = float(row[i])
    return out


def collect(data_dir: Path) -> list[dict]:
    """The rows of the table.

    Args:
        data_dir: Folder holding summary_per_organ_and_overall.csv and the final_pipeline_ablation folder.

    Returns:
        One dict per row in order: group, label and values (the column's (organ, metric) to number).
    """
    summary = {
        (r["model"], r["class_name"]): r
        for r in read_csv(data_dir / "summary_per_organ_and_overall.csv")
    }
    pipeline = pipeline_scores(data_dir)
    rows = []
    for group, members in GROUPS:
        for label, source, key in members:
            if source == "models":
                values = {(o, m): float(summary[key, o][m]) for _, o, m in COLUMNS}
            else:
                values = pipeline[key]
            rows.append({"group": group, "label": label, "values": values})
    return rows


def figure(rows: list[dict], plt):
    """The figure: the organs' Dice as dots on one axis, with the mean Dice, HD95, ASSD and the HD95 change beside it.

    Args:
        rows: Output of collect().
        plt: matplotlib.pyplot.

    Returns:
        The matplotlib figure; the reference of the dashed line and of the pills is the full pipeline.
    """
    full = next(r for r in rows if r["label"] == FULL)["values"]
    lines = []
    for group, members in GROUPS:
        lines.append(("group", group))
        lines += [
            ("row", r)
            for r in rows
            if r["group"] == group and r["label"] in {m[0] for m in members}
        ]
    n = len(lines)
    fig, ax = plt.subplots(figsize=(12.1, 5.0))
    fig.subplots_adjust(left=0.27, right=0.62, top=0.9, bottom=0.2)
    for i, (kind, item) in enumerate(lines):
        y = n - 1 - i
        if kind == "group":
            ax.text(
                -0.7, y, item.upper(), transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=9, fontweight="bold", color="#6B6B6B",
            )  # fmt: skip
            continue
        fill, line = BOX_COLORS[item["group"]]
        ax.axhspan(y - 0.5, y + 0.5, color=tint(fill, 0.22), lw=0, zorder=0)
        if lines[i - 1][0] == "group":
            ax.axhline(y + 0.5, color="#666666", lw=0.8, zorder=1)
        v = item["values"]
        dice = [v[o, "dice"] for o in ORGANS]
        ax.plot([min(dice), max(dice)], [y, y], color="#BBBBBB", lw=2, zorder=2)
        for o, d in zip(ORGANS, dice):
            ax.scatter(
                d, y, s=46, color=ORGAN_COLORS[o], edgecolor="white", lw=0.8, zorder=3
            )
        ax.scatter(
            v["OVERALL", "dice"], y, marker="*", s=190, color="white",
            edgecolor="#222222", lw=1.0, zorder=5,
        )  # fmt: skip
        ax.text(
            -0.02, y, item["label"], transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=11, fontweight="bold" if item["label"] == FULL else "normal", color=TEXT,
        )  # fmt: skip
        for x, text in (
            (1.1, f"{v['OVERALL', 'dice']:.3f}"),
            (1.3, f"{v['OVERALL', 'hd95']:.1f}"),
            (1.5, f"{v['OVERALL', 'assd']:.2f}"),
        ):
            ax.text(
                x, y, text, transform=ax.get_yaxis_transform(), ha="center", va="center",
                fontsize=11, fontweight="bold", color=line,
            )  # fmt: skip
        if item["label"] != FULL:
            change = v["OVERALL", "hd95"] - full["OVERALL", "hd95"]
            ax.text(
                1.72, y, change_text(change), transform=ax.get_yaxis_transform(), ha="center",
                va="center", fontsize=10.5, fontweight="bold", color="white",
                bbox={
                    "boxstyle": "round,pad=0.25", "ec": "none",
                    "fc": NEUTRAL if abs(change) < 0.05 else BETTER if change < 0 else WORSE,
                },
            )  # fmt: skip
    ax.axvline(full["OVERALL", "dice"], color="#444444", lw=1.0, ls="--", zorder=1)
    ax.set_xlim(*DICE_LIM)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([])
    ax.grid(axis="x", color="#D0D0D0", lw=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=11, labelcolor=TEXT)
    ax.set_xlabel("Dice of each organ (3D, original CT grid)", fontsize=11, color=TEXT)
    for x, name in (
        (1.1, "Dice"),
        (1.3, "HD95\n(mm)"),
        (1.5, "ASSD\n(mm)"),
        (1.72, "\u0394 HD95"),
    ):
        ax.text(
            x, n - 0.35, name, transform=ax.get_yaxis_transform(), ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=TEXT,
        )  # fmt: skip
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=ORGAN_COLORS[o], ms=7, label=o.capitalize())
        for o in ORGANS
    ]  # fmt: skip
    handles += [
        plt.Line2D([], [], marker="*", mfc="white", mec="#222222", color="#222222", ls="", ms=12, label="mean of the four"),
        plt.Line2D([], [], color="#444444", ls="--", label="full pipeline mean"),
    ]  # fmt: skip
    fig.legend(
        handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=10.5,
        handletextpad=0.2, columnspacing=1.0, labelcolor=TEXT,
    )  # fmt: skip
    return fig


def table(rows: list[dict]) -> list[dict]:
    """One row per model or pipeline step with the seven numbers of the figure."""
    return [
        {
            "group": r["group"],
            "model": r["label"],
            **{
                f"{organ}_{metric}": round(r["values"][organ, metric], 4)
                for _, organ, metric in COLUMNS
            },
        }
        for r in rows
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=REPO / "figures")
    args = parser.parse_args(argv)

    plt = pyplot()
    rows = collect(args.data_dir)
    fig = figure(rows, plt)
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"all_results_table.{ext}")
    write_csv(args.out / "all_results_table.csv", table(rows))


if __name__ == "__main__":
    main()
