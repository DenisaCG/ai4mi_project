"""Every result of the project on one heat-map table: per-organ Dice, mean Dice, HD95 and ASSD.

    python dataset_analysis/all_results_table.py [--data-dir deck/results_for_figures] [--out figures]

all_results_table.{png,pdf}: one row per model or pipeline step, grouped (our three models, the steps of the final
model's prediction pipeline, nnU-Net, earlier and side runs); each cell is shaded from light to dark teal by how good
the value is among all rows of its column (higher Dice, lower HD95 and ASSD are darker), the best value of a column is
bold, and the full pipeline is outlined. Our single-run scores use the best epoch of every run (about 0.005 optimistic),
nnU-Net its final checkpoint. All are 3D on the original CT grid, over the 40 validation patients of the 4 folds.
Writes also all_results_table.csv.
"""

import argparse
import csv
import sys
from pathlib import Path

from utils import REPO, pyplot, read_csv, write_csv

sys.path.insert(0, str(REPO / "tools"))
from plot_style import EARTH  # noqa: E402

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
    (
        "Earlier and side runs",
        [
            ("Base ENet, 25 epochs", "models", "x_base_enet_ce_25ep"),
            ("Improved ENet, 25 epochs", "models", "x_improved_enet_dice_ce_25ep"),
            ("ResEnc U-Net, 25 epochs", "models", "x_resenc_ds_baseline_25ep"),
            ("Pretrained ResNet34, 25 epochs", "models", "x_resnet34_pretrained_25ep"),
            ("Final recipe, cosine LR", "models", "x_final_recipe_cosine_decay"),
        ],
    ),
]
COLUMNS = [(o.capitalize(), o, "dice") for o in ORGANS] + [
    ("Dice", "OVERALL", "dice"),
    ("HD95 (mm)", "OVERALL", "hd95"),
    ("ASSD (mm)", "OVERALL", "assd"),
]
LOWER_IS_BETTER = {"hd95", "assd"}
FULL = "Full pipeline"
TEXT = "#1A1A1A"
DARK = EARTH[1]  # teal


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


def shade(value: float, lo: float, hi: float, lower_is_better: bool) -> float:
    """Position of a value in its column's range, 1 for the best row and 0 for the worst."""
    t = (value - lo) / (hi - lo)
    return 1 - t if lower_is_better else t


def figure(rows: list[dict], plt):
    """The heat-map table.

    Args:
        rows: Output of collect().
        plt: matplotlib.pyplot.

    Returns:
        The matplotlib figure.
    """
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("teal", ["#E4EBED", DARK])
    lines = []  # (kind, payload)
    last = None
    for r in rows:
        if r["group"] != last:
            lines.append(("group", r["group"]))
            last = r["group"]
        lines.append(("row", r))
    n = len(lines)
    label_w, cell_w = 3.9, 1.1
    fig, ax = plt.subplots(figsize=(12.1, 5.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.075)
    ax.set_xlim(0, label_w + cell_w * len(COLUMNS))
    ax.set_ylim(n + 1.2, 0)
    ax.axis("off")
    ranges = {}
    best = {}
    for c, (_, organ, metric) in enumerate(COLUMNS):
        values = [r["values"][organ, metric] for r in rows]
        ranges[c] = (min(values), max(values))
        best[c] = min(values) if metric in LOWER_IS_BETTER else max(values)
    for c, (name, _, _) in enumerate(COLUMNS):
        ax.text(
            label_w + (c + 0.5) * cell_w, 0.6, name,
            ha="center", va="center", fontsize=10.5, fontweight="bold", color=TEXT,
        )  # fmt: skip
    ax.plot([0, label_w + cell_w * len(COLUMNS)], [1.2, 1.2], color="#444444", lw=1)
    for i, (kind, item) in enumerate(lines):
        y = i + 1.2
        if kind == "group":
            ax.text(
                0.05, y + 0.5, item.upper(), ha="left", va="center", fontsize=8.5,
                fontweight="bold", color="#6B6B6B",
            )  # fmt: skip
            continue
        full = item["label"] == FULL
        ax.text(
            0.15, y + 0.5, item["label"], ha="left", va="center", fontsize=10.5,
            fontweight="bold" if full else "normal", color=TEXT,
        )  # fmt: skip
        for c, (_, organ, metric) in enumerate(COLUMNS):
            v = item["values"][organ, metric]
            lo, hi = ranges[c]
            t = shade(v, lo, hi, metric in LOWER_IS_BETTER)
            x = label_w + c * cell_w
            ax.add_patch(
                plt.Rectangle((x + 0.03, y + 0.06), cell_w - 0.06, 0.88, color=cmap(t * 0.85), lw=0)
            )  # fmt: skip
            dark_text = t > 0.6
            text = (
                f"{v:.3f}"
                if metric == "dice"
                else f"{v:.1f}"
                if metric == "hd95"
                else f"{v:.2f}"
            )
            ax.text(
                x + cell_w / 2, y + 0.5, text, ha="center", va="center", fontsize=10,
                fontweight="bold" if v == best[c] else "normal",
                color="white" if dark_text else TEXT,
            )  # fmt: skip
        if full:
            ax.add_patch(
                plt.Rectangle(
                    (0.02, y + 0.03), label_w + cell_w * len(COLUMNS) - 0.04, 0.94,
                    fill=False, ec=EARTH[0], lw=1.8,
                )
            )  # fmt: skip
    fig.text(
        0.01, 0.01, "Darker = better within a column; bold = best of the column. Our scores use the best epoch of each run "
        "(about 0.005 optimistic);\n"
        "nnU-Net uses its final checkpoint. 3D, 40 validation patients, 4 folds.",
        ha="left", va="bottom", fontsize=8.5, color="#4D4D4D", style="italic",
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
