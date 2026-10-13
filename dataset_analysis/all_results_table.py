"""The final model's results on one figure of boxes: with and without test-time augmentation and post-processing.

    python dataset_analysis/all_results_table.py [--data-dir ~/results_for_figures] [--out figures]

all_results_table.{png,pdf}: one box per model with its mean Dice, HD95 and ASSD over the four organs. Left, the base and
improved ENet; right the final model as a grid of the single run and the 3-seed average with the largest component of heart
and trachea (post-processing), each without and with test-time augmentation, the change of each to the single run in the
box; below, the boundary-loss and 2.5D arms of boundary_arms.csv with the change of their HD95 to the full pipeline. Our
scores use the best epoch of every run (about 0.005 optimistic). All are 3D on the original CT grid, over the 40
validation patients of the 4 folds.
Writes also all_results_table.csv (every row of GROUPS with its per-organ values).
"""

import argparse
import csv
import sys
from pathlib import Path

from utils import INPUTS, REPO, pyplot, read_csv, write_csv

sys.path.insert(0, str(REPO / "tools"))
from plot_style import EARTH, PALETTE, tint  # noqa: E402

DATA_DIR = INPUTS / "results_for_figures"
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
# training setup under the name of each ENet card
SETUP = {
    "Base ENet": "cross-entropy,\nno preprocessing",
    "Improved ENet": "Dice + cross-entropy,\nfull preprocessing",
}
COLUMNS = [(o.capitalize(), o, "dice") for o in ORGANS] + [
    ("Dice", "OVERALL", "dice"),
    ("HD95 (mm)", "OVERALL", "hd95"),
    ("ASSD (mm)", "OVERALL", "assd"),
]
RAW = "Final model"
FULL = "Full pipeline"
SEEDS_LCC = "+ seed average, largest component"
# the final model's cards: rows without and with TTA, columns without and with post-processing (3-seed average, largest component)
FINAL_GRID = [
    [RAW, SEEDS_LCC],
    ["+ test-time augmentation", FULL],
]
WIDE, SMALL_W, SMALL_H = 1.2, 1.3, 1.65  # card sizes in inches
FINAL_W = 1.9  # width of the final model's cards
ARM_W = 1.9  # width of the boundary-loss boxes
REF_FILL, REF_LINE, GREY_TEXT = (
    "#FBF0CF",
    "#C9A227",
    "#6B6B6B",
)  # earlier models and the reference
WORSE = "#A02A1B"  # text for a worse value, a deep red that stays crisp on yellow, green and orange
GREEN, GREEN_FILL = "#5E9142", "#EEF5EA"
TEXT = "#1A1A1A"


# label -> (cross-validation experiment of its training runs, timing source, number of runs it needs); the source is the
# per-patient file timing/<stem>_inference_time.csv, or the mode ("none" or "tta") of final_model_retime.csv for the
# final model, which was timed again in one job with the boundary-loss variants
FINAL_EXPERIMENT = "full_cv4_resenc_ds_unet_dice_ce_strong_geometric_50ep"
RUNS = {
    "Base ENet": ("full_cv4_enet_ce_50ep", "ours_base_enet", 1),
    "Improved ENet": ("full_cv4_enet_dice_ce_50ep", "ours_improved_enet", 1),
    RAW: (FINAL_EXPERIMENT, "none", 1),
    SEEDS_LCC: (FINAL_EXPERIMENT, "none", 3),
    "+ test-time augmentation": (FINAL_EXPERIMENT, "tta", 1),
    FULL: (FINAL_EXPERIMENT, "tta", 3),
}


DICE_NOISE = 0.002  # smaller Dice changes are within the seed spread
HD95_NOISE = 0.2  # mm


def tone(change: float, noise: float, worse_when_positive: bool = False) -> str:
    """Colour of a change: grey inside the noise, red if worse, blue if better.

    Args:
        change: Signed change of the metric.
        noise: Changes smaller than this in magnitude are grey.
        worse_when_positive: True for metrics where an increase is worse (HD95).

    Returns:
        A colour.
    """
    if abs(change) < noise:
        return GREY_TEXT
    return WORSE if (change > 0) == worse_when_positive else EARTH[1]


def run_times(data_dir: Path, metrics_dir: Path) -> dict[str, tuple[float, float]]:
    """Training and inference time of every box that has runs behind it.

    Args:
        data_dir: Folder holding timing/final_model_retime.csv and the per-patient timing/<stem>_inference_time.csv.
        metrics_dir: Folder holding the cross-validation runs, each with an epochs.csv of seconds per epoch.

    Returns:
        Label to (training minutes, inference seconds per patient), the sum over the runs the box needs.
    """
    retime = {
        r["mode"]: float(r["mean_s_per_patient"])
        for r in read_csv(data_dir / "timing" / "final_model_retime.csv")
        if r["model"] == "control"
    }
    out = {}
    for label, (experiment, source, n_runs) in RUNS.items():
        train = [
            sum(float(r["seconds"]) for r in read_csv(f)) / 60
            for f in sorted(metrics_dir.glob(f"{experiment}_fold*/seed*/epochs.csv"))
        ]
        if source in retime:
            infer = retime[source]
        else:
            rows = read_csv(data_dir / "timing" / f"{source}_inference_time.csv")
            infer = sum(float(r["seconds"]) for r in rows) / len(rows)
        out[label] = (n_runs * sum(train) / len(train), n_runs * infer)
    return out


def time_lines(
    x: float, y: float, w: float, name: str, times: dict, label, size: float = 9
):
    """Draw the train and infer lines of a box: the value left, the factor to the single run right.

    Args:
        x: Left edge of the box.
        y: Height of the lower line.
        w: Width of the box.
        name: Label of the box in times.
        times: Output of run_times().
        label: The figure's text function.
        size: Font size.
    """
    for dy, word, value, text, ref in (
        (0.17, "train", times[name][0], duration(times[name][0]), times[RAW][0]),
        (0.0, "infer", times[name][1], f"{times[name][1]:.2f} s", times[RAW][1]),
    ):
        ratio = value / ref
        colour = WORSE if ratio > 1.05 else EARTH[1] if ratio < 0.95 else TEXT
        label(x + 0.1, y + dy, f"{word} {text}", size, ha="left", color=TEXT)
        label(
            x + w - 0.1, y + dy, "ref" if name == RAW else f"\u00d7{ratio:.1f}", size,
            ha="right", fontweight="bold", color=colour,
        )  # fmt: skip


def corner_ribbon(
    ax, x: float, top: float, text: str, colour: str, size: float = 0.72
) -> None:
    """Draw a diagonal ribbon across the top-left corner of a box.

    Args:
        ax: Axes of the figure.
        x: Left edge of the box.
        top: Top edge of the box.
        text: Text on the ribbon.
        colour: Fill of the ribbon.
        size: Distance of the ribbon's outer edge from the corner, in inches, along both edges.
    """
    from matplotlib.patches import Polygon

    inner = size * 0.55
    ax.add_patch(
        Polygon(
            [(x, top - size), (x, top - inner), (x + inner, top), (x + size, top)],
            closed=True,
            fc=colour,
            ec="none",
            zorder=5,
        )
    )
    ax.text(
        x + (size + inner) / 4,
        top - (size + inner) / 4,
        text,
        rotation=45,
        ha="center",
        va="center",
        fontsize=6.5,
        fontweight="bold",
        color="white",
        zorder=6,
    )


def arm_times(data_dir: Path, arms: list[dict]) -> dict[str, tuple[float, float]]:
    """Training and inference time of the boundary-loss variants, post-processed without TTA (three runs, three passes).

    Args:
        data_dir: Folder holding timing/final_model_retime.csv.
        arms: Rows of boundary_arms.csv.

    Returns:
        Label to (training minutes, inference seconds per patient).
    """
    retime = {
        r["model"]: float(r["mean_s_per_patient"])
        for r in read_csv(data_dir / "timing" / "final_model_retime.csv")
        if r["mode"] == "none"
    }
    return {
        a["label"]: (3 * float(a["train_min_per_run"]), 3 * retime[a["retime_model"]])
        for a in arms
    }


def duration(minutes: float) -> str:
    """Minutes as "55 min", or as hours from two hours on."""
    return f"{minutes:.0f} min" if minutes < 120 else f"{minutes / 60:.1f} h"


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


def figure(rows: list[dict], arms: list[dict], times: dict, plt):
    """The card figure.

    Args:
        rows: Output of collect().
        times: Output of run_times().
        arms: Rows of boundary_arms.csv, the boundary-loss and 2.5D arms (label, dice, hd95, assd).
        plt: matplotlib.pyplot.

    Returns:
        The matplotlib figure.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    value = {r["label"]: r["values"] for r in rows}
    for arm in arms:
        value[arm["label"]] = {
            ("OVERALL", m): float(arm[m]) for m in ("dice", "hd95", "assd")
        }
    width, height = 12.5, 5.6
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(5.0 - height, 5.0)
    ax.axis("off")

    def metric(label, name):
        return value[label]["OVERALL", name]

    def signed(v, spec):
        if float(f"{v:{spec}}") == 0:
            return f"\u00b1{0:{spec}}"
        return f"{v:+{spec}}".replace("-", "\u2212")

    def card(x, y, w, h, fc, ec, lw=1.1):
        ax.add_patch(
            FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec=ec, lw=lw)
        )  # fmt: skip

    def label(x, y, text, size, **kw):
        ax.text(
            x,
            y,
            text,
            ha=kw.pop("ha", "center"),
            va=kw.pop("va", "center"),
            fontsize=size,
            **kw,
        )

    y0, top = 0.2, 3.85
    for key, x in (("Base ENet", 0.1), ("Improved ENet", 1.95)):
        card(x, y0, WIDE, top - y0, REF_FILL, REF_LINE)
        label(x + WIDE / 2, top - 0.3, key, 10, fontweight="bold", color=TEXT)
        label(
            x + WIDE / 2, top - 0.62, SETUP[key], 7.5, color=GREY_TEXT, linespacing=1.3
        )
        label(
            x + WIDE / 2,
            top - 1.1,
            f"{metric(key, 'dice'):.3f}",
            22,
            fontweight="bold",
            color=TEXT,
        )
        label(x + WIDE / 2, top - 1.5, "Dice", 9, color=GREY_TEXT)
        label(
            x + WIDE / 2,
            top - 1.95,
            f"HD95 {metric(key, 'hd95'):.1f} mm",
            9.5,
            color=TEXT,
        )
        label(
            x + WIDE / 2,
            top - 2.3,
            f"ASSD {metric(key, 'assd'):.2f} mm",
            9.5,
            color=TEXT,
        )
        time_lines(x, top - 3.0, WIDE, key, times, label, 7.5)
    mid = (y0 + top) / 2
    gain = signed(metric("Improved ENet", "dice") - metric("Base ENet", "dice"), ".3f")
    drop = signed(metric("Improved ENet", "hd95") - metric("Base ENet", "hd95"), ".1f")
    for x1, x2, text, size in (
        (0.1 + WIDE + 0.04, 1.91, f"{gain}\n{drop} mm", 8),
        (
            1.95 + WIDE + 0.04,
            1.95 + WIDE + 0.04 + (1.91 - (0.1 + WIDE + 0.04)),
            "ResEnc\nU-Net\n+\naugmentation",
            7,
        ),
    ):
        ax.add_patch(FancyArrowPatch((x1, mid), (x2, mid), arrowstyle="-|>", mutation_scale=22, color="#555555", lw=2.8))  # fmt: skip
        label(
            (x1 + x2) / 2 + (0.11 if size < 8 else 0),
            mid + 0.12,
            text,
            size,
            va="bottom",
            color=EARTH[1],
            fontweight="bold",
        )

    def small(x, y, name, fc, ec, lw, note, note_color, w=SMALL_W, tag=None):
        card(x, y, w, SMALL_H, fc, ec, lw)
        if tag:
            label(
                x + w / 2, y + SMALL_H - 0.12, tag, 7.5, color=GREY_TEXT, style="italic"
            )
        label(
            x + w / 2,
            y + SMALL_H - 0.36,
            f"{metric(name, 'dice'):.3f}",
            17,
            fontweight="bold",
            color=TEXT,
        )
        label(
            x + w / 2, y + SMALL_H - 0.72,
            f"HD95 {metric(name, 'hd95'):.1f} mm \u00b7 ASSD {metric(name, 'assd'):.2f} mm", 7.5, color=TEXT,
        )  # fmt: skip
        for k, (text, colour) in enumerate(
            [(note, note_color)] if isinstance(note, str) else note
        ):
            label(
                x + w / 2,
                y + (0.64 if isinstance(note, str) else 0.72 - 0.17 * k),
                text,
                8,
                color=colour,
                fontweight="bold",
                linespacing=1.3,
            )
        time_lines(x, y + 0.12, w, name, times, label, 8)

    def rows_top(i):
        return top - i * (SMALL_H + 0.35)

    mx = 4.05
    label(
        mx + (2 * FINAL_W + 0.08) / 2,
        4.45,
        "DICE + CE LOSS VARIANTS",
        10.5,
        fontweight="bold",
        color=GREEN,
    )
    for j, head in enumerate(("no post-processing", "post-processing")):
        label(
            mx + j * (FINAL_W + 0.08) + FINAL_W / 2,
            top + 0.08,
            head,
            9,
            va="bottom",
            color=GREY_TEXT,
            style="italic",
        )
    for i, tta in enumerate(("no TTA", "TTA")):
        label(
            mx - 0.08,
            rows_top(i) - SMALL_H / 2,
            tta,
            9.5,
            ha="right",
            color=GREY_TEXT,
            style="italic",
        )
        for j, name in enumerate(FINAL_GRID[i]):
            x, y = mx + j * (FINAL_W + 0.08), rows_top(i) - SMALL_H
            if name == RAW:
                small(
                    x,
                    y,
                    name,
                    GREEN_FILL,
                    REF_LINE,
                    2.4,
                    "single models,\nmean of 3 seeds",
                    GREY_TEXT,
                    FINAL_W,
                )
                corner_ribbon(ax, x, y + SMALL_H, "REFERENCE", REF_LINE)
                continue
            full = name == FULL
            d_dice = metric(name, "dice") - metric(RAW, "dice")
            d_hd95 = metric(name, "hd95") - metric(RAW, "hd95")
            note = [
                (f"{signed(d_dice, '.3f')} Dice", tone(d_dice, DICE_NOISE)),
                (f"{signed(d_hd95, '.1f')} mm HD95", tone(d_hd95, HD95_NOISE, True)),
            ]
            small(
                x, y, name, tint(PALETTE[3], 0.45) if full else GREEN_FILL,
                GREEN if full else tint(GREEN, 0.5), 2.4 if full else 1.1, note, EARTH[1], FINAL_W,
            )  # fmt: skip
            if full:
                corner_ribbon(ax, x, y + SMALL_H, "FINAL MODEL", GREEN)
    nx = 8.45  # left edge of the boundary-loss boxes
    ax.plot([8.17, 8.17], [0.15, 4.7], color="#BBBBBB", lw=3.5)
    label(
        nx + (2 * ARM_W + 0.08) / 2,
        4.45,
        "BOUNDARY-LOSS VARIANTS, NO TTA",
        10.5,
        fontweight="bold",
        color=GREY_TEXT,
    )
    for j, head in enumerate(("2D", "2.5D")):
        label(
            nx + j * (ARM_W + 0.08) + ARM_W / 2, top + 0.08, head, 9,
            va="bottom", color=GREY_TEXT, style="italic",
        )  # fmt: skip
    for arm in arms:
        dim, _, setting = arm["label"].partition(" + boundary loss (")
        j, i = (0 if dim == "2D" else 1), (0 if setting.startswith("ramped") else 1)
        d_dice = value[arm["label"]]["OVERALL", "dice"] - metric(SEEDS_LCC, "dice")
        d_hd95 = value[arm["label"]]["OVERALL", "hd95"] - metric(SEEDS_LCC, "hd95")
        hd95_change = signed(d_hd95, ".1f")
        small(
            nx + j * (ARM_W + 0.08), rows_top(i) - SMALL_H, arm["label"], "#FDE6D2", "#E08A3C", 1.1,
            [
                (f"{signed(d_dice, '.3f')} Dice", tone(d_dice, DICE_NOISE)),
                (f"{hd95_change} mm HD95", tone(d_hd95, HD95_NOISE, True)),
            ],
            None, ARM_W, f"{setting.rstrip(')')} weight" if not setting.startswith("weight") else f"fixed {setting.rstrip(')')}",
        )  # fmt: skip
    fig.text(
        0.99, 0.985, "Accuracy and cost of each model",
        ha="right", va="top", fontsize=13.5, fontweight="bold",
    )  # fmt: skip
    fig.text(
        0.01, 0.02,
        "Green boxes: change from the reference (top left). Orange boxes: change from the post-processed model (top middle), no TTA. Unrounded values.\n"
        "Red worse, blue better, grey within noise (under 0.002 Dice or 0.2 mm HD95). Factor: times the reference. Best epoch.\n"
        "Post-processing: 3-seed average, then largest component of heart and trachea. Train and infer: one run or pass on an A100; three for 3 seeds.",
        fontsize=10.5, color="#4D4D4D", style="italic", linespacing=1.25,
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
    parser.add_argument("--metrics-dir", type=Path, default=REPO / "metrics")
    parser.add_argument("--out", type=Path, default=REPO / "figures")
    args = parser.parse_args(argv)

    plt = pyplot()
    rows = collect(args.data_dir)
    arms = read_csv(args.data_dir / "boundary_arms.csv")
    times = run_times(args.data_dir, args.metrics_dir) | arm_times(args.data_dir, arms)
    fig = figure(rows, arms, times, plt)
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"all_results_table.{ext}")
    write_csv(args.out / "all_results_table.csv", table(rows))


if __name__ == "__main__":
    main()
