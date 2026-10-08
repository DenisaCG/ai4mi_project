"""Effect of each post-processing method on Dice and HD95 against the unprocessed model, sized for a slide.

    python dataset_analysis/postprocessing_figure.py [--out figures]

One row per method; the base model (ResEnc deep-supervision U-Net, Dice + CE, no augmentation, 4 folds x 3 seeds) is the
dashed reference line with its seed and fold spread as shaded bands. The line runs from the base mean to the mean of the method, teal where the metric improves and brick
where it worsens. Small dots are the 40 patients (mean over the seeds and over the 4 organs); dots beyond the axis are drawn at
its edge as triangles. Rows are tinted by the type of method. The oracle gates organs per slice with the ground truth, so it
is an upper bound (hatched row, open marker), not a method.
Writes postprocessing.{png,pdf} and postprocessing.csv.
"""

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patheffects as pe

from utils import (
    METRICS,
    REPO,
    load_cv_runs,
    pyplot,
    read_csv,
    summarise,
    write_csv,
)

sys.path.insert(0, str(REPO / "tools"))
from plot_style import EARTH, GRID_COLOR, PALETTE, tint  # noqa: E402

BASE_EXPERIMENT = "full_cv4_resenc_ds_unet_dice_ce"
PP_DIR = REPO / "deck/snellius_inventory/igardner1_pp"
# (variant in the inventory, label on the slide, group, is the ground truth used)
METHODS = [
    ("gating_oracle:lcc", "Largest component,\nall organs", "filter", False),
    (
        "gating_oracle:lcc_heart_trachea",
        "Largest component,\nheart and trachea",
        "filter",
        False,
    ),
    ("gating_oracle:heart_hull", "Heart convex hull", "shape", False),
    ("seed_ensemble:seed_vote", "3-seed majority\nvote", "vote", False),
    (
        "seed_ensemble:seed_vote_lcc_heart_trachea",
        "Vote + largest comp.,\nheart and trachea",
        "vote",
        False,
    ),
    ("gating_oracle:gate", "Oracle gating\n(upper bound)", "oracle", True),
]
# Group name on the slide and background colour of its rows.
GROUPS = {
    "filter": ("Filter", tint(PALETTE[0], 0.22)),
    "shape": ("Shape", tint(PALETTE[3], 0.3)),
    "vote": ("Vote", tint(PALETTE[1], 0.25)),
    "oracle": ("Oracle", "none"),
}
TEXT = "#1A1A1A"
VALUE = "#0A0A0A"
# DejaVu Sans has no semi-bold; a thin stroke in the text colour gives the in-between weight.
SEMIBOLD = [pe.withStroke(linewidth=0.3, foreground=VALUE)]
BASE_LABEL = "Unprocessed\nbase model"
BETTER, WORSE = EARTH[1], EARTH[0]  # teal, brick
DICE_LIM, HD95_LIM = (0.78, 0.97), (0, 40)
FOLD_BAND = "#8A8272"
JITTER = 0.26  # half-height of the patient dot cloud in a row


def patient_values(rows: list[dict], column: str) -> dict[str, float]:
    """Mean of a metric over the seeds, then over the organs, for every patient.

    Args:
        rows: Per-patient rows of one method from the inventory.
        column: Metric column to average (blank cells are skipped).

    Returns:
        Patient name to value.
    """
    cells = defaultdict(list)
    for r in rows:
        if r[column]:
            cells[r["patient"], r["organ"]].append(float(r[column]))
    per_patient = defaultdict(list)
    for (patient, _), values in cells.items():
        per_patient[patient].append(statistics.fmean(values))
    return {p: statistics.fmean(v) for p, v in per_patient.items()}


def collect(metrics_dir: Path, pp_dir: Path = PP_DIR) -> dict:
    """Base model from the local runs and the methods from the post-processing inventory.

    Args:
        metrics_dir: Folder with the per-run metrics_3d.csv of the base experiment.
        pp_dir: Folder with pp_summary_from_rows.csv and pp_per_patient_from_rows.csv.

    Returns:
        The base summary (dice, hd95, seed and fold stds, per-patient values) and one dict per method with its
        label, oracle flag, run count, means and per-patient values.
    """
    runs = load_cv_runs(metrics_dir, BASE_EXPERIMENT, METRICS)
    base = summarise(runs)
    wanted = {key for key, _, _, _ in METHODS}
    summary = {
        f"{r['root'].split('results/')[1].split('/')[0]}:{r['method']}": r
        for r in read_csv(pp_dir / "pp_summary_from_rows.csv")
        if r["organ"] == "mean"
    }
    cells = defaultdict(list)
    for r in read_csv(pp_dir / "pp_per_patient_from_rows.csv"):
        if r["method"] in wanted:
            cells[r["method"]].append(r)
    methods = []
    for key, label, group, oracle in METHODS:
        row = summary[key]
        rows = cells[key]
        methods.append(
            {
                "key": key,
                "label": label,
                "group": group,
                "oracle": oracle,
                "n_runs": int(row["n_runs"]),
                "dice": float(row["dice"]),
                "hd95": float(row["hd95"]),
                "patient_dice": patient_values(rows, "dice"),
                "patient_hd95": patient_values(rows, "hd95"),
                "base_dice": float(row["base_dice"]),
                "base_hd95": float(row["base_hd95"]),
                "rows": rows,
            }
        )
    first = methods[0]["rows"]
    base["patient_dice"] = patient_values(
        [{**r, "dice": r["base_dice"]} for r in first], "dice"
    )
    base["patient_hd95"] = patient_values(
        [{**r, "hd95": r["base_hd95"]} for r in first], "hd95"
    )
    return {"base": base, "methods": methods}


def draw_cloud(ax, values: dict[str, float], y: float, limits, color) -> None:
    """Draw the patients of one row as small dots with a fixed vertical spread; dots past the axis sit on its edge."""
    low, high = limits
    for i, v in enumerate(values[k] for k in sorted(values)):
        dy = ((i * 0.618) % 1 - 0.5) * 2 * JITTER
        clipped = min(max(v, low), high)
        marker = "o" if clipped == v else ("<" if v < low else ">")
        ax.scatter(
            clipped, y + dy, s=9, marker=marker, color=color, alpha=0.5, lw=0,
            zorder=2, clip_on=False,
        )  # fmt: skip


def panel(ax, data: dict, metric: str, limits, higher_is_better: bool, fmt) -> None:
    """One metric: patient dots, the base reference, and a line per method."""
    base = data["base"]
    n = len(data["methods"])
    for std, alpha in ((f"{metric}_fold_std", 0.18), (f"{metric}_seed_std", 0.34)):
        ax.axvspan(
            base[metric] - base[std], base[metric] + base[std],
            color=FOLD_BAND, alpha=alpha, lw=0, zorder=0,
        )  # fmt: skip
    ax.axvline(base[metric], color="#555555", lw=1.4, ls="--", zorder=1)
    draw_cloud(ax, base[f"patient_{metric}"], n, limits, "#777777")
    for i, m in enumerate(data["methods"]):
        y = n - 1 - i
        background = GROUPS[m["group"]][1]
        if background != "none":
            ax.axhspan(y - 0.5, y + 0.5, color=background, lw=0, zorder=-1)
        if m["oracle"]:
            ax.axhspan(
                y - 0.5,
                y + 0.5,
                facecolor="none",
                hatch="///",
                edgecolor=GRID_COLOR,
                lw=0,
            )
        improved = (m[metric] > base[metric]) == higher_is_better
        color = BETTER if improved else WORSE
        draw_cloud(ax, m[f"patient_{metric}"], y, limits, color)
        ax.plot(
            [base[metric], m[metric]], [y, y], color=color, lw=3,
            ls="--" if m["oracle"] else "-", solid_capstyle="butt", zorder=4,
        )  # fmt: skip
        ax.scatter(
            m[metric], y, s=60, zorder=5, color="white" if m["oracle"] else color,
            edgecolor=color, lw=2,
        )  # fmt: skip
        right = m[metric] >= base[metric]
        ax.text(
            m[metric] + (1 if right else -1) * 0.02 * (limits[1] - limits[0]),
            y + 0.34,
            fmt(m[metric]),
            ha="left" if right else "right",
            va="center",
            fontsize=11,
            color=VALUE,
            path_effects=SEMIBOLD,
            zorder=6,
        )
    ax.text(
        base[metric],
        n + 0.34,
        fmt(base[metric]),
        ha="center",
        va="center",
        fontsize=11,
        color=VALUE,
        path_effects=SEMIBOLD,
    )
    ax.axhline(0.5, color="#999999", lw=1.0, zorder=1)
    ax.set_xlim(*limits)
    ax.set_ylim(-0.6, n + 0.7)
    ax.grid(axis="x", color=GRID_COLOR, lw=0.8, ls=(0, (3, 3)))
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def figure(data: dict, plt):
    """The two-panel slide figure."""
    n = len(data["methods"])
    fig, (ax_d, ax_h) = plt.subplots(
        1, 2, figsize=(7.9, 4.85), sharey=True, gridspec_kw={"wspace": 0.08}
    )
    panel(ax_d, data, "dice", DICE_LIM, True, lambda v: f"{v:.3f}")
    panel(ax_h, data, "hd95", HD95_LIM, False, lambda v: f"{v:.1f}")
    ax_d.set_xticks([0.8, 0.85, 0.9, 0.95], ["0.80", "0.85", "0.90", "0.95"])
    ax_h.set_xticks([0, 10, 20, 30, 40])
    ax_d.set_xlabel("Dice (higher is better)", fontsize=12, color=TEXT)
    ax_h.set_xlabel("HD95, mm (lower is better)", fontsize=12, color=TEXT)
    labels = [BASE_LABEL] + [m["label"] for m in data["methods"]]
    ax_d.set_yticks(range(n, -1, -1), labels, fontsize=11, color=TEXT)
    for ax in (ax_d, ax_h):
        ax.tick_params(axis="x", labelsize=11, labelcolor=TEXT)
    handles = [
        plt.Line2D([], [], color=BETTER, lw=2.2, label="improves"),
        plt.Line2D([], [], color=WORSE, lw=2.2, label="worsens"),
        plt.Line2D([], [], marker="o", mfc="white", mec=BETTER, mew=2, ls="--",
                   color=BETTER, label="oracle"),
        plt.Line2D([], [], marker="o", color="#777777", alpha=0.6, ls="", ms=4, label="patient"),
        plt.Rectangle((0, 0), 1, 1, fc=FOLD_BAND, alpha=0.34, label="seed \u00b11 SD"),
        plt.Rectangle((0, 0), 1, 1, fc=FOLD_BAND, alpha=0.18, label="fold \u00b11 SD"),
    ]  # fmt: skip
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.55, 0.0), ncol=6, fontsize=11, frameon=False, labelcolor=TEXT,
        handletextpad=0.3, columnspacing=0.9, handlelength=1.4,
    )  # fmt: skip
    fig.subplots_adjust(left=0.335, right=0.985, bottom=0.2, top=0.9, wspace=0.1)
    fig.text(
        0.004,
        0.55,
        "Method",
        rotation=90,
        ha="left",
        va="center",
        fontsize=12,
        color=TEXT,
    )
    rows = [m["group"] for m in data["methods"]]
    to_fig = ax_d.transData + fig.transFigure.inverted()
    for group in dict.fromkeys(rows):
        ys = [n - 1 - i for i, g in enumerate(rows) if g == group]
        fig.text(
            0.055,
            to_fig.transform((0, statistics.fmean(ys)))[1],
            GROUPS[group][0],
            rotation=90,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=TEXT,
        )
    fig.text(
        ax_h.get_position().x1,
        0.985,
        "Effect of post-processing on Dice and HD95, mean and per patient",
        ha="right",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    return fig


def table(data: dict) -> list[dict]:
    """One row per method and the base, with the change to the base."""
    base = data["base"]
    rows = [
        {
            "method": "base",
            "n_runs": base["n_runs"],
            "dice": round(base["dice"], 4),
            "hd95": round(base["hd95"], 2),
            "delta_dice": 0.0,
            "delta_hd95": 0.0,
            "uses_ground_truth": False,
        }
    ]
    for m in data["methods"]:
        rows.append(
            {
                "method": m["key"],
                "n_runs": m["n_runs"],
                "dice": round(m["dice"], 4),
                "hd95": round(m["hd95"], 2),
                "delta_dice": round(m["dice"] - base["dice"], 4),
                "delta_hd95": round(m["hd95"] - base["hd95"], 2),
                "uses_ground_truth": m["oracle"],
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metrics-dir", type=Path, default=REPO / "metrics")
    parser.add_argument("--pp-dir", type=Path, default=PP_DIR)
    parser.add_argument("--out", type=Path, default=REPO / "figures")
    args = parser.parse_args(argv)

    plt = pyplot()
    data = collect(args.metrics_dir, args.pp_dir)
    fig = figure(data, plt)
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"postprocessing.{ext}")
    write_csv(args.out / "postprocessing.csv", table(data))


if __name__ == "__main__":
    main()
