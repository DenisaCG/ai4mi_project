"""Effect of every post-processing method on Dice and HD95 against the unprocessed model.

    python dataset_analysis/postprocessing_figure.py [--out figures]

The base model is the ResEnc deep-supervision U-Net (Dice + CE, no augmentation, 4 folds x 3 seeds).
postprocessing.{png,pdf}, the slide: one box plot per main method over its run means (the 3-seed ensembles of the votes
and the softmax rows give one point per fold), a star at their mean, a black line at their median, the marker shape
giving the fold; the mean Dice and HD95 sit to its right, with the change of HD95 to the unprocessed model in a pill
(teal: lower, brick: higher). Rows are tinted by the type of method.
postprocessing_all.{png,pdf}: every method, one row each. The line runs from the base mean to the mean of the method,
teal where the metric improves and brick where it worsens; small dots are the 40 patients (mean over the seeds and over
the 4 organs, beyond the axis drawn at its edge as triangles); the dashed line and bands are the base with its seed and
fold spread. The oracle gates organs per slice with the ground truth, so it is an upper bound (hatched row, open marker),
not a method. Methods scored on seed 0 only (CRF on seed 0, the slice-presence head) start from the base of their own
runs, marked by a small tick.
Writes also postprocessing.csv (every method).
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
# (key in the inventory, label, group, is the ground truth used)
METHODS = [
    ("gating_oracle:lcc", "Largest comp., all organs", "filter", False),
    ("gating_oracle:lcc_esophagus", "Largest comp., esophagus", "filter", False),
    ("gating_oracle:lcc_heart", "Largest comp., heart", "filter", False),
    ("gating_oracle:lcc_trachea", "Largest comp., trachea", "filter", False),
    ("gating_oracle:lcc_aorta", "Largest comp., aorta", "filter", False),
    (
        "gating_oracle:lcc_heart_trachea",
        "Largest comp., heart + trachea",
        "filter",
        False,
    ),
    (
        "gating_oracle:esophagus_size_0.2x",
        "Esophagus size filter 0.2x",
        "filter",
        False,
    ),
    (
        "gating_oracle:esophagus_size_0.05x",
        "Esophagus size filter 0.05x",
        "filter",
        False,
    ),
    (
        "gating_oracle:esophagus_size_500vox",
        "Esophagus size filter 500 vox",
        "filter",
        False,
    ),
    ("gating_oracle:area_gate_5", "Area gate 5", "filter", False),
    ("gating_oracle:area_gate_10", "Area gate 10", "filter", False),
    ("gating_oracle:area_gate_25", "Area gate 25", "filter", False),
    ("gating_oracle:area_gate_50", "Area gate 50", "filter", False),
    ("gating_oracle:adjacent_gate", "Adjacent-slice gate", "filter", False),
    ("postprocess_extra:zrun_3", "Min z-run 3", "filter", False),
    ("postprocess_extra:zrun_5", "Min z-run 5", "filter", False),
    ("postprocess_extra:zextent_0", "z-extent 0", "filter", False),
    ("postprocess_extra:zextent_5", "z-extent 5", "filter", False),
    ("postprocess_extra:slice_cc_0.1", "Per-slice component 0.1", "filter", False),
    ("postprocess_extra:slice_cc_0.25", "Per-slice component 0.25", "filter", False),
    ("postprocess_extra2:contiguous_z_0", "Contiguous z, gap 0", "filter", False),
    ("postprocess_extra2:contiguous_z_2", "Contiguous z, gap 2", "filter", False),
    ("postprocess_extra2:contiguous_z_5", "Contiguous z, gap 5", "filter", False),
    ("postprocess_extra2:trachea_anchor_20", "Trachea anchor 20", "filter", False),
    ("postprocess_extra2:trachea_anchor_40", "Trachea anchor 40", "filter", False),
    ("postprocess_extra2:trachea_anchor_60", "Trachea anchor 60", "filter", False),
    ("gating_oracle:heart_hull", "Heart convex hull", "shape", False),
    ("gating_oracle:combo", "Combination 1", "combo", False),
    ("postprocess_extra:combo", "Combination 2", "combo", False),
    ("postprocess_extra:combo_hd95", "Combination 2, HD95-tuned", "combo", False),
    ("postprocess_extra2:combo", "Combination 3", "combo", False),
    ("postprocess_extra2:combo_hd95", "Combination 3, HD95-tuned", "combo", False),
    ("seed_ensemble:seed_vote", "3-seed majority vote", "vote", False),
    (
        "seed_ensemble:seed_vote_lcc_heart_trachea",
        "Vote + largest comp., heart + trachea",
        "vote",
        False,
    ),
    ("seed_ensemble:vote_t1", "Vote, 1 of 3 seeds", "vote", False),
    ("seed_ensemble:vote_t2", "Vote, 2 of 3 seeds", "vote", False),
    ("seed_ensemble:vote_t3", "Vote, 3 of 3 seeds", "vote", False),
    ("seed_ensemble:vote_per_organ", "Vote, per-organ threshold", "vote", False),
    (
        "seed_ensemble:vote_per_organ_lcc_heart_trachea",
        "Per-organ vote + largest comp.",
        "vote",
        False,
    ),
    ("seed_ensemble:vote_per_organ_hd95", "Per-organ vote, HD95-tuned", "vote", False),
    (
        "seed_ensemble:vote_per_organ_hd95_lcc_heart_trachea",
        "HD95-tuned vote + largest comp.",
        "vote",
        False,
    ),
    ("prob_postprocess:avg", "Softmax average", "softmax", False),
    (
        "prob_postprocess:avg_lcc_heart_trachea",
        "Softmax avg + largest comp., heart + trachea",
        "softmax",
        False,
    ),
    ("prob_postprocess:avg_conf_0.6", "Softmax avg, confidence 0.6", "softmax", False),
    ("prob_postprocess:avg_conf_0.8", "Softmax avg, confidence 0.8", "softmax", False),
    ("prob_postprocess:avg_conf_0.9", "Softmax avg, confidence 0.9", "softmax", False),
    ("prob_postprocess:avg_crf_a", "Softmax avg + CRF a", "softmax", False),
    ("prob_postprocess:avg_crf_b", "Softmax avg + CRF b", "softmax", False),
    ("prob_postprocess:avg_crf_c", "Softmax avg + CRF c", "softmax", False),
    ("prob_postprocess:seed0_crf_a", "CRF a, seed 0 only", "softmax", False),
    ("prob_postprocess:seed0_crf_b", "CRF b, seed 0 only", "softmax", False),
    ("prob_postprocess:seed0_crf_c", "CRF c, seed 0 only", "softmax", False),
    ("presence:presence_head_gated", "Slice-presence head (4 runs)", "learned", False),
    ("gating_oracle:gate", "Oracle gating (upper bound)", "oracle", True),
    ("gating_oracle:gate_lcc", "Oracle gating + largest comp.", "oracle", True),
]
# The rows of the slide figure, with the two-line labels it uses.
SLIDE = {
    "gating_oracle:lcc": "Largest comp., all organs",
    "gating_oracle:lcc_heart": "Largest comp., heart only",
    "gating_oracle:lcc_heart_trachea": "Largest comp., heart+trachea",
    "gating_oracle:heart_hull": "Heart convex hull",
    "postprocess_extra2:combo_hd95": "Combination, HD95-tuned",
    "seed_ensemble:seed_vote": "3-seed majority vote",
    "seed_ensemble:seed_vote_lcc_heart_trachea": "Vote + largest comp.",
    "prob_postprocess:avg_lcc_heart_trachea": "Softmax avg + largest comp.",
    "prob_postprocess:avg_crf_a": "Softmax avg + CRF a",
}
# Group name on the figure and background colour of its rows.
GROUPS = {
    "filter": ("Filter", tint(PALETTE[0], 0.22)),
    "shape": ("Shape", tint(PALETTE[3], 0.3)),
    "combo": ("Combo", tint(PALETTE[5], 0.3)),
    "vote": ("Vote", tint(PALETTE[1], 0.25)),
    "softmax": ("Softmax", tint(PALETTE[6], 0.4)),
    "learned": ("Learned", tint(PALETTE[4], 0.3)),
    "oracle": ("Oracle", "none"),
}
TEXT = "#1A1A1A"
VALUE = "#0A0A0A"
# DejaVu Sans has no semi-bold; a thin stroke in the text colour gives the in-between weight.
SEMIBOLD = [pe.withStroke(linewidth=0.3, foreground=VALUE)]
BASE_LABEL = "Unprocessed base model"
# (fill, line) colours of the rows of each group in the box-plot figure.
BOX_COLORS = {
    "base": ("#AAAAAA", "#555555"),
    "filter": (PALETTE[0], "#3F6A8F"),
    "shape": (PALETTE[3], "#5E9142"),
    "combo": (PALETTE[5], EARTH[0]),
    "vote": (PALETTE[1], "#C77A45"),
    "softmax": (PALETTE[6], "#B48A2E"),
}
FOLD_MARKERS = "osD^"  # one per fold
BETTER, WORSE = EARTH[1], EARTH[0]  # teal, brick
DICE_LIM, HD95_LIM = (0.68, 0.97), (0, 40)
FOLD_BAND = "#8A8272"
JITTER = 0.3  # half-height of the patient dot cloud in a row


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
    """Base model from the local runs and every method from the post-processing inventory.

    Args:
        metrics_dir: Folder with the per-run metrics_3d.csv of the base experiment.
        pp_dir: Folder with the inventory's pp_summary*.csv and pp_per_patient*.csv.

    Returns:
        The base summary (dice, hd95, seed and fold stds, per-patient values) and one dict per method of METHODS with
        its label, group, oracle flag, run count, means, the base of its own runs and its per-patient values.
    """
    runs = load_cv_runs(metrics_dir, BASE_EXPERIMENT, METRICS)
    base = summarise(runs)
    summary, cells = {}, defaultdict(list)
    for name, tag in (("", None), ("_from_rows", "rows")):
        for r in read_csv(pp_dir / f"pp_summary{name}.csv"):
            if r["organ"] == "mean":
                folder = (
                    r["root"].split("results/")[1].split("/")[0] if tag else "presence"
                )
                summary[f"{folder}:{r['method']}"] = r
        for r in read_csv(pp_dir / f"pp_per_patient{name}.csv"):
            cells[r["method"] if tag else f"presence:{r['method']}"].append(r)
    methods = []
    for key, label, group, oracle in METHODS:
        row, rows = summary[key], cells[key]
        methods.append(
            {
                "key": key,
                "label": label,
                "group": group,
                "oracle": oracle,
                "n_runs": int(row["n_runs"]),
                "dice": float(row["dice"]),
                "hd95": float(row["hd95"]),
                "own_base_dice": float(row["base_dice"]),
                "own_base_hd95": float(row["base_hd95"]),
                "patient_dice": patient_values(rows, "dice"),
                "patient_hd95": patient_values(rows, "hd95"),
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


def run_means(rows: list[dict], column: str) -> dict[tuple[str, str], float]:
    """Mean over the patients of a run of their mean over the organs, for every run of a method.

    Args:
        rows: Per-patient rows of one method from the inventory.
        column: Metric column to average (blank cells are skipped).

    Returns:
        (fold, seed label) to value; the seed label is "vote" or "avg" for the fold-level ensembles.
    """
    cells = defaultdict(list)
    for r in rows:
        if r[column]:
            cells[r["fold"], r["seed"], r["patient"], r["organ"]].append(
                float(r[column])
            )
    patients = defaultdict(list)
    for (fold, seed, patient, _), values in cells.items():
        patients[fold, seed, patient].append(statistics.fmean(values))
    runs = defaultdict(list)
    for (fold, seed, _), values in patients.items():
        runs[fold, seed].append(statistics.fmean(values))
    return {run: statistics.fmean(v) for run, v in runs.items()}


def change_text(change: float) -> str:
    """The change of a metric to the base as a signed number with a true minus sign.

    Args:
        change: Value of the method minus the base's.

    Returns:
        For example "\u22121.8" or "+4.5", to one decimal.
    """
    return f"{change:+.1f}".replace("-", "\u2212")


def box_figure(data: dict, plt):
    """The slide figure: one box plot of the run means per method, with Dice and HD95 beside it.

    Args:
        data: Output of collect().
        plt: matplotlib.pyplot.

    Returns:
        The matplotlib figure. Rows are the base model and the methods of SLIDE; each box covers the run means (the
        folds' ensembles for the votes and softmax rows), a star marks their mean, a black line their median.
    """
    base = data["base"]
    first = data["methods"][0]["rows"]
    rows = [
        {
            "label": BASE_LABEL,
            "group": "base",
            "runs": run_means([{**r, "dice": r["base_dice"]} for r in first], "dice"),
            "dice": base["dice"],
            "hd95": base["hd95"],
        }
    ]
    for m in data["methods"]:
        if m["key"] in SLIDE:
            runs = run_means(m["rows"], "dice")
            rows.append({**m, "label": SLIDE[m["key"]], "runs": runs})
    n = len(rows)
    fig, ax = plt.subplots(figsize=(7.9, 4.85))
    fig.subplots_adjust(left=0.33, right=0.74, top=0.84, bottom=0.2)
    for i, r in enumerate(rows):
        y = n - 1 - i
        fill, line = BOX_COLORS[r["group"]]
        ax.axhspan(y - 0.5, y + 0.5, color=tint(fill, 0.22), lw=0, zorder=0)
        if i and rows[i - 1]["group"] != r["group"]:
            ax.axhline(y + 0.5, color="#666666", lw=0.8, zorder=1)
        values = list(r["runs"].values())
        ax.boxplot(
            values, positions=[y], vert=False, widths=0.5, patch_artist=True,
            showfliers=False, manage_ticks=False,
            boxprops={"facecolor": tint(fill, 0.55), "edgecolor": line, "lw": 1.2},
            whiskerprops={"color": line, "lw": 1.2}, capprops={"color": line, "lw": 1.2},
            medianprops={"color": "#111111", "lw": 1.4}, zorder=2,
        )  # fmt: skip
        for k, ((fold, _), v) in enumerate(sorted(r["runs"].items())):
            ax.scatter(
                v, y + (k % 3 - 1) * 0.12, marker=FOLD_MARKERS[int(fold)], s=14,
                color=line, alpha=0.85, lw=0, zorder=3,
            )  # fmt: skip
        ax.scatter(
            statistics.fmean(values), y, marker="*", s=130, color="white",
            edgecolor="#222222", lw=0.9, zorder=5,
        )  # fmt: skip
        for x, text in ((1.09, f"{r['dice']:.3f}"), (1.27, f"{r['hd95']:.1f}")):
            ax.text(
                x, y, text, transform=ax.get_yaxis_transform(), ha="center",
                va="center", fontsize=11, fontweight="bold", color=line,
            )  # fmt: skip
        if r["group"] != "base":
            change = r["hd95"] - base["hd95"]
            ax.text(
                1.46, y, change_text(change), transform=ax.get_yaxis_transform(),
                ha="center", va="center", fontsize=10.5, fontweight="bold", color="white",
                bbox={
                    "boxstyle": "round,pad=0.25", "ec": "none",
                    "fc": BETTER if change < 0 else WORSE,
                },
            )  # fmt: skip
    ax.axvline(base["dice"], color="#444444", lw=1.0, ls="--", zorder=1)
    ax.set_xlim(0.82, 0.90)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks([0.82, 0.84, 0.86, 0.88])
    ax.set_yticks(
        range(n), [r["label"] for r in reversed(rows)], fontsize=11, color=TEXT
    )
    ax.grid(axis="x", color="#D0D0D0", lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=11, labelcolor=TEXT)
    ax.set_xlabel(
        "Dice (3D, original CT grid), one point per run; votes, softmax: one per fold",
        fontsize=11, color=TEXT,
    )  # fmt: skip
    ax.set_title(
        "Dice, mean of four organs", fontsize=11, fontweight="bold", loc="left"
    )
    for x, name in ((1.09, "Dice"), (1.27, "HD95\n(mm)"), (1.46, "\u0394 HD95")):
        ax.text(
            x, n - 0.35, name, transform=ax.get_yaxis_transform(), ha="center",
            va="bottom", fontsize=11, fontweight="bold", color=TEXT,
        )  # fmt: skip
    fig.text(
        0.012,
        0.5,
        "Method",
        rotation=90,
        ha="left",
        va="center",
        fontsize=12,
        color=TEXT,
    )
    handles = [
        plt.Line2D([], [], marker=m, color="#555555", ls="", ms=5, label=f"fold {k}")
        for k, m in enumerate(FOLD_MARKERS)
    ]
    handles += [
        plt.Line2D([], [], marker="*", mfc="white", mec="#222222", color="#222222", ls="", ms=11, label="mean"),
        plt.Line2D([], [], color="#111111", lw=1.4, label="median"),
        plt.Line2D([], [], color="#444444", ls="--", label="base mean"),
    ]  # fmt: skip
    fig.legend(
        handles=handles, loc="lower center", ncol=len(handles), fontsize=11, frameon=False,
        handletextpad=0.2, columnspacing=0.6, labelcolor=TEXT,
    )  # fmt: skip
    fig.text(
        0.99, 0.985, "Run means by post-processing method", ha="right", va="top",
        fontsize=14, fontweight="bold",
    )  # fmt: skip
    return fig


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


def panel(ax, data: dict, metric: str, limits, higher_is_better: bool) -> None:
    """One metric of the figure of every method: patient dots, the base reference, and a line per method."""
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
        start = m[f"own_base_{metric}"]
        background = GROUPS[m["group"]][1]
        if background != "none":
            ax.axhspan(y - 0.5, y + 0.5, color=background, lw=0, zorder=-1)
        if m["oracle"]:
            ax.axhspan(
                y - 0.5, y + 0.5, facecolor="none", hatch="///", edgecolor=GRID_COLOR, lw=0
            )  # fmt: skip
        improved = (m[metric] > start) == higher_is_better
        color = BETTER if improved else WORSE
        draw_cloud(ax, m[f"patient_{metric}"], y, limits, color)
        if abs(start - base[metric]) > 1e-3 * (limits[1] - limits[0]):
            ax.scatter(start, y, marker="|", s=90, color="#444444", lw=1.5, zorder=3)
        ax.plot(
            [start, m[metric]], [y, y], color=color, lw=2,
            ls="--" if m["oracle"] else "-", solid_capstyle="butt", zorder=4,
        )  # fmt: skip
        ax.scatter(
            min(max(m[metric], limits[0]), limits[1]), y, s=24, zorder=5,
            color="white" if m["oracle"] else color, edgecolor=color, lw=1.5,
        )  # fmt: skip
    ax.set_xlim(*limits)
    ax.set_ylim(-0.6, n + 0.7)
    ax.grid(axis="x", color=GRID_COLOR, lw=0.8, ls=(0, (3, 3)))
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def all_methods_figure(data: dict, plt):
    """The tall two-panel figure listing every method, with its Dice and HD95 in columns beside the panels.

    Args:
        data: Output of collect().
        plt: matplotlib.pyplot.

    Returns:
        The matplotlib figure.
    """
    n = len(data["methods"])
    size = (11.5, 0.215 * (n + 1) + 2.0)
    fig, (ax_d, ax_h) = plt.subplots(
        1, 2, figsize=size, sharey=True, gridspec_kw={"wspace": 0.08}
    )
    panel(ax_d, data, "dice", DICE_LIM, True)
    panel(ax_h, data, "hd95", HD95_LIM, False)
    ax_d.set_xticks([0.7, 0.8, 0.9])
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
        plt.Line2D([], [], marker="o", mfc="white", mec=BETTER, mew=2, ls="--", color=BETTER, label="oracle"),
        plt.Line2D([], [], marker="o", color="#777777", alpha=0.6, ls="", ms=4, label="patient"),
        plt.Rectangle((0, 0), 1, 1, fc=FOLD_BAND, alpha=0.34, label="seed \u00b11 SD"),
        plt.Rectangle((0, 0), 1, 1, fc=FOLD_BAND, alpha=0.18, label="fold \u00b11 SD"),
    ]  # fmt: skip
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.55, 0.0), ncol=len(handles),
        fontsize=11, frameon=False, labelcolor=TEXT, handletextpad=0.3, columnspacing=0.9,
        handlelength=1.4,
    )  # fmt: skip
    fig.subplots_adjust(
        left=0.41, right=0.88, bottom=0.75 / size[1] + 0.04, top=1 - 0.55 / size[1], wspace=0.1
    )  # fmt: skip
    rows = [m["group"] for m in data["methods"]]
    to_fig = ax_d.transData + fig.transFigure.inverted()
    for group in dict.fromkeys(rows):
        top = n - 1 - rows.index(group)
        fig.text(
            0.01, to_fig.transform((0, top))[1], GROUPS[group][0], ha="left",
            va="center", fontsize=11, fontweight="bold", color=TEXT,
        )  # fmt: skip
    for x, name in ((1.04, "Dice"), (1.24, "HD95")):
        ax_h.text(x, n + 0.2, name, transform=ax_h.get_yaxis_transform(), ha="left", va="center", fontsize=11, color=TEXT)  # fmt: skip
    for i, m in enumerate(data["methods"]):
        for x, v in ((1.04, f"{m['dice']:.3f}"), (1.24, f"{m['hd95']:.1f}")):
            ax_h.text(x, n - 1 - i, v, transform=ax_h.get_yaxis_transform(), ha="left", va="center", fontsize=10, color=VALUE)  # fmt: skip
    fig.text(
        0.985, 0.985, "All post-processing methods tried", ha="right", va="top",
        fontsize=14, fontweight="bold",
    )  # fmt: skip
    return fig


def table(data: dict) -> list[dict]:
    """One row per method and the base, with the change to the base of the method's own runs."""
    base = data["base"]
    rows = [
        {
            "method": "base",
            "group": "",
            "n_runs": base["n_runs"],
            "dice": round(base["dice"], 4),
            "hd95": round(base["hd95"], 2),
            "delta_dice": 0.0,
            "delta_hd95": 0.0,
            "uses_ground_truth": False,
            "on_slide": False,
        }
    ]
    for m in data["methods"]:
        rows.append(
            {
                "method": m["key"],
                "group": m["group"],
                "n_runs": m["n_runs"],
                "dice": round(m["dice"], 4),
                "hd95": round(m["hd95"], 2),
                "delta_dice": round(m["dice"] - m["own_base_dice"], 4),
                "delta_hd95": round(m["hd95"] - m["own_base_hd95"], 2),
                "uses_ground_truth": m["oracle"],
                "on_slide": m["key"] in SLIDE,
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
    for fig, name in (
        (box_figure(data, plt), "postprocessing"),
        (all_methods_figure(data, plt), "postprocessing_all"),
    ):
        for ext in ("png", "pdf"):
            fig.savefig(args.out / f"{name}.{ext}")
    write_csv(args.out / "postprocessing.csv", table(data))


if __name__ == "__main__":
    main()
