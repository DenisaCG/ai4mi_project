"""Slide-ready figures for the 2D vs 2.5D context study (ENet and residual-encoder deep-supervision U-Net).

Reads metrics/<experiment>_fold<k>/seed<s>/summary.json (read-only), 4 folds x 3 seeds = 12 runs per condition. Runs are paired
across conditions by (fold, seed): the same validation patients. "All organs" = mean of the four organs per run.

  A  forest plot of the paired 2.5D effect (2.5D variant - 2D baseline of the same architecture), for Dice, HD95 and ASSD
  B  architecture effect (U-Net 2D - ENet 2D), organs sorted by effect size, for Dice, HD95 and ASSD
  C  absolute Dice of the 2D baselines and the best 2.5D variant per architecture

Writes <out>/{a_context_effect_<metric>,b_architecture_effect_<metric>,c_summary}.{png,svg,pdf} (16:9, PNG at 300 dpi) and delta_table.csv, and prints the table.
CIs are percentile bootstrap (10000 resamples of the 12 runs, seed 0). The 12 runs are not independent (folds share patients).
"""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator
from utils import CLASSES, COLORS  # dataset_analysis/utils.py: the repo-wide organ colours

REPO = Path(__file__).resolve().parents[1]
ORGANS = [name for label, name in sorted(CLASSES.items())]
ALL = "All organs"
ROWS = [*ORGANS, ALL]
N_BOOT, BOOT_SEED = 10_000, 0
FOLDS, SEEDS = 4, 3
SIZE = (13.33, 7.5)
# key in summary.json eval.val_<key>_<organ>: axis label, higher is better, headline titles of figures A and B
METRICS = {
    "dice": {"label": "Dice", "up": True, "a": "2.5D context: no measurable gain", "b": "U-Net's advantage comes mainly from the esophagus"},
    "hd95": {"label": "HD95 (mm)", "up": False, "a": "HD95: no consistent 2.5D gain", "b": "HD95: U-Net better on esophagus, worse on heart"},
    "assd": {"label": "ASSD (mm)", "up": False, "a": "ASSD: no measurable 2.5D gain", "b": "ASSD: U-Net better on 3 organs, worse on heart"},
}

# Colour = organ (repo palette; grey for the all-organs mean, as in architecture_progress.py). Variants and architectures are told apart by marker.
ORGAN_COLOR = {name: COLORS[label] for label, name in CLASSES.items()} | {ALL: "#555555"}
GREY = "0.25"
ARCH = {
    "ENet": {
        "base": "full_cv4_enet_dice_ce",
        "variants": [
            ("full_cv4_enet_dice_ce_25d_c1", "2.5D, 3 slices (±1)", "o", True),
            ("full_cv4_enet_dice_ce_25d_c2", "2.5D, 5 slices (±2)", "D", False),
        ],
        "marker": "o",
    },
    "Residual-encoder U-Net": {
        "base": "full_cv4_resenc_ds_unet_dice_ce",
        "variants": [
            ("full_cv4_resenc_ds_unet_dice_ce_25d_c1", "2.5D, 3 slices (±1)", "o", True),
            ("full_cv4_resenc_ds_unet_dice_ce_25d_c1_s4", "2.5D, 3 slices, stride 4 (±10 mm)", "D", False),
        ],
        "marker": "s",
    },
}
ENET_BASE, UNET_BASE = ARCH["ENet"]["base"], ARCH["Residual-encoder U-Net"]["base"]


def load(metrics_dir: Path, experiment: str) -> tuple[list[tuple[int, int]], dict[str, np.ndarray]]:
    """Per-run metrics of one experiment: ((fold, seed) keys, {metric: array [runs, ORGANS + All organs]})."""
    runs = {}
    for path in sorted(metrics_dir.glob(f"{experiment}_fold*/seed*/summary.json")):
        summary = json.loads(path.read_text())
        match = re.fullmatch(rf"{re.escape(experiment)}_fold(\d+)", summary["experiment"])
        if not match:
            continue
        per_metric = {}
        for metric in METRICS:
            values = [summary["eval"][f"val_{metric}_{organ}"] for organ in ORGANS]
            per_metric[metric] = [*values, float(np.mean(values))]
        runs[int(match[1]), int(summary["seed"])] = per_metric
    if len(runs) != FOLDS * SEEDS:
        raise SystemExit(f"{experiment}: {len(runs)}/{FOLDS * SEEDS} runs found in {metrics_dir}")
    keys = sorted(runs)
    data = {m: np.array([runs[k][m] for k in keys], dtype=float) for m in METRICS}
    for m, values in data.items():
        if not np.isfinite(values).all():
            raise SystemExit(f"{experiment}: undefined (NaN or inf) {m} in some run")
    return keys, data


def boot_ci(x: np.ndarray) -> tuple[float, float]:
    """95% percentile bootstrap CI of the mean over the runs."""
    rng = np.random.default_rng(BOOT_SEED)
    means = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.grid": True,
            "axes.grid.axis": "x",
            "grid.color": "0.88",
            "grid.linewidth": 1.0,
            "pdf.fonttype": 42,
        }
    )


def row_positions() -> dict[str, float]:
    """y of each row, top to bottom, with a gap before 'All organs'."""
    return {name: y for name, y in zip(ROWS, [5.0, 4.0, 3.0, 2.0, 0.6])}


def rows_axis(ax, positions: dict[str, float], labels: bool = True) -> None:
    ax.set_yticks(list(positions.values()), [n.capitalize() for n in positions] if labels else [])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(min(positions.values()) - 0.7, max(positions.values()) + 0.7)
    ax.spines["left"].set_visible(False)


def title_and_note(fig, title: str, note: str) -> None:
    fig.text(0.015, 0.955, title, fontsize=28, fontweight="bold", ha="left", va="top")
    fig.text(0.015, 0.012, note, fontsize=14, color="0.35", ha="left", va="bottom")


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=300)
    fig.savefig(out / f"{name}.svg")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def tick_axis(ax, metric: str, dice_step: float) -> None:
    if metric == "dice":
        ax.xaxis.set_major_locator(MultipleLocator(dice_step))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))


def delta_points(ax, positions, deltas, offset, marker, hollow, jitter_rng) -> None:
    """Faint per-run paired deltas, then mean and CI per row, coloured by organ."""
    for j, name in enumerate(ROWS):
        color = ORGAN_COLOR[name]
        y = positions[name] + offset
        d = deltas[:, j]
        ax.scatter(d, y + jitter_rng.uniform(-0.05, 0.05, len(d)), s=55, marker=marker, color=color, alpha=0.28, linewidths=0, zorder=3)
        lo, hi = boot_ci(d)
        ax.errorbar(
            d.mean(), y, xerr=[[d.mean() - lo], [hi - d.mean()]], fmt=marker, ms=12, color=color,
            mfc="white" if hollow else color, mec=color, mew=2.5, elinewidth=3.5, capsize=6, capthick=3.5, zorder=5,
        )


def marker_legend(ax, entries, **kwargs) -> None:
    """Neutral legend: (label, marker, hollow) entries."""
    handles = [
        Line2D([], [], marker=m, ms=12, lw=0, color=GREY, mfc="white" if hollow else GREY, mec=GREY, mew=2.5, label=label)
        for label, m, hollow in entries
    ]
    ax.legend(handles=handles, frameon=False, handletextpad=0.3, **kwargs)


def figure_a(out: Path, data: dict, table: list[dict], metric: str) -> None:
    spec_m = METRICS[metric]
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=SIZE)
    positions = row_positions()
    rng = np.random.default_rng(1)
    for ax, (arch, spec) in zip(axes, ARCH.items()):
        base = data[spec["base"]][1]
        offsets = np.linspace(0.2, -0.2, len(spec["variants"]))
        for off, (experiment, label, marker, hollow) in zip(offsets, spec["variants"]):
            deltas = data[experiment][1] - base
            delta_points(ax, positions, deltas, off, marker, hollow, rng)
            record(table, metric, f"{arch}: {label} - 2D", deltas)
        ax.axvline(0, color="black", lw=3, zorder=2)
        rows_axis(ax, positions)
        ax.set_title(arch, fontsize=20, fontweight="bold", loc="left")
        tick_axis(ax, metric, 0.02)
        marker_legend(ax, [(v[1], v[2], v[3]) for v in spec["variants"]], loc="upper center", bbox_to_anchor=(0.5, -0.30))
    lim = max(abs(np.array(axes[0].get_xlim())).max(), abs(np.array(axes[1].get_xlim())).max())
    axes[0].set_xlim(-lim, lim)
    fig.supxlabel(f"Δ {spec_m['label']} vs the 2D baseline" + ("" if spec_m["up"] else " (lower is better)"), fontsize=20, y=0.235)
    title_and_note(
        fig,
        spec_m["a"],
        "4-fold CV × 3 seeds, paired by fold/seed, 95% bootstrap CI. Not matched: ENet 5 slices vs U-Net stride 4.",
    )
    fig.subplots_adjust(left=0.115, right=0.985, top=0.80, bottom=0.33, wspace=0.06)
    save(fig, out, f"a_context_effect_{metric}")


def figure_b(out: Path, data: dict, table: list[dict], metric: str) -> None:
    spec_m = METRICS[metric]
    deltas = data[UNET_BASE][1] - data[ENET_BASE][1]
    record(table, metric, "U-Net 2D - ENet 2D", deltas)
    sign = -1 if spec_m["up"] else 1  # the largest improvement first
    order = sorted(ORGANS, key=lambda o: sign * deltas[:, ORGANS.index(o)].mean())
    names = [*order, ALL]
    positions = {n: y for n, y in zip(names, [5.0, 4.0, 3.0, 2.0, 0.6])}
    fig, ax = plt.subplots(figsize=SIZE)
    rng = np.random.default_rng(1)
    for name in names:
        d = deltas[:, ROWS.index(name)]
        y = positions[name]
        color = ORGAN_COLOR[name]
        ax.scatter(d, y + rng.uniform(-0.1, 0.1, len(d)), s=70, color=color, alpha=0.3, linewidths=0, zorder=3)
        lo, hi = boot_ci(d)
        ax.errorbar(
            d.mean(), y, xerr=[[d.mean() - lo], [hi - d.mean()]], fmt="o", ms=15, color=color, mec="white", mew=1.5,
            elinewidth=4, capsize=7, capthick=4, zorder=5,
        )
    ax.axvline(0, color="black", lw=3, zorder=2)
    span = max(deltas.max(), 0.0) - min(deltas.min(), 0.0)
    right = max(deltas.max(), 0.0) + 0.3 * span  # room for the mean labels
    for name in names:
        ax.text(right, positions[name], f"{deltas[:, ROWS.index(name)].mean():+.3f}", fontsize=20, fontweight="bold", ha="right", va="center")
    rows_axis(ax, positions)
    ax.set_xlim(min(deltas.min(), 0.0) - 0.06 * span, right)
    tick_axis(ax, metric, 0.05)
    ax.set_xlabel(f"Δ {spec_m['label']}, U-Net 2D − ENet 2D" + ("" if spec_m["up"] else " (lower is better)"))
    title_and_note(fig, spec_m["b"], "4-fold CV × 3 seeds, paired by fold/seed, 95% bootstrap CI. Mean Δ on the right.")
    fig.subplots_adjust(left=0.13, right=0.985, top=0.84, bottom=0.17)
    save(fig, out, f"b_architecture_effect_{metric}")


def figure_c(out: Path, data: dict, table: list[dict]) -> None:
    best = {}
    for arch, spec in ARCH.items():
        best[arch] = max(spec["variants"], key=lambda v: data[v[0]][1][:, -1].mean())
    conditions = []  # (label, experiment, marker, hollow): shape = architecture, hollow = 2D, filled = best 2.5D
    for arch, spec in ARCH.items():
        short = "ENet" if arch == "ENet" else "U-Net"
        conditions.append((f"{short} 2D", spec["base"], spec["marker"], True))
        conditions.append((f"{short} {best[arch][1]}", best[arch][0], spec["marker"], False))
        print(f"best 2.5D variant, {arch}: {best[arch][1]}")
    fig, ax = plt.subplots(figsize=SIZE)
    positions = row_positions()
    offsets = np.array([0.36, 0.14, -0.14, -0.36])
    for off, (label, experiment, marker, hollow) in zip(offsets, conditions):
        values = data[experiment][1]
        for j, name in enumerate(ROWS):
            color = ORGAN_COLOR[name]
            lo, hi = boot_ci(values[:, j])
            m = values[:, j].mean()
            ax.errorbar(
                m, positions[name] + off, xerr=[[m - lo], [hi - m]], fmt=marker, ms=12, color=color,
                mfc="white" if hollow else color, mec=color, mew=2.5, elinewidth=3, capsize=5, capthick=3, zorder=5,
            )
    rows_axis(ax, positions)
    ax.set_xlim(0.62, 0.96)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel("Dice (mean of 12 runs, 95% bootstrap CI)")
    marker_legend(ax, [(label, marker, hollow) for label, _, marker, hollow in conditions], loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, columnspacing=1.5)
    title_and_note(
        fig,
        "U-Net beats ENet; 2.5D context adds nothing",
        "4-fold CV × 3 seeds, 95% bootstrap CI. 2.5D = best variant per architecture (highest all-organs mean).",
    )
    fig.subplots_adjust(left=0.13, right=0.985, top=0.84, bottom=0.30)
    save(fig, out, "c_summary")


def record(table: list[dict], metric: str, comparison: str, deltas: np.ndarray) -> None:
    for j, name in enumerate(ROWS):
        lo, hi = boot_ci(deltas[:, j])
        table.append(
            {
                "metric": metric,
                "comparison": comparison,
                "organ": name,
                "mean_delta": round(float(deltas[:, j].mean()), 4),
                "ci_low": round(lo, 4),
                "ci_high": round(hi, 4),
                "ci_excludes_0": not (lo <= 0 <= hi),
                "n_runs": len(deltas),
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics-dir", type=Path, default=REPO / "metrics")
    parser.add_argument("--out", type=Path, default=REPO / "figures/25d_presentation")
    args = parser.parse_args()

    experiments = [ARCH[a]["base"] for a in ARCH] + [v[0] for a in ARCH for v in ARCH[a]["variants"]]
    loaded = {e: load(args.metrics_dir, e) for e in experiments}
    keys = loaded[experiments[0]][0]
    for e, (k, _) in loaded.items():
        if k != keys:
            raise SystemExit(f"{e}: (fold, seed) runs differ from {experiments[0]}; cannot pair")
    args.out.mkdir(parents=True, exist_ok=True)

    style()
    table: list[dict] = []
    for metric in METRICS:
        per_metric = {e: (k, d[metric]) for e, (k, d) in loaded.items()}
        figure_a(args.out, per_metric, table, metric)
        figure_b(args.out, per_metric, table, metric)
        if metric == "dice":
            figure_c(args.out, per_metric, table)

    with open(args.out / "delta_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"\n{'metric':<7}{'comparison':<62}{'organ':<12}{'mean Δ':>9}{'95% CI':>20}  excl. 0")
    for r in table:
        ci = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
        print(f"{r['metric']:<7}{r['comparison']:<62}{r['organ']:<12}{r['mean_delta']:>+9.3f}{ci:>20}  {'yes' if r['ci_excludes_0'] else 'no'}")
    for metric in METRICS:
        context = [abs(r["mean_delta"]) for r in table if r["metric"] == metric and r["comparison"] != "U-Net 2D - ENet 2D"]
        print(f"\n{metric}: largest |mean Δ| over the 2.5D comparisons: {max(context):.3f}")
    print(f"figures + delta_table.csv -> {args.out}")


if __name__ == "__main__":
    main()
