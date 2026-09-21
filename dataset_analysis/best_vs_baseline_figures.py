"""Five chart options comparing the best pipeline against the baseline, on every 3D metric.

Baseline: native spacing, per-volume min-max, 256x256, plain cross-entropy (9 seeds).
Best:     median spacing + HU window/z-score + ROI crop (288x288) + CE(all classes) + Dice(foreground) loss (3 seeds).
Corrected SEGTHOR, validation fold 0 (5 patients), 3D metrics on the original CT grid, read from metrics/<experiment>/.

Options: 1 grouped bars per organ, 2 dumbbell (before -> after), 3 error-reduction heatmap, 4 per-patient slopes,
5 ablation ladder. Titles are built from the numbers, so a change in the data changes the wording. Read-only on
metrics/; writes plots/ and tables/ under dataset_analysis/results/best_vs_baseline/. Light enough for a login node.

    python dataset_analysis/best_vs_baseline_figures.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from utils import INK, MUTED, REPO, clean_axis, decorate, pyplot, tint
from plot_style import EARTH, legend_below
from matplotlib import patheffects as pe
from matplotlib.colors import to_rgb

BASELINE_EXP = "segthor_enet_ce_repro_corrected_fixed"
BEST_EXP = "segthor_enet_dice_ce_all_corrected_ct_window_zscore_median_spacing_roi_crop"
STACK = "segthor_enet_ce_corrected_ct_window_zscore_median_spacing"
LADDER = [("Baseline", BASELINE_EXP),
          ("+ HU window\n+ z-score", "segthor_enet_ce_corrected_ct_window_zscore"),
          ("+ median\nspacing", STACK),
          ("+ ROI crop", STACK + "_roi_crop"),
          ("+ CE + Dice\nloss", BEST_EXP)]
ORGANS = ["esophagus", "heart", "trachea", "aorta"]
GROUPS = ORGANS + ["average"]
LABELS = [o.capitalize() for o in ORGANS] + ["Average"]
METRICS = {"dice": ("Dice", "higher is better", ""), "hd95": ("HD95", "lower is better", " mm"),
           "assd": ("ASSD", "lower is better", " mm")}

BASE_C = "#9A9184"  # baseline: the de-emphasis warm gray
BEST_C = EARTH[1]  # best pipeline: the project's deep teal
DUMBBELL_BEST = "#5B3D8C"  # option 2 only: dark purple for the best pipeline (validated: inside the lightness band, dE >= 17 to the line)
DUMBBELL_IMPROVED = "#9370C0"  # option 2 only: purple for an improved row
WORSE_C = EARTH[0]  # brick, only for "got worse" (lines, heatmap cells)
NEUTRAL = "#EFECE6"  # heatmap midpoint: nothing changed
MINUS = "−"


def nanmean(values) -> float:
    a = np.asarray(list(values), dtype=float)
    a = a[~np.isnan(a)]
    return float(a.mean()) if a.size else float("nan")


def load_arm(exp: str) -> dict:
    """Per seed: mean over the val patients of each (metric, organ), as evaluate.py does; plus per-patient Dice."""
    files = sorted((REPO / "metrics" / exp).glob("seed*/metrics_3d.csv"))
    if not files:
        raise FileNotFoundError(f"no metrics_3d.csv under metrics/{exp}")
    per_seed = {}
    for f in files:
        d: dict = {}
        with f.open() as fh:
            for r in csv.DictReader(fh):
                for m in METRICS:
                    d.setdefault((m, r["class_name"]), {})[r["patient"]] = float(r[m]) if r[m] not in ("", "nan") else math.nan
        per_seed[f.parent.name] = d
    seeds = list(per_seed)
    value = {}
    for m in METRICS:
        for o in ORGANS:
            value[(m, o)] = np.array([nanmean(per_seed[s][(m, o)].values()) for s in seeds])
        value[(m, "average")] = np.array([nanmean(value[(m, o)][i] for o in ORGANS) for i in range(len(seeds))])
    patients = sorted(per_seed[seeds[0]][("dice", "heart")])
    patient = {o: {p: nanmean(per_seed[s][("dice", o)][p] for s in seeds) for p in patients} for o in ORGANS}
    empty = sum(math.isnan(v) for s in seeds for o in ORGANS for v in per_seed[s][("hd95", o)].values())
    arm = {"exp": exp, "seeds": seeds, "value": value, "patient": patient, "empty": empty,
           "predictions": len(seeds) * len(ORGANS) * len(patients)}
    for i, s in enumerate(seeds):  # the recomputation must agree with what evaluate.py stored
        stored = json.loads((REPO / "metrics" / exp / s / "summary.json").read_text())["eval"]
        for m in METRICS:
            assert abs(value[(m, "average")][i] - stored[f"val_{m}_fg"]) < 1e-6, (exp, s, m)
    # A seed whose model never predicted an organ has no HD95/ASSD for it (NaN for every patient): that seed is absent
    # from that organ's bars and dots, as evaluate.py leaves it out of the mean.
    arm["value"] = {k: a[~np.isnan(a)] for k, a in value.items()}
    return arm


def summary_values(exp: str) -> dict:
    """Foreground (average over the 4 organs) Dice/HD95/ASSD per seed, straight from summary.json."""
    runs = [json.loads(f.read_text())["eval"] for f in sorted((REPO / "metrics" / exp).glob("seed*/summary.json"))]
    return {m: np.array([r[f"val_{m}_fg"] for r in runs]) for m in METRICS}


def error_reduction(b: float, t: float, metric: str) -> float:
    """Share of the baseline error removed: 1 - Dice for Dice, the distance itself for HD95/ASSD. Negative = worse."""
    return ((1 - b) - (1 - t)) / (1 - b) if metric == "dice" else (b - t) / b


def build_rows(base: dict, best: dict) -> list[dict]:
    rows = []
    for m in METRICS:
        for g in GROUPS:
            b, t = base["value"][(m, g)], best["value"][(m, g)]
            rows.append({"metric": m, "group": g, "baseline": float(b.mean()), "best": float(t.mean()),
                         "delta": float(t.mean() - b.mean()), "error_reduction": error_reduction(b.mean(), t.mean(), m),
                         "baseline_sd": float(b.std()), "best_sd": float(t.std()), "n_baseline": len(b), "n_best": len(t)})
    return rows


def fmt(metric: str, v: float, signed: bool = False) -> str:
    unit = METRICS[metric][2]
    s = f"{abs(v):.2f}" if metric == "dice" else f"{abs(v):.1f}"
    sign = ("+" if v >= 0 else MINUS) if signed else (MINUS if v < 0 else "")
    return f"{sign}{s}{unit}"


def worse_cells(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["group"] != "average" and r["error_reduction"] < 0]


def metric_title(m: str) -> str:
    name, direction, unit = METRICS[m]
    return f"{name}{' (mm)' if unit else ''} — {direction}"


def phrase(cells: list[dict]) -> str:
    return " and ".join(f"{c['group']} {METRICS[c['metric']][0]}" for c in cells)


def footnote(base: dict, best: dict, ce_stack: dict, extra: str = "") -> str:
    heart_ce = float(ce_stack["value"][("hd95", "heart")].mean())
    return (f"Corrected SEGTHOR labels, validation fold 0 (5 patients), 3D metrics on the original CT grid. Baseline = native spacing, "
            f"min-max, 256x256, CE loss ({len(base['seeds'])} seeds); best = median spacing + HU window/z-score + ROI crop + CE(all)+Dice(fg) "
            f"loss ({len(best['seeds'])} seeds). HD95/ASSD leave out organs predicted empty ({base['empty']} of {base['predictions']} baseline "
            f"organ predictions, {best['empty']} of {best['predictions']} for the best pipeline), which flatters the baseline. "
            f"The heart HD95 rise comes from the Dice loss, not the crop: the same stack with plain CE gets {heart_ce:.0f} mm. {extra}").strip()


def swatch(ax, color, label: str) -> None:
    """Legend proxy: a colored square (an empty bar() has no patch, so its legend entry falls back to the default color)."""
    ax.plot([], [], linestyle="", marker="s", markersize=11, color=color, label=label)


def label_at(ax, x: float, y: float, text: str) -> None:
    """Direct label just above a mark, with a white halo so it stays readable over gridlines and seed dots."""
    ax.annotate(text, (x, y), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=9, color=INK,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")], zorder=6)


def contrast(a, b) -> float:
    def lum(c):
        r, g, bl = (v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in c[:3])
        return 0.2126 * r + 0.7152 * g + 0.0722 * bl
    hi, lo = sorted((lum(to_rgb(a)), lum(to_rgb(b))), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def option_1(base, best, rows, notes, out: Path):
    """Grouped bars: one panel per metric (own axis, never dual), organs on x, dots = single seeds."""
    plt = pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    x, w = np.arange(len(GROUPS)), 0.34
    for ax, m in zip(axes, METRICS):
        top = 0.0
        for j, (arm, color) in enumerate(((base, BASE_C), (best, BEST_C))):
            xs = x + (j - 0.5) * (w + 0.05)
            ax.bar(xs, [arm["value"][(m, g)].mean() for g in GROUPS], w, color=color, zorder=2)
            for xi, g in zip(xs, GROUPS):
                v = arm["value"][(m, g)]
                ax.scatter(xi + np.linspace(-0.09, 0.09, len(v)) * (len(v) > 1), v, s=14, color=INK, edgecolor="white", linewidth=0.8, zorder=4)
                top = max(top, v.max())
                if j == 1:  # label only the best pipeline's bar (unit is in the panel title); the table has the rest
                    label_at(ax, xi, max(v.max(), v.mean()), fmt(m, v.mean()).replace(" mm", ""))
        ax.set_ylim(0, 1.05 if m == "dice" else top * 1.15)
        ax.set_xticks(x, LABELS)
        ax.set_title(metric_title(m), fontsize=12, color="black", loc="left")
        clean_axis(ax)
    swatch(axes[0], BASE_C, "Baseline")
    swatch(axes[0], BEST_C, "Best pipeline")
    axes[0].scatter([], [], s=16, color=INK, label="One seed")
    legend_below(axes[0], ncol=3)
    decorate(fig, f"Best pipeline improves {notes['n_ok']} of {notes['n_all']} organ-metric results; only {phrase(notes['worse']) or 'none'} gets worse",
             subtitle="Mean over seeds (bars) with each seed as a dot; one axis per metric.",
             footnote_text=notes["footnote"])
    fig.savefig(out / "option_1_grouped_bars.png")
    plt.close(fig)


def option_2(base, best, rows, notes, out: Path):
    """Dumbbell: baseline -> best per organ and metric, line colored by whether it got better or worse."""
    plt = pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
    ypos = np.array([0, 1, 2, 3, 4.6])
    for ax, m in zip(axes, METRICS):
        hi = max(max(base["value"][(m, g)].mean(), best["value"][(m, g)].mean()) for g in GROUPS)
        for y, g in zip(ypos, GROUPS):
            b, t = base["value"][(m, g)].mean(), best["value"][(m, g)].mean()
            worse = error_reduction(b, t, m) < 0
            ax.plot([b, t], [y, y], color=tint(WORSE_C, 0.5) if worse else DUMBBELL_IMPROVED, linewidth=4, solid_capstyle="round", zorder=2)
            ax.scatter([b], [y], s=95, color=BASE_C, edgecolor="white", linewidth=1.5, zorder=3, marker="o")
            ax.scatter([t], [y], s=110, color=DUMBBELL_BEST, edgecolor="white", linewidth=1.5, zorder=4, marker="D")
            ax.annotate(fmt(m, t - b, signed=True) + (" (worse)" if worse else ""), (max(b, t), y), xytext=(10, 0), textcoords="offset points",
                        va="center", fontsize=10, color=INK)
        ax.set_xlim(0, 1.32 if m == "dice" else hi * 1.42)
        if m == "dice":
            ax.set_xticks(np.arange(0, 1.01, 0.2))  # Dice cannot exceed 1; the room to the right is for the labels
        ax.set_yticks(ypos, LABELS)
        ax.set_title(metric_title(m), fontsize=12, color="black", loc="left")
        clean_axis(ax, "x")
    axes[0].invert_yaxis()  # shared y axis: organs read top to bottom, inverted once
    axes[0].scatter([], [], s=95, color=BASE_C, label="Baseline", marker="o")
    axes[0].scatter([], [], s=110, color=DUMBBELL_BEST, label="Best pipeline", marker="D")
    axes[0].plot([], [], color=DUMBBELL_IMPROVED, linewidth=4, label="Improved")
    axes[0].plot([], [], color=tint(WORSE_C, 0.5), linewidth=4, label="Got worse")
    legend_below(axes[0], ncol=4)
    ok = [METRICS[m][0] for m in METRICS if not [c for c in notes["worse"] if c["metric"] == m]]
    bad = [c for c in notes["worse"]]
    decorate(fig, (f"{' and '.join(ok)} improve for every organ; " if ok else "") + (f"{phrase(bad)} is the exception" if bad else "every metric improves"),
             subtitle="Each row moves from the baseline (circle) to the best pipeline (diamond); the number is the change in the metric's own units.",
             footnote_text=notes["footnote"])
    fig.savefig(out / "option_2_dumbbell.png")
    plt.close(fig)


def option_3(base, best, rows, notes, out: Path):
    """Heatmap of the share of baseline error removed, diverging around zero (teal = better, brick = worse)."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
    from matplotlib.patches import Rectangle

    plt = pyplot()
    cmap = LinearSegmentedColormap.from_list("better_worse", [WORSE_C, NEUTRAL, BEST_C])
    norm = TwoSlopeNorm(vcenter=0, vmin=-100, vmax=100)
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    by = {(r["metric"], r["group"]): r for r in rows}
    for i, m in enumerate(METRICS):
        for j, g in enumerate(GROUPS):
            r = by[(m, g)]
            rgb = cmap(norm(100 * r["error_reduction"]))
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=rgb, edgecolor="white", linewidth=3, zorder=2))
            ink = INK if contrast(rgb, INK) >= contrast(rgb, "white") else "white"
            ax.text(j + 0.5, i + 0.42, f"{100 * r['error_reduction']:+.0f}%".replace("-", MINUS), ha="center", va="center", fontsize=17, fontweight="bold", color=ink, zorder=3)
            ax.text(j + 0.5, i + 0.70, f"{fmt(m, r['baseline'])} → {fmt(m, r['best'])}", ha="center", va="center", fontsize=9.5, color=ink, zorder=3)
    ax.set_xlim(0, len(GROUPS))
    ax.set_ylim(len(METRICS), 0)
    ax.set_xticks(np.arange(len(GROUPS)) + 0.5, LABELS)
    ax.set_yticks(np.arange(len(METRICS)) + 0.5, [f"{METRICS[m][0]}\n({'error = 1 − Dice' if m == 'dice' else 'mm'})" for m in METRICS])
    ax.xaxis.tick_top()
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, pad=8)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.06, pad=0.06, aspect=40, shrink=0.55)
    cb.set_ticks([-100, -50, 0, 50, 100])
    cb.set_ticklabels([f"{MINUS}100%", f"{MINUS}50%", "0", "+50%", "+100%"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    cb.set_label("Error reduction vs baseline (brick = worse, teal = better)", fontsize=10, color=MUTED)
    good = [100 * r["error_reduction"] for r in rows if r["group"] != "average" and r["error_reduction"] > 0]
    bad = notes["worse"]
    title = f"Baseline error falls by {min(good):.0f}–{max(good):.0f}% in {len(good)} of {notes['n_all']} organ-metric cells"
    if bad:
        title += "; " + "; ".join(f"{c['group']} {METRICS[c['metric']][0]} rises {abs(100 * c['error_reduction']):.0f}%" for c in bad)
    decorate(fig, title, subtitle="Error = 1 − Dice for Dice, the distance itself for HD95 and ASSD; each cell also shows baseline → best.",
             footnote_text=notes["footnote"])
    fig.savefig(out / "option_3_error_reduction_heatmap.png")
    plt.close(fig)


def option_4(base, best, rows, notes, out: Path):
    """Per-patient slopes of Dice: does every validation patient improve, not just the mean?"""
    plt = pyplot()
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.4), sharey=True)
    up = total = 0
    for ax, o in zip(axes, ORGANS):
        pats = sorted(base["patient"][o])
        gains = {}
        for p in pats:
            b, t = base["patient"][o][p], best["patient"][o][p]
            gains[p] = t - b
            up += t > b
            total += 1
            ax.plot([0, 1], [b, t], color=tint(BEST_C if t > b else WORSE_C, 0.45), linewidth=2, zorder=2)
            ax.scatter([0], [b], s=55, color=BASE_C, edgecolor="white", linewidth=1.2, zorder=3, marker="o")
            ax.scatter([1], [t], s=60, color=BEST_C, edgecolor="white", linewidth=1.2, zorder=4, marker="D")
        mb, mt = np.mean([base["patient"][o][p] for p in pats]), np.mean([best["patient"][o][p] for p in pats])
        ax.plot([0, 1], [mb, mt], color=INK, linewidth=3.2, zorder=5, solid_capstyle="round")
        picks = sorted({max(gains, key=gains.get), min(gains, key=gains.get)}, key=lambda q: best["patient"][o][q])  # selective labels
        ys = [best["patient"][o][q] for q in picks]
        ys_text = ys if len(ys) < 2 or ys[1] - ys[0] >= 0.09 else [sum(ys) / 2 - 0.045, sum(ys) / 2 + 0.045]  # nudge apart
        for q, yt in zip(picks, ys_text):
            ax.annotate(f"{q.replace('Patient_', 'P')} {gains[q]:+.2f}".replace("-", MINUS), (1, best["patient"][o][q]), xytext=(1.14, yt),
                        textcoords="data", va="center", fontsize=9.5, color=INK,
                        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7, shrinkA=0, shrinkB=5))
        ax.set_xlim(-0.25, 1.75)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 1], ["Baseline", "Best"])
        ax.set_title(f"{o.capitalize()}  (mean {mb:.2f} → {mt:.2f})", fontsize=12, color="black", loc="left")
        clean_axis(ax)
    axes[0].set_ylabel("Dice (higher is better)")
    axes[0].scatter([], [], s=55, color=BASE_C, label="Baseline", marker="o")
    axes[0].scatter([], [], s=60, color=BEST_C, label="Best pipeline", marker="D")
    axes[0].plot([], [], color=tint(BEST_C, 0.45), linewidth=2, label="One patient (mean over seeds)")
    axes[0].plot([], [], color=INK, linewidth=3.2, label="Mean over patients")
    legend_below(axes[0], ncol=4)
    title = (f"Dice improves for every one of the {total} patient–organ pairs" if up == total
             else f"Dice improves for {up} of {total} patient–organ pairs")
    decorate(fig, title, subtitle="Each line is one validation patient, averaged over seeds; labels mark the largest and smallest gain per organ.",
             footnote_text=notes["footnote"])
    fig.savefig(out / "option_4_per_patient_slopes.png")
    plt.close(fig)
    notes["patient_pairs"] = (int(up), int(total))


def option_5(notes, out: Path):
    """Ablation ladder: each step adds one thing to the previous experiment; foreground metrics, one panel each."""
    plt = pyplot()
    steps = [(name, summary_values(exp)) for name, exp in LADDER]
    mean = {m: [v[m].mean() for _, v in steps] for m in METRICS}
    d = {m: np.diff(mean[m]) for m in METRICS}
    # the title below states these; fail loudly if new results contradict it
    assert d["dice"][0] > 0.15 and d["dice"][2] > 0.15 and d["dice"][1] < 0 and d["hd95"][1] < -10 and d["hd95"][3] > 5, (
        "results changed: reword the option 5 title", {m: d[m].round(3).tolist() for m in METRICS})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    colors = [BASE_C] + [tint(BEST_C, 0.5)] * 3 + [BEST_C]
    x = np.arange(len(steps))
    for ax, m in zip(axes, METRICS):
        top = max(v[m].max() for _, v in steps)
        ax.bar(x, mean[m], 0.5, color=colors, zorder=2)
        for i, (_, v) in enumerate(steps):
            ax.scatter(i + np.linspace(-0.1, 0.1, len(v[m])), v[m], s=14, color=INK, edgecolor="white", linewidth=0.8, zorder=4)
            if i:  # the step's change against the previous experiment
                label_at(ax, i, max(v[m].max(), mean[m][i]), fmt(m, mean[m][i] - mean[m][i - 1], signed=True))
        ax.set_ylim(0, 1.0 if m == "dice" else top * 1.18)
        ax.set_xticks(x, [n for n, _ in steps], fontsize=10)
        ax.set_title(f"Average over organs: {metric_title(m)}", fontsize=12, color="black", loc="left")
        clean_axis(ax)
    ax0 = axes[0]
    swatch(ax0, BASE_C, "Baseline (9 seeds)")
    swatch(ax0, tint(BEST_C, 0.5), "Cumulative steps (3 seeds each)")
    swatch(ax0, BEST_C, "Best pipeline")
    ax0.scatter([], [], s=16, color=INK, label="One seed")
    legend_below(ax0, ncol=4)
    decorate(fig, "Window + z-score and the crop each add about 0.2 Dice; median spacing nearly halves HD95, and the Dice loss gives some of it back",
             subtitle="Each bar adds one step to the previous experiment (the number is its change); the crop requires median spacing.",
             footnote_text="Corrected SEGTHOR labels, validation fold 0 (5 patients), 3D metrics. Contributions depend on the order the steps are added. "
                           "HD95/ASSD leave out organs predicted empty, which flatters the low-Dice runs.")
    fig.savefig(out / "option_5_ablation_ladder.png")
    plt.close(fig)
    return steps, mean


def write_tables(rows, steps, mean, out: Path):
    (out / "tables").mkdir(parents=True, exist_ok=True)
    with (out / "tables" / "best_vs_baseline.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    with (out / "tables" / "ladder_foreground.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "experiment", "seeds", "dice_fg", "hd95_fg_mm", "assd_fg_mm"])
        for (name, exp), (_, v), i in zip(LADDER, steps, range(len(steps))):
            w.writerow([name.replace("\n", " "), exp, len(v["dice"]), *(round(float(mean[m][i]), 4) for m in METRICS)])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out_dir", type=Path, default=REPO / "dataset_analysis/results/best_vs_baseline")
    args = p.parse_args()
    out = args.out_dir
    (out / "plots").mkdir(parents=True, exist_ok=True)
    base, best, ce_stack = load_arm(BASELINE_EXP), load_arm(BEST_EXP), load_arm(LADDER[3][1])
    rows = build_rows(base, best)
    worse = worse_cells(rows)
    n_all = len(METRICS) * len(ORGANS)
    notes = {"worse": worse, "n_all": n_all, "n_ok": n_all - len(worse), "footnote": footnote(base, best, ce_stack)}
    plots = out / "plots"
    option_1(base, best, rows, notes, plots)
    option_2(base, best, rows, notes, plots)
    option_3(base, best, rows, notes, plots)
    option_4(base, best, rows, notes, plots)
    steps, mean = option_5(notes, plots)
    write_tables(rows, steps, mean, out)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    (out / "run.json").write_text(json.dumps({"git_commit": commit, "baseline": BASELINE_EXP, "best": BEST_EXP,
                                              "ladder": [e for _, e in LADDER], "segthor_version": "corrected",
                                              "worse_cells": [(c["metric"], c["group"]) for c in worse],
                                              "patient_organ_pairs_improved": notes["patient_pairs"]}, indent=2))
    print(f"wrote 5 options to {plots}; worse cells: {[(c['metric'], c['group']) for c in worse]}; patient pairs improved {notes['patient_pairs']}")


if __name__ == "__main__":
    main()
