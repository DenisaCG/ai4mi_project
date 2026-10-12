"""Table of every architecture variant tried before the 50-epoch run: mean Dice, HD95 and ASSD with change columns.

    python dataset_analysis/arch_variants_table.py [--out figures]

Writes arch_variants_v1/v2/v3.{png,pdf} and arch_variants.csv. All numbers are 25-epoch 4-fold cross-validation runs
on the corrected SEGTHOR data, read from metrics/cv_pooled_<experiment>.csv: the mean over the four organs of the mean
over the 40 patients (seed-averaged per patient). Green = better than the reference row, red = worse (HD95 and ASSD:
lower is better). Reference rows differ per version: v1 the head of each block, v2 and v3 the upgraded ENet baseline.
"""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from utils import REPO, load_cv_runs, pyplot

ORGANS = ("esophagus", "heart", "trachea", "aorta")
GREEN, RED, GREY, INK, MUTED = "#2e8b57", "#c0392b", "#8a8a8a", "#222222", "#6b6b6b"
GROUP_COLORS = {"ENet": "#4f7ca8", "U-Net": "#d9822b", "ResEnc U-Net": "#3d9a5f"}
BASELINE = "full_cv4_enet_dice_ce"
P = "full_cv4_"

# (group, row label, experiment, reference experiment for v1, topic for v3)
ROWS = [
    ("ENet", "Upgraded baseline (ENet)", BASELINE, None, "base"),
    ("ENet", "+ 2.5D ±1 slice", P + "enet_dice_ce_25d_c1", BASELINE, "2.5D"),
    ("ENet", "+ 2.5D ±2 slices", P + "enet_dice_ce_25d_c2", BASELINE, "2.5D"),
    ("U-Net", "U-Net", P + "unet_dice_ce", None, "base"),
    ("U-Net", "+ attention gates", P + "attention_unet_dice_ce", P + "unet_dice_ce", "U-Net"),
    ("U-Net", "+ deep supervision", P + "ds_unet_dice_ce", P + "unet_dice_ce", "U-Net"),
    ("U-Net", "+ cosine LR", P + "unet_dice_ce_cosine", P + "unet_dice_ce", "U-Net"),
    ("U-Net", "+ frozen DINOv2 encoder", P + "dino_unet_dice_ce", P + "unet_dice_ce", "U-Net"),
    ("ResEnc U-Net", "ResEnc U-Net", P + "resenc_unet_dice_ce", None, "base"),
    ("ResEnc U-Net", "+ deep supervision", P + "resenc_ds_unet_dice_ce", P + "resenc_unet_dice_ce", "ResEnc"),
    ("ResEnc U-Net", "+ DS, 2.5D ±1 slice", P + "resenc_ds_unet_dice_ce_25d_c1", P + "resenc_unet_dice_ce", "2.5D"),
    ("ResEnc U-Net", "+ DS, 2.5D ±1, 4 apart", P + "resenc_ds_unet_dice_ce_25d_c1_s4", P + "resenc_unet_dice_ce", "2.5D"),
    ("ResEnc U-Net", "+ DS, rot/scale/shift augmentation", P + "resenc_ds_unet_dice_ce_augmented", P + "resenc_unet_dice_ce", "ResEnc"),
]
METRICS = (("dice", "Dice", 3, 1), ("hd95", "HD95 mm", 1, -1), ("assd", "ASSD mm", 2, -1))


def load(metrics_dir: Path) -> dict[str, dict[str, float]]:
    """Mean over the four organs of the per-organ patient mean, for every experiment in ROWS."""
    out = {}
    for _, _, experiment, *_ in ROWS:
        d = pd.read_csv(metrics_dir / f"cv_pooled_{experiment}.csv")
        d = d[d.class_name.isin(ORGANS)]
        if d.class_name.nunique() != len(ORGANS) or d.patient.nunique() != 40:
            raise ValueError(f"{experiment}: expected {len(ORGANS)} organs x 40 patients")
        out[experiment] = d.groupby("class_name")[["dice", "hd95", "assd"]].mean().mean().to_dict()
    return out


def delta_text(delta: float, decimals: int) -> str:
    if round(delta, decimals) == 0:
        return "±0"
    return f"{'+' if delta > 0 else '−'}{abs(delta):.{decimals}f}"


def delta_color(delta: float, sign: int, decimals: int) -> str:
    return GREY if round(delta, decimals) == 0 else (GREEN if delta * sign > 0 else RED)


def tint(color: str, strength: float) -> tuple:
    """Color blended with white; strength 0 is white."""
    r, g, b = (int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return (1 - strength * (1 - r), 1 - strength * (1 - g), 1 - strength * (1 - b))


def reference(version: int, row: tuple) -> str | None:
    """Experiment a row's change is measured against (None for the reference rows themselves)."""
    if version == 1:
        return row[3]
    return None if row[2] == BASELINE else BASELINE


def layout_rows(version: int) -> list:
    """Display items: ('header', text), ('gap',) or ('row', ROWS entry)."""
    items = []
    if version < 3:
        last = None
        for row in ROWS:
            if row[0] != last:
                if last is not None:
                    items.append(("gap",))
                last = row[0]
            items.append(("row", row))
        return items
    # v3: one block per thing that was varied; the 2.5D block collects every context-stack run
    blocks = [
        ("2.5D variants (neighbouring slices as extra input channels)", "2.5D", ["ENet", "ResEnc U-Net"]),
        ("U-Net variants", "U-Net", ["U-Net"]),
        ("ResEnc U-Net variants", "ResEnc", ["ResEnc U-Net"]),
    ]
    for i, (title, topic, groups) in enumerate(blocks):
        if i:
            items.append(("gap",))
        items.append(("header", title))
        pool = [r for r in ROWS if r[0] in groups and r[4] in (topic, "base")]
        if topic == "2.5D":
            pool = [r for r in ROWS if r[4] == "2.5D" or r[2] in (BASELINE, P + "resenc_ds_unet_dice_ce")]
        elif topic == "ResEnc":
            pool = [r for r in pool if r[4] == "ResEnc" or r[2] == P + "resenc_unet_dice_ce"]
        for r in pool:
            items.append(("row", r))
    return items


def items_title(items: list, row: tuple) -> str:
    """Title of the v3 block the row sits in."""
    title = ""
    for item in items:
        if item[0] == "header":
            title = item[1]
        if item[0] == "row" and item[1] is row:
            return title
    return title


def draw(version: int, stats: dict, out: Path) -> None:
    plt = pyplot()
    items = layout_rows(version)
    n_rows = sum(1 for i in items if i[0] == "row")
    all_rows = [r for r in ROWS]
    best = {
        k: (max if sign > 0 else min)(stats[r[2]][k] for r in all_rows) for k, _, _, sign in METRICS
    }
    row_h, head_h = 0.36, 0.95
    height = head_h + sum(0.2 if i[0] == "gap" else row_h for i in items) + 0.5
    fig = plt.figure(figsize=(13.4, height), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 13.4)
    ax.set_ylim(height, 0)
    ax.axis("off")

    label_x = 0.55
    col_x = {"dice": 5.35, "hd95": 8.0, "assd": 10.65}  # metric value centre; delta sits 1.15 to the right
    ref_text = {
        1: "Δ vs the first row of each block",
        2: "Δ vs upgraded baseline (ENet)",
        3: "Δ vs upgraded baseline (ENet)",
    }[version]
    ax.text(0.25, 0.28, f"mean of 12 runs (4 folds × 3 seeds), mean over the 4 organs · {ref_text} · green = better, red = worse",
            fontsize=10.5, color=MUTED, va="center")
    for key, name, _, _ in METRICS:
        ax.text(col_x[key], 0.72, name, fontsize=12.5, fontweight="bold", color=INK, ha="center", va="center")
        ax.text(col_x[key] + 1.2, 0.72, "Δ", fontsize=12.5, fontweight="bold", color=INK, ha="center", va="center")

    y = head_h
    last_group = None
    for item in items:
        if item[0] == "gap":
            ax.plot([0.25, 13.15], [y + 0.1] * 2, color="#dddddd", lw=1)
            y += 0.2
            continue
        if item[0] == "header":
            ax.text(0.25, y + row_h / 2, item[1], fontsize=12, fontweight="bold", color=INK, va="center")
            y += row_h
            continue
        row = item[1]
        group, label, experiment = row[0], row[1], row[2]
        is_head = row[3] is None if version < 3 else row[2] in (BASELINE, P + "unet_dice_ce", P + "resenc_unet_dice_ce")
        color = GROUP_COLORS[group]
        ax.add_patch(plt.Rectangle((0.25, y + 0.04), 0.1, row_h - 0.08, color=color, lw=0))
        if version == 3 and group == "ENet" and "baseline" not in label.lower():
            label = "ENet " + label
        if version == 3 and experiment == P + "resenc_ds_unet_dice_ce" and "2.5D" in items_title(items, row):
            label = "ResEnc U-Net + deep supervision (2D)"
        if version == 3 and group == "ResEnc U-Net" and label.startswith("+ DS, 2.5D"):
            label = label.replace("+ DS, 2.5D", "ResEnc DS + 2.5D")
        ax.text(label_x if is_head else label_x + 0.3, y + row_h / 2, label, fontsize=12,
                fontweight="bold" if is_head else "normal", color=color if is_head else INK, va="center")
        ref = reference(version, row)
        for key, _, dec, sign in METRICS:
            v = stats[experiment][key]
            is_best = abs(v - best[key]) < 1e-12
            ax.text(col_x[key], y + row_h / 2, f"{v:.{dec}f}", fontsize=12, color=INK, ha="center", va="center",
                    fontweight="normal")
            if ref is None:
                continue
            d = v - stats[ref][key]
            c = delta_color(d, sign, dec)
            x = col_x[key] + 1.2
            if version == 3 and c != GREY:
                scale = {"dice": 0.04, "hd95": 8.0, "assd": 1.0}[key]
                ax.add_patch(plt.Rectangle((x - 0.45, y + 0.04), 0.9, row_h - 0.08,
                                           color=tint(c, min(1.0, abs(d) / scale) * 0.55), lw=0, zorder=0))
            if version == 2 and c != GREY:
                scale = {"dice": 0.04, "hd95": 8.0, "assd": 1.0}[key]
                w = min(1.0, abs(d) / scale) * 0.45
                ax.add_patch(plt.Rectangle((x + 0.5, y + 0.12), w if d * sign > 0 else w, row_h - 0.24,
                                           color=tint(c, 0.45), lw=0))
            ax.text(x, y + row_h / 2, delta_text(d, dec), fontsize=12, color=c, ha="center", va="center",
                    fontweight="bold")
        y += row_h
    ax.text(0.25, y + 0.25, "DS = deep supervision. All runs 25 epochs, before the 50-epoch run.", fontsize=9.5, color=MUTED, va="center")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"arch_variants_v{version}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)

# Verified parent of every row (the row differing in exactly one config setting; see configs/full_cv4_<experiment>.yaml).
# The frozen-DINOv2 U-Net replaces the architecture (model.name dino_unet) rather than adding to the U-Net, so its parent
# is the plain U-Net only as the family reference.
DS = P + "resenc_ds_unet_dice_ce"
PARENT = {
    P + "enet_dice_ce_25d_c1": BASELINE,
    P + "enet_dice_ce_25d_c2": BASELINE,
    P + "attention_unet_dice_ce": P + "unet_dice_ce",
    P + "ds_unet_dice_ce": P + "unet_dice_ce",
    P + "unet_dice_ce_cosine": P + "unet_dice_ce",
    P + "dino_unet_dice_ce": P + "unet_dice_ce",
    DS: P + "resenc_unet_dice_ce",
    DS + "_25d_c1": DS,
    DS + "_25d_c1_s4": DS + "_25d_c1",
    DS + "_augmented": DS,
}
INDENT = {DS + "_25d_c1": 2, DS + "_25d_c1_s4": 3, DS + "_augmented": 2}
FINAL_MODEL = DS  # ResEnc U-Net + deep supervision, the row highlighted as final model before the 50-epoch run
RUN_DIRS = ("metrics", "metrics/augmentation_metrics/joint")
LABELS = {r[2]: r[1] for r in ROWS}
LABELS[DS + "_25d_c1_s4"] = "+ 2.5D ±1, 4 apart"
LABELS[DS + "_25d_c1"] = "+ 2.5D ±1 slice"
LABELS[DS + "_augmented"] = "+ rot/scale/shift augmentation"


def load_runs(experiment: str) -> dict[tuple[int, int], dict[str, float]]:
    """Per (fold, seed): mean over the four organs of the mean over the validation patients, from metrics_3d.csv."""
    for d in RUN_DIRS:
        if (REPO / d / f"{experiment}_fold0").exists():
            rows = load_cv_runs(REPO / d, experiment, ("dice", "hd95", "assd"))
            break
    else:
        raise FileNotFoundError(experiment)
    cells = defaultdict(list)
    for r in rows:
        cells[r["fold"], r["seed"], r["organ"]].append(r)
    runs = {}
    for fold, seed in sorted({k[:2] for k in cells}):
        runs[fold, seed] = {}
        for m, *_ in METRICS:
            organ_means = []
            for organ in ORGANS:
                vals = [r[m] for r in cells[fold, seed, organ]]
                bad = [v for v in vals if not math.isfinite(v)]
                if bad:
                    print(f"WARNING {experiment} fold{fold} seed{seed} {organ} {m}: {len(bad)} non-finite of {len(vals)}")
                organ_means.append(statistics.fmean(vals))
            runs[fold, seed][m] = statistics.fmean(organ_means)
    if len(runs) != 12:
        raise ValueError(f"{experiment}: {len(runs)} runs, expected 12")
    return runs


def paired(runs: dict, a: str, b: str, key: str, rng) -> tuple[float, float, float]:
    """Mean of the 12 paired differences a - b (paired by fold and seed) and its bootstrap 95% CI (10k resamples)."""
    diff = np.array([runs[a][k][key] - runs[b][k][key] for k in sorted(runs[a])])
    boot = rng.choice(diff, size=(10000, len(diff)), replace=True).mean(axis=1)
    return diff.mean(), *np.percentile(boot, [2.5, 97.5])


def fixed_stats(out: Path) -> tuple[dict, dict]:
    """Per-run means, and for every row the paired change vs its verified parent (and vs its old block head)."""
    experiments = [r[2] for r in ROWS]
    runs = {e: load_runs(e) for e in experiments}
    rng = np.random.default_rng(0)
    old_parent = {r[2]: r[3] for r in ROWS}
    res = {}
    for e in experiments:
        res[e] = {"mean": {m: statistics.fmean(v[m] for v in runs[e].values()) for m, *_ in METRICS}}
        for tag, ref in (("new", PARENT.get(e)), ("old", old_parent[e]), ("vs_enet", None if e == BASELINE else BASELINE)):
            if ref is not None:
                res[e][tag] = {m: paired(runs, e, ref, m, rng) for m, *_ in METRICS}
    with open(out / "arch_variants_fixed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "experiment", "old_parent", "verified_parent", "metric", "mean", "old_delta", "new_delta", "new_ci_low", "new_ci_high", "delta_vs_enet", "enet_ci_low", "enet_ci_high"])
        for e in experiments:
            for m, *_ in METRICS:
                o, n, v = res[e].get("old"), res[e].get("new"), res[e].get("vs_enet")
                w.writerow([LABELS[e], e, old_parent[e] or "", PARENT.get(e, ""), m, f"{res[e]['mean'][m]:.5f}",
                            f"{o[m][0]:.5f}" if o else "", f"{n[m][0]:.5f}" if n else "", f"{n[m][1]:.5f}" if n else "",
                            f"{n[m][2]:.5f}" if n else "", f"{v[m][0]:.5f}" if v else "", f"{v[m][1]:.5f}" if v else "", f"{v[m][2]:.5f}" if v else ""])
    return runs, res


def draw_fixed(res: dict, out: Path) -> None:
    """v1 layout with every Δ against the verified parent; grey when the paired 95% CI crosses 0."""
    plt = pyplot()
    experiments = [r[2] for r in ROWS]
    best = {m: (max if sign > 0 else min)(res[e]["mean"][m] for e in experiments) for m, _, _, sign in METRICS}
    row_h, head_h = 0.36, 0.95
    n_gaps = 2
    height = head_h + len(ROWS) * row_h + n_gaps * 0.2 + 1.5
    W = 14.4
    fig = plt.figure(figsize=(W, height), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(height, 0)
    ax.axis("off")
    col_x = {"dice": 7.4, "hd95": 9.9, "assd": 12.4}
    ax.text(0.25, 0.28, "Δ vs the configuration each change was added to · paired by fold & seed · green = better, red = worse, "
            "grey = 95% CI crosses 0", fontsize=10.5, color=MUTED, va="center")
    ax.text(4.9, 0.72, "reference", fontsize=11, color=MUTED, va="center")
    for key, name, _, _ in METRICS:
        ax.text(col_x[key], 0.72, name, fontsize=12.5, fontweight="bold", color=INK, ha="center", va="center")
        ax.text(col_x[key] + 1.15, 0.72, "Δ", fontsize=12.5, fontweight="bold", color=INK, ha="center", va="center")
    y, last = head_h, None
    for group, label, e, old, _ in ROWS:
        if last is not None and group != last:
            ax.plot([0.25, W - 0.25], [y + 0.1] * 2, color="#dddddd", lw=1)
            y += 0.2
        last = group
        color = GROUP_COLORS[group]
        is_head = e not in PARENT
        level = 0 if is_head else INDENT.get(e, 1)
        label = LABELS[e] if (e in INDENT or is_head and False) else label
        ax.add_patch(plt.Rectangle((0.25, y + 0.04), 0.1, row_h - 0.08, color=color, lw=0))
        ax.text(0.55 + 0.3 * level, y + row_h / 2, label, fontsize=12, fontweight="bold" if is_head else "normal",
                color=color if is_head else INK, va="center")
        if not is_head:
            ref = LABELS[PARENT[e]] if PARENT[e] not in (BASELINE, P + "unet_dice_ce", P + "resenc_unet_dice_ce") else \
                {BASELINE: "ENet baseline", P + "unet_dice_ce": "U-Net", P + "resenc_unet_dice_ce": "ResEnc U-Net"}[PARENT[e]]
            if PARENT[e] == DS:
                ref = "+ deep supervision"
            ax.text(4.9, y + row_h / 2, f"vs {ref}", fontsize=10, color=MUTED, va="center")
        for key, _, dec, sign in METRICS:
            v = res[e]["mean"][key]
            ax.text(col_x[key], y + row_h / 2, f"{v:.{dec}f}", fontsize=12, color=INK, ha="center", va="center",
                    fontweight="bold" if abs(v - best[key]) < 1e-12 else "normal")
            if is_head:
                continue
            d, lo, hi = res[e]["new"][key]
            c = GREY if (lo <= 0 <= hi or round(d, dec) == 0) else (GREEN if d * sign > 0 else RED)
            ax.text(col_x[key] + 1.15, y + row_h / 2, delta_text(d, dec), fontsize=12, color=c, ha="center", va="center",
                    fontweight="bold")
        y += row_h
    fin = res[FINAL_MODEL]["vs_enet"]
    parts = [f"{name} {delta_text(fin[m][0], dec)} (95% CI {fin[m][1]:+.{dec}f} to {fin[m][2]:+.{dec}f})".replace("-", "−")
             for m, name, dec, _ in ((m, n.split()[0], d, s) for m, n, d, s in METRICS)]
    ax.text(0.25, y + 0.3, "Final model (ResEnc U-Net + deep supervision) vs ENet baseline, paired: " + " · ".join(parts),
            fontsize=10.5, color=INK, va="center")
    ax.text(0.25, y + 0.65, "Bold = best value in column. 12 runs per row (4 folds × 3 seeds), 25 epochs, mean over the 4 organs.",
            fontsize=9.5, color=MUTED, va="center")
    ax.text(0.25, y + 0.95, "Frozen DINOv2 replaces the U-Net (dino_unet), so its reference is only the U-Net family.",
            fontsize=9.5, color=MUTED, va="center")
    fig.savefig(out / "arch_variants_v1_fixed.png", dpi=300, facecolor="white")
    fig.savefig(out / "arch_variants_v1_fixed.svg", facecolor="white")
    plt.close(fig)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO / "figures")
    parser.add_argument("--metrics", type=Path, default=REPO / "metrics")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stats = load(args.metrics)
    with open(args.out / "arch_variants.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "variant", "experiment", "dice", "hd95", "assd"])
        for group, label, experiment, *_ in ROWS:
            s = stats[experiment]
            w.writerow([group, label, experiment, f"{s['dice']:.4f}", f"{s['hd95']:.3f}", f"{s['assd']:.4f}"])
    for version in (1, 2, 3):
        draw(version, stats, args.out)
    _, res = fixed_stats(args.out)
    draw_fixed(res, args.out)


if __name__ == "__main__":
    main()
