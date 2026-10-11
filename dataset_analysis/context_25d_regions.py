#!/usr/bin/env python3
"""
Where does 2.5D context change the segmentation? 2.5D minus 2D, scored separately in the middle of each
organ, near its ends, and on the empty slices just past its ends.

For every organ the ground truth splits a patient's slices (original CT grid) into:
  middle  organ slices more than EDGE_MM from the organ's most superior and most inferior slice
  end     organ slices within EDGE_MM of either end
  past    slices without the organ within EDGE_MM beyond either end
and each run's stitched validation volume (runs/<exp>_fold<k>/seed<s>/volumes/val) is scored as:
  middle / end  Dice over the voxels of those slices
  past          % of those slices on which the model predicts the organ (false presence)
Values are averaged over the 3 seeds per patient, then 2.5D - 2D is taken per patient (paired, 40 patients)
and summarised as mean with a 95% bootstrap interval.

Writes into <out-dir>/:
  context_25d_regions.png/.pdf   one row per architecture, one column per region score
  context_25d_regions.csv        per experiment, patient, organ: the three scores (seed mean)
  context_25d_regions_delta.csv  per comparison, organ, score: mean difference and 95% interval
  context_25d_ends.png/.pdf      where predicted organs stop, 2.5D - 2D per organ end (top/bottom), mm
  context_25d_ends_scatter.png/.pdf  esophagus bottom-end error, 2D vs 2.5D, one dot per patient (U-Net)
  context_25d_ends_delta.csv     paired differences of the end errors and missed end slices
  context_25d_regions_log.json   inputs, git commit, time
  cache/                         one csv per scored run, so a rerun skips finished runs

Usage:
    python dataset_analysis/context_25d_regions.py --runs-dir runs --out-dir figures/context_25d_regions
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import CLASSES, NAMES  # noqa: E402  (also puts tools/ on sys.path)
from plot_style import EARTH_LABEL_COLORS, PALETTE, apply_style, decorate, legend_below, tint  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

EDGE_MM = 10.0
MISSED_MM = 20.0  # end slices checked for being missed entirely
GT_PATTERN = "data/segthor_train_full/train/{patient}/GT.nii.gz"
FOLDS, SEEDS = range(4), range(3)
# (row title, 2D experiment, [(label, 2.5D experiment), ...])
COMPARISONS = [
    (
        "Residual-encoder U-Net",
        "full_cv4_resenc_ds_unet_dice_ce",
        [
            ("2.5D, 3 slices, stride 1 (±2.5 mm)", "full_cv4_resenc_ds_unet_dice_ce_25d_c1"),
            ("2.5D, 3 slices, stride 4 (±10 mm)", "full_cv4_resenc_ds_unet_dice_ce_25d_c1_s4"),
        ],
    ),
    (
        "ENet",
        "full_cv4_enet_dice_ce",
        [
            ("2.5D, 3 slices, stride 1 (±2.5 mm)", "full_cv4_enet_dice_ce_25d_c1"),
            ("2.5D, 5 slices, stride 1 (±5 mm)", "full_cv4_enet_dice_ce_25d_c2"),
        ],
    ),
]
SCORES = [
    ("dice_middle", "Dice, middle of organ"),
    ("dice_end", f"Dice, within {EDGE_MM:g} mm of an end"),
    ("false_past", f"Wrong organ slices just past\nits ends (% of slices, ≤{EDGE_MM:g} mm)"),
]
END_SCORES = [
    ("top_err_mm", "top end"),
    ("bottom_err_mm", "bottom end"),
    ("top_missed_mm", "top end, missed"),
    ("bottom_missed_mm", "bottom end, missed"),
]


def regions(gt_mask: np.ndarray, z_mm: float) -> dict[str, np.ndarray] | None:
    """Slice indices of the middle, end and past-end regions of one organ.

    Args:
        gt_mask: Boolean ground-truth mask of the organ, shape (x, y, z).
        z_mm: Slice spacing in mm.

    Returns:
        Region name -> slice indices, or None if the organ is absent.
    """
    occupied = gt_mask.any(axis=(0, 1))
    idx = np.flatnonzero(occupied)
    if idx.size == 0:
        return None
    z = np.arange(gt_mask.shape[2])
    to_end = np.minimum(np.abs(z - idx[0]), np.abs(z - idx[-1])) * z_mm
    inside = (z >= idx[0]) & (z <= idx[-1])
    return {
        "middle": np.flatnonzero(inside & occupied & (to_end > EDGE_MM)),
        "end": np.flatnonzero(inside & occupied & (to_end <= EDGE_MM)),
        "past": np.flatnonzero(~inside & (to_end <= EDGE_MM)),
    }


def end_errors(gt_mask: np.ndarray, pred_mask: np.ndarray, z_mm: float) -> dict[str, float]:
    """How far the prediction's top and bottom are from the ground truth's, and how many end slices it misses.

    The predicted ends are the first and last slice of the longest unbroken run of predicted slices (the run
    holding the most predicted voxels), so a stray blob on slices further away does not count as the organ's
    end. This is 1D along z: a full 3D connected-component labelling per volume made the job ~60x slower.

    Args:
        gt_mask: Boolean ground-truth mask of the organ, (x, y, z), z toward superior (LPS).
        pred_mask: Boolean predicted mask of the organ, same shape.
        z_mm: Slice spacing in mm.

    Returns:
        top_err_mm / bottom_err_mm: positive = the prediction stops short of that end (organ cut short),
        negative = it runs past it; NaN if nothing is predicted.
        top_missed_mm / bottom_missed_mm: ground-truth slices within MISSED_MM of that end with no prediction
        on them at all, in mm.
    """
    gz = np.flatnonzero(gt_mask.any(axis=(0, 1)))
    bottom, top = gz[0], gz[-1]
    out = {"top_err_mm": np.nan, "bottom_err_mm": np.nan}
    voxels = pred_mask.sum(axis=(0, 1))
    predicted = voxels > 0
    runs, n = label(predicted)
    if n:
        largest = np.argmax(np.bincount(runs, weights=voxels)[1:]) + 1
        pz = np.flatnonzero(runs == largest)
        out = {"top_err_mm": (top - pz[-1]) * z_mm, "bottom_err_mm": (pz[0] - bottom) * z_mm}
    reach = int(round(MISSED_MM / z_mm))
    for name, near in (("top", gz[gz > top - reach]), ("bottom", gz[gz < bottom + reach])):
        out[f"{name}_missed_mm"] = float((~predicted[near]).sum() * z_mm)
    out["ends_v"] = 2  # end definition version; 1 used 3D components
    return out


def score_run(task: tuple[str, int, int, str, str]) -> str:
    """Score every validation volume of one run and cache the rows as csv.

    Args:
        task: (experiment, fold, seed, run directory, cache csv path).

    Returns:
        A one-line status message.
    """
    exp, fold, seed, run_dir, cache = task
    rows = []
    for pred_path in sorted(Path(run_dir, "volumes", "val").glob("Patient_*.nii.gz")):
        patient = pred_path.name.removesuffix(".nii.gz")
        try:
            gt_nii = nib.load(GT_PATTERN.format(patient=patient))
            pred_nii = nib.load(pred_path)
            assert gt_nii.shape == pred_nii.shape, f"shape {pred_nii.shape} != GT {gt_nii.shape}"
            assert np.allclose(gt_nii.affine, pred_nii.affine, atol=1e-3), "affine differs from GT"
            gt, pred = np.asarray(gt_nii.dataobj), np.asarray(pred_nii.dataobj)
            z_mm = float(gt_nii.header.get_zooms()[2])
        except Exception as e:  # one unreadable volume should not lose the other runs
            print(f"[skip] {pred_path}: {type(e).__name__}: {e}", flush=True)
            continue
        for c in CLASSES:
            g, p = gt == c, pred == c
            reg = regions(g, z_mm)
            if reg is None:
                continue
            row = {"experiment": exp, "fold": fold, "seed": seed, "patient": patient, "label": c}
            for name in ("middle", "end"):
                gs, ps = g[:, :, reg[name]], p[:, :, reg[name]]
                denom = gs.sum() + ps.sum()
                row[f"dice_{name}"] = 2 * np.logical_and(gs, ps).sum() / denom if denom else np.nan
            past = reg["past"]
            row["false_past"] = 100 * p[:, :, past].any(axis=(0, 1)).mean() if past.size else np.nan
            row["n_middle"], row["n_end"], row["n_past"] = (len(reg[k]) for k in ("middle", "end", "past"))
            row |= end_errors(g, p, z_mm)
            rows.append(row)
    pd.DataFrame(rows).to_csv(cache, index=False)
    return f"{exp} fold{fold} seed{seed}: {len({r['patient'] for r in rows})} patients"


def bootstrap_ci(values: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean.

    Args:
        values: Paired per-patient differences.
        n: Number of bootstrap resamples.
        seed: RNG seed.

    Returns:
        (low, high).
    """
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, len(values))).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def deltas(per_patient: pd.DataFrame, scores: list[tuple[str, str]]) -> pd.DataFrame:
    """Paired 2.5D - 2D differences per comparison, organ and score.

    Args:
        per_patient: Seed-mean scores, one row per experiment, patient and organ.
        scores: (column, title) pairs to compare.

    Returns:
        One row per (architecture, variant, organ, score) with mean, ci_low, ci_high, n_patients,
        and the 2D and 2.5D means.
    """
    rows = []
    keyed = per_patient.set_index(["experiment", "patient", "label"])
    for arch, base, variants in COMPARISONS:
        for label, exp in variants:
            for c in sorted(per_patient.label.unique()):
                for score, _ in scores:
                    a = keyed.loc[base][score].xs(c, level="label")
                    b = keyed.loc[exp][score].xs(c, level="label")
                    d = (b - a).dropna()
                    lo, hi = bootstrap_ci(d.to_numpy())
                    rows.append(
                        {
                            "architecture": arch,
                            "variant": label,
                            "experiment": exp,
                            "label": c,
                            "score": score,
                            "mean_2d": a.loc[d.index].mean(),
                            "mean_25d": b.loc[d.index].mean(),
                            "mean_delta": d.mean(),
                            "ci_low": lo,
                            "ci_high": hi,
                            "n_patients": len(d),
                        }
                    )
    return pd.DataFrame(rows)


def draw(delta: pd.DataFrame, labels: list[int], out_dir: Path) -> None:
    """Grid of dot-and-interval plots: rows = architectures, columns = scores, y = organ.

    Args:
        delta: Output of deltas.
        labels: Organs drawn, top to bottom.
        out_dir: Folder to write the figure into.
    """
    apply_style()
    fig, axes = plt.subplots(len(COMPARISONS), len(SCORES), figsize=(15, 4.2 * len(COMPARISONS)), sharey=True)
    for r, (arch, _, variants) in enumerate(COMPARISONS):
        for col, (score, title) in enumerate(SCORES):
            ax = axes[r, col]
            ax.axvline(0, color="#555555", lw=1, zorder=1)
            ax.grid(axis="x", color="#E6E6E6")
            ax.grid(axis="y", visible=False)
            for v, (label, exp) in enumerate(variants):
                color = PALETTE[0] if v == 0 else PALETTE[7]
                offset = (v - (len(variants) - 1) / 2) * 0.28
                for i, c in enumerate(labels):
                    row = delta[(delta.experiment == exp) & (delta.label == c) & (delta.score == score)].iloc[0]
                    y = i + offset
                    ax.plot([row.ci_low, row.ci_high], [y, y], color=color, lw=2.2, solid_capstyle="round", zorder=2)
                    ax.plot(row.mean_delta, y, "o", color=color, ms=7, mec="white", mew=1, zorder=3)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels([NAMES[c] for c in labels], fontweight="bold")
            for tick, c in zip(ax.get_yticklabels(), labels):
                tick.set_color(EARTH_LABEL_COLORS[c])
            ax.set_ylim(len(labels) - 0.5, -0.5)
            ax.set_title(title if r == 0 else "", loc="left", fontsize=12, fontweight="bold")
            ax.set_xlabel("2.5D − 2D (Dice)" if score.startswith("dice") else "2.5D − 2D (percentage points)")
            # symmetric x-range so left = worse, right = better reads the same in every panel
            lim = max(abs(delta[delta.score == score][["ci_low", "ci_high"]]).max().max(), 1e-3) * 1.15
            ax.set_xlim(-lim, lim)
        axes[r, 0].set_ylabel(arch, fontsize=13, fontweight="bold", labelpad=12)
    # colour means the same in both rows: 3 slices at stride 1, or the row's wider context
    axes[0, 0].plot([], [], "o-", color=PALETTE[0], lw=2.2, label="3 slices, stride 1 (±2.5 mm)")
    axes[0, 0].plot([], [], "o-", color=PALETTE[7], lw=2.2,
                    label="wider context: U-Net 3 slices stride 4 (±10 mm), ENet 5 slices stride 1 (±5 mm)")
    legend_below(axes[0, 0], ncol=2)

    decorate(
        fig,
        "2.5D Changes Almost Nothing Mid-Organ; at Organ Ends the Effects Are Larger but Inconsistent",
        subtitle=(
            "Paired difference 2.5D − 2D per patient (40 patients, 4-fold CV, each patient averaged over 3 seeds); "
            "dot = mean, bar = 95% bootstrap interval. Dice: right of 0 is better. Wrong slices past the ends: "
            "left of 0 is better."
        ),
        footnote_text=(
            "Data: data/segthor_train_full/train, corrected labels (aorta separate); stitched validation volumes on "
            f"the original CT grid. Regions from the ground truth per organ: ends = organ slices within {EDGE_MM:g} mm "
            f"of its top or bottom slice, middle = the other organ slices, past = empty slices within {EDGE_MM:g} mm "
            "beyond either end. Dice over the voxels of those slices. 48 intervals are shown, so 2-3 are expected to "
            "exclude 0 by chance alone."
        ),
    )
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"context_25d_regions.{ext}")
    plt.close(fig)
    print(f"wrote {out_dir / 'context_25d_regions.png'}")


def per_patient_deltas(per_patient: pd.DataFrame, base: str, exp: str, label_: int, score: str) -> np.ndarray:
    """Per-patient 2.5D - 2D differences of one score.

    Args:
        per_patient: Seed-mean scores, one row per experiment, patient and organ.
        base: 2D experiment.
        exp: 2.5D experiment.
        label_: Organ label.
        score: Column to compare.

    Returns:
        One difference per patient with both values.
    """
    keyed = per_patient[per_patient.label == label_].set_index(["experiment", "patient"])[score]
    return (keyed.loc[exp] - keyed.loc[base]).dropna().to_numpy()


def draw_ends(per_patient: pd.DataFrame, delta: pd.DataFrame, labels: list[int], out_dir: Path) -> None:
    """Change in where the prediction ends, 2.5D - 2D: one column per architecture, one row per organ end.

    Args:
        per_patient: Seed-mean scores, for the faint per-patient dots.
        delta: Output of deltas with END_SCORES.
        labels: Organs drawn, top to bottom.
        out_dir: Folder to write the figure into.
    """
    apply_style()
    rows = [(c, score, f"{NAMES[c]}, {title}") for c in labels for score, title in END_SCORES[:2]]
    fig, axes = plt.subplots(1, len(COMPARISONS), figsize=(13, 7.5), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (arch, base, variants) in zip(axes, COMPARISONS):
        ax.axvline(0, color="#555555", lw=1, zorder=1)
        ax.grid(axis="x", color="#E6E6E6")
        ax.grid(axis="y", visible=False)
        for v, (_, exp) in enumerate(variants):
            color = PALETTE[0] if v == 0 else PALETTE[7]
            offset = (v - (len(variants) - 1) / 2) * 0.3
            for i, (c, score, _) in enumerate(rows):
                y = i + offset
                d = per_patient_deltas(per_patient, base, exp, c, score)
                ax.scatter(d, y + rng.uniform(-0.08, 0.08, len(d)), s=9, color=color, alpha=0.25, lw=0, zorder=2)
                r = delta[(delta.experiment == exp) & (delta.label == c) & (delta.score == score)].iloc[0]
                ax.plot([r.ci_low, r.ci_high], [y, y], color=color, lw=2.4, solid_capstyle="round", zorder=3)
                ax.plot(r.mean_delta, y, "o", color=color, ms=7.5, mec="white", mew=1, zorder=4)
        for i in range(1, len(labels)):
            ax.axhline(2 * i - 0.5, color="#DDDDDD", lw=0.8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([t for *_, t in rows])
        for tick, (c, *_) in zip(ax.get_yticklabels(), rows):
            tick.set_color(EARTH_LABEL_COLORS[c])
            tick.set_fontweight("bold")
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xlim(-20, 20)  # a few patients reach further; the means and intervals are well inside
        ax.set_title(arch, loc="left", fontsize=13, fontweight="bold")
        ax.set_xlabel("2.5D − 2D (mm)\n←  2.5D runs further      2.5D stops earlier  →")
    axes[0].plot([], [], "o-", color=PALETTE[0], lw=2.2, label="3 slices, stride 1 (±2.5 mm)")
    axes[0].plot([], [], "o-", color=PALETTE[7], lw=2.2,
                 label="wider context: U-Net 3 slices stride 4 (±10 mm), ENet 5 slices stride 1 (±5 mm)")
    legend_below(axes[0], ncol=2)
    decorate(
        fig,
        "2.5D Does Not Systematically Cut Organs Short; Where They Stop Varies More Between Patients Than Between Models",
        subtitle=(
            "Change in where the predicted organ stops, per organ end, 2.5D − 2D per patient (40 patients, each "
            "averaged over 3 seeds). Big dot = mean, bar = 95% bootstrap interval, faint dots = individual patients."
        ),
        footnote_text=(
            "Predicted ends = first and last slice of the longest unbroken run of slices with that organ predicted, on the original "
            "CT grid; error = distance from the ground-truth end slice, positive when the prediction stops short. "
            "Data: data/segthor_train_full/train, corrected labels. x-axis cut at ±20 mm."
        ),
    )
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"context_25d_ends.{ext}")
    plt.close(fig)
    print(f"wrote {out_dir / 'context_25d_ends.png'}")


def draw_end_scatter(per_patient: pd.DataFrame, out_dir: Path, organ: int = 1) -> None:
    """Per-patient bottom-end error of 2D against 2.5D for the U-Net variants (esophagus by default).

    Args:
        per_patient: Seed-mean scores.
        out_dir: Folder to write the figure into.
        organ: Organ label.
    """
    apply_style()
    arch, base, variants = COMPARISONS[0]
    keyed = per_patient[per_patient.label == organ].set_index(["experiment", "patient"]).bottom_err_mm
    fig, axes = plt.subplots(1, len(variants), figsize=(11, 5.6), sharex=True, sharey=True)
    lo, hi = -15, 35
    for ax, (label_, exp) in zip(axes, variants):
        both = pd.concat([keyed.loc[base].rename("d2"), keyed.loc[exp].rename("d25")], axis=1).dropna()
        ax.fill_between([lo, hi], [lo, hi], hi, color=tint(PALETTE[7], 0.12), lw=0, zorder=0)
        ax.plot([lo, hi], [lo, hi], color="#888888", lw=1, ls="--", zorder=1)
        ax.axhline(0, color="#BBBBBB", lw=0.8)
        ax.axvline(0, color="#BBBBBB", lw=0.8)
        ax.scatter(both.d2.clip(lo, hi), both.d25.clip(lo, hi), s=34, color=EARTH_LABEL_COLORS[organ],
                   edgecolor="white", lw=0.8, zorder=3)
        above = int((both.d25 > both.d2).sum())
        ax.text(0.03, 0.97, f"2.5D stops earlier for {above} of {len(both)} patients", transform=ax.transAxes,
                va="top", fontsize=10.5, color="#444444")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(label_, loc="left", fontsize=12, fontweight="bold")
        ax.set_xlabel("2D: bottom end cut short by (mm)")
    axes[0].set_ylabel("2.5D: bottom end cut short by (mm)")
    decorate(
        fig,
        f"{NAMES[organ]} Bottom End: 2.5D Stops Earlier for About Half the Patients, as Chance Would Give",
        subtitle=(
            f"{arch}. How far above the true bottom of the {NAMES[organ].lower()} the prediction stops (negative = "
            "runs past it), averaged over 3 seeds. Dots in the shaded half: 2.5D stops earlier than 2D."
        ),
        footnote_text=(
            "Predicted end = bottom slice of the longest unbroken run of predicted slices; original CT grid; "
            f"values clipped to [{lo}, {hi}] mm. Data: data/segthor_train_full/train, corrected labels."
        ),
    )
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"context_25d_ends_scatter.{ext}")
    plt.close(fig)
    print(f"wrote {out_dir / 'context_25d_ends_scatter.png'}")


def main():
    """Score every run (cached), compute paired differences, write tables, log and figure."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--out-dir", type=Path, default=Path("figures/context_25d_regions"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cache_dir = args.out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()

    experiments = sorted({e for _, base, vs in COMPARISONS for e in [base, *(x for _, x in vs)]})
    tasks, missing = [], []
    for exp in experiments:
        for fold in FOLDS:
            for seed in SEEDS:
                run_dir = args.runs_dir / f"{exp}_fold{fold}" / f"seed{seed}"
                cache = cache_dir / f"{exp}_fold{fold}_seed{seed}.csv"
                if not (run_dir / "volumes" / "val").is_dir():
                    missing.append(str(run_dir))
                elif not cache.exists() or "ends_v" not in pd.read_csv(cache, nrows=0).columns:
                    tasks.append((exp, fold, seed, str(run_dir), str(cache)))
    print(f"{len(tasks)} runs to score, {len(experiments) * len(FOLDS) * len(SEEDS) - len(tasks) - len(missing)} cached, "
          f"{len(missing)} missing", flush=True)
    assert not missing, f"runs without volumes/val: {missing}"
    with ProcessPoolExecutor(args.workers) as pool:
        for msg in pool.map(score_run, tasks):
            print(msg, flush=True)

    runs = pd.concat([pd.read_csv(p) for p in sorted(cache_dir.glob("*.csv"))], ignore_index=True)
    runs = runs[runs.experiment.isin(experiments)]
    per_patient = runs.groupby(["experiment", "patient", "label"]).mean(numeric_only=True).reset_index()
    per_patient.drop(columns=["fold", "seed"]).to_csv(args.out_dir / "context_25d_regions.csv", index=False)
    delta = deltas(per_patient, SCORES)
    delta.to_csv(args.out_dir / "context_25d_regions_delta.csv", index=False)
    labels = [c for c in CLASSES if c in set(per_patient.label)]
    draw(delta, labels, args.out_dir)
    end_delta = deltas(per_patient, END_SCORES)
    end_delta.to_csv(args.out_dir / "context_25d_ends_delta.csv", index=False)
    draw_ends(per_patient, end_delta, labels, args.out_dir)
    draw_end_scatter(per_patient, args.out_dir)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    log = {
        "runs_dir": str(args.runs_dir),
        "gt_pattern": GT_PATTERN,
        "edge_mm": EDGE_MM,
        "comparisons": COMPARISONS,
        "runs_scored": int(runs.groupby(["experiment", "fold", "seed"]).ngroups),
        "patients_per_experiment": per_patient.groupby("experiment").patient.nunique().to_dict(),
        "git_commit": commit,
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(),
    }
    (args.out_dir / "context_25d_regions_log.json").write_text(json.dumps(log, indent=2))
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(delta.round(4).to_string(index=False))
        print(end_delta.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
