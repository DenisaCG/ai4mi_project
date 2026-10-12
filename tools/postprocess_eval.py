"""Shared evaluation of post-processed 3D predictions of finished runs: the largest 6-connected component per
organ, the per-run and per-fold aggregation of the metrics, and the tables and CSVs written from them.

The seed ensemble (tools.seed_ensemble) and the probability post-processing (tools.prob_postprocess) use it.
Every variant row carries the run, patient, variant, organ and the metrics of src.metrics_3d; the variant named
"baseline" must reproduce each run's own eval/metrics_3d.csv (check_baseline).
"""

import csv
import hashlib
from pathlib import Path

import numpy as np
from scipy import ndimage

from src.config import REPO
from src.metrics_3d import METRICS, volume_metrics

HD95_DICE_TOLERANCE = 0.002  # the HD95 vote choice may lose this much Dice per organ


def largest_component(pred: np.ndarray, classes: list[int]) -> np.ndarray:
    """Keeps, per organ, only the largest 6-connected component of the volume."""
    out = pred.copy()
    for k in classes:
        labels, n = ndimage.label(out == k)
        if n > 1:
            largest = 1 + np.argmax(np.bincount(labels.ravel())[1:])
            out[(labels > 0) & (labels != largest)] = 0
    return out


def volume_rows(
    run_dir: Path,
    patient: str,
    variant: str,
    vol: np.ndarray,
    gt: np.ndarray,
    spacing: tuple[float, float, float],
    names: list[str],
    classes: list[int],
    scored: dict[tuple[int, bytes], dict],
) -> list[dict]:
    """One row per organ for the label volume `vol` of a patient. `scored` caches the metrics of organ
    masks already seen for this patient, so identical masks are scored once."""
    rows = []
    for k in classes:
        mask = vol == k
        false_positive = mask & ~(gt == k).any(axis=(0, 1))
        key = (k, hashlib.blake2b(np.packbits(mask).tobytes()).digest())
        if key not in scored:
            scored[key] = volume_metrics(gt == k, mask, spacing)
        rows.append(
            {
                "run": run_dir.relative_to(REPO / "runs").as_posix(),
                "patient": patient,
                "variant": variant,
                "class_idx": k,
                "class_name": names[k],
                "fp_slices": int(false_positive.any(axis=(0, 1)).sum()),
                "fp_voxels": int(false_positive.sum()),
                "pred_voxels": int(mask.sum()),
                **scored[key],
            }
        )
    return rows


def run_dirs(experiment: str, folds: int, seeds: list[int]) -> list[Path]:
    """Existing, finished run directories of `experiment`, fold-major."""
    dirs = [
        REPO / "runs" / f"{experiment}_fold{f}" / f"seed{s}"
        for f in range(folds)
        for s in seeds
    ]
    missing = [d for d in dirs if not (d / "eval" / ".done").exists()]
    if missing:
        raise FileNotFoundError(
            f"unfinished runs (no eval/.done): {[str(d) for d in missing]}"
        )
    return dirs


def check_baseline(rows: list[dict], dirs: list[Path]) -> float:
    """Largest |difference| of the baseline rows to each run's eval/metrics_3d.csv (Dice, HD95, ASSD)."""
    ours = {
        (r["run"], r["patient"], r["class_idx"]): r
        for r in rows
        if r["variant"] == "baseline"
    }
    worst = 0.0
    for d in dirs:
        with (d / "eval" / "metrics_3d.csv").open(newline="") as f:
            for row in csv.DictReader(f):
                mine = ours[
                    d.relative_to(REPO / "runs").as_posix(),
                    row["patient"],
                    int(row["class_idx"]),
                ]
                for m in METRICS:
                    a, b = float(row[m]), mine[m]
                    if np.isnan(a) != np.isnan(b):
                        return float("inf")
                    if not np.isnan(a):
                        worst = max(worst, abs(a - b))
    return worst


def mean(values) -> float:
    """NaN-aware mean; NaN if every value is NaN (as src.evaluate.mean)."""
    a = np.asarray(values, dtype=float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def variants_of(rows: list[dict]) -> list[str]:
    """Variant names in order of first appearance."""
    return list(dict.fromkeys(r["variant"] for r in rows))


def fold_of(run: str) -> int:
    """Fold index of a run name such as full_cv4_..._fold2/seed0."""
    return int(run.split("/")[0].rsplit("fold", 1)[1])


def aggregate(rows: list[dict], metric: str, classes: list[int]) -> dict[str, dict]:
    """variant -> {(scope, fold): mean}, scope "fg" or an organ index, fold "all" or a fold index. As
    src.aggregate: per run the organ mean over patients, fg the mean of the organs, then the mean over
    the runs of the fold (or of all folds)."""
    values: dict[tuple[str, str, int], list[float]] = {}
    for r in rows:
        values.setdefault((r["variant"], r["run"], r["class_idx"]), []).append(
            r[metric]
        )
    out = {}
    for variant in variants_of(rows):
        per_run = {
            run: {k: mean(values.get((variant, run, k), [])) for k in classes}
            for run in sorted({r["run"] for r in rows})
        }
        fg = {run: mean(list(v.values())) for run, v in per_run.items()}
        out[variant] = {}
        for fold in ["all", *sorted({fold_of(run) for run in per_run})]:
            sel = [run for run in per_run if fold == "all" or fold_of(run) == fold]
            out[variant]["fg", fold] = mean([fg[run] for run in sel])
            for k in classes:
                out[variant][k, fold] = mean([per_run[run][k] for run in sel])
    return out


def table(
    title: str,
    metric: str,
    rows: list[dict],
    classes: list[int],
    names: list[str],
    fmt: str,
) -> str:
    agg = aggregate(rows, metric, classes)
    folds = sorted({f for _, f in agg["baseline"] if f != "all"})
    head = [
        "variant",
        "fg",
        "fg vs baseline",
        *(f"{names[k]} (vs baseline)" for k in classes),
        *(f"fold {f} fg" for f in folds),
    ]
    lines = [
        f"### {title}",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    base = agg["baseline"]
    for variant, a in agg.items():
        cells = [
            format(a["fg", "all"], fmt),
            format(a["fg", "all"] - base["fg", "all"], "+" + fmt),
            *(
                f"{a[k, 'all']:{fmt}} ({a[k, 'all'] - base[k, 'all']:+{fmt}})"
                for k in classes
            ),
            *(format(a["fg", f], fmt) for f in folds),
        ]
        lines.append(f"| {variant} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def means_rows(rows: list[dict], classes: list[int], names: list[str]) -> list[dict]:
    """Long form of every table: one row per metric, variant, scope (fg or organ), fold (all or index)."""
    out = []
    for metric in METRICS:
        agg = aggregate(rows, metric, classes)
        for variant, a in agg.items():
            for (scope, fold), value in a.items():
                out.append(
                    {
                        "metric": metric,
                        "variant": variant,
                        "scope": scope if scope == "fg" else names[scope],
                        "fold": fold,
                        "mean": value,
                        "delta_to_baseline": value - agg["baseline"][scope, fold],
                    }
                )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
