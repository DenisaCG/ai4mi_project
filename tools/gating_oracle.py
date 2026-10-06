"""Oracle for a slice-presence head: what would perfect per-slice organ gating, a 3D largest connected
component per organ, or both, do to the 3D predictions of finished runs?

    python -m tools.gating_oracle --experiment full_cv4_resenc_ds_unet_dice_ce --seeds 0 1 2

Reads runs/<experiment>_fold<k>/seed<s>/volumes/val (the stitched predictions) and scores them against
the source GT with src.metrics_3d, the metrics of src.evaluate. Variants:
  baseline  the prediction as is (must reproduce the run's eval/metrics_3d.csv)
  gate      organ k set to background on every z slice where the GT has no voxel of organ k
  lcc       per organ, only the largest 6-connected component of the volume is kept
  gate_lcc  gate, then lcc
Writes, into --out: rows.csv (per run, patient, variant, organ), summary.md (fg and per-organ means,
deltas to the baseline, false-positive slices) and slivers.csv (slices whose GT organ area is at most T
pixels, on the 256x256 grid the network is trained on).
"""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage

from src.config import REPO, read_yaml
from src.metrics_3d import METRICS, volume_metrics

VARIANTS = ("baseline", "gate", "lcc", "gate_lcc")
SLIVER_AREAS = (1, 3, 5, 10, 25, 50)  # pixels on the training grid


def gate(pred: np.ndarray, gt: np.ndarray, classes: list[int]) -> np.ndarray:
    """Sets each organ to background on the z slices (last axis) where `gt` has none of it."""
    out = pred.copy()
    for k in classes:
        out[(out == k) & ~(gt == k).any(axis=(0, 1))] = 0
    return out


def largest_component(pred: np.ndarray, classes: list[int]) -> np.ndarray:
    """Keeps, per organ, only the largest 6-connected component of the volume."""
    out = pred.copy()
    for k in classes:
        labels, n = ndimage.label(out == k)
        if n > 1:
            largest = 1 + np.argmax(np.bincount(labels.ravel())[1:])
            out[(labels > 0) & (labels != largest)] = 0
    return out


def score_patient(task: tuple[str, str, str, list[str], list[int]]) -> list[dict]:
    """Scores the four variants of one patient of one run; one row per (variant, organ)."""
    run, pattern, patient, names, classes = task
    run_dir = Path(run)
    ref = nib.load(REPO / pattern.format(patient=patient))
    gt = np.asarray(ref.dataobj)
    spacing = tuple(float(s) for s in ref.header.get_zooms()[:3])
    pred = np.asarray(
        nib.load(run_dir / "volumes" / "val" / f"{patient}.nii.gz").dataobj
    )
    gated = gate(pred, gt, classes)
    volumes = {
        "baseline": pred,
        "gate": gated,
        "lcc": largest_component(pred, classes),
        "gate_lcc": largest_component(gated, classes),
    }
    rows = []
    for variant, vol in volumes.items():
        for k in classes:
            absent = ~(gt == k).any(axis=(0, 1))
            false_positive = (vol == k) & absent
            rows.append(
                {
                    "run": run_dir.relative_to(REPO / "runs").as_posix(),
                    "patient": patient,
                    "variant": variant,
                    "class_idx": k,
                    "class_name": names[k],
                    "fp_slices": int(false_positive.any(axis=(0, 1)).sum()),
                    "fp_voxels": int(false_positive.sum()),
                    "pred_voxels": int((vol == k).sum()),
                    **volume_metrics(gt == k, vol == k, spacing),
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


def slivers(dirs: list[Path], classes: list[int], names: list[str]) -> list[dict]:
    """Per organ and area threshold T: slices whose GT organ area is 1..T pixels, and how many of
    them are the first or last slice of that organ in its patient. One config per fold (seed 0)."""
    counts = {k: {"present": 0, "end": 0, "end_area": []} for k in classes}
    small = {
        k: {t: [0, 0] for t in SLIVER_AREAS} for k in classes
    }  # [all, at an organ end]
    for d in dirs:
        cfg = read_yaml((d / "config.yaml").read_text())
        root, scale = REPO / cfg["data"]["root"], cfg["data"]["label_scale"]
        by_patient: dict[str, list[np.ndarray]] = {}
        for path in sorted((root / "val" / "gt").glob("*.png")):
            by_patient.setdefault(path.stem.rsplit("_", 1)[0], []).append(
                np.array(Image.open(path)) // scale
            )
        for stack in by_patient.values():
            labels = np.stack(stack)  # (Z, H, W), slices in z order
            for k in classes:
                area = (labels == k).sum(axis=(1, 2))
                present = np.flatnonzero(area)
                if not len(present):
                    continue
                ends = {present[0], present[-1]}
                counts[k]["present"] += len(present)
                counts[k]["end"] += len(ends)
                counts[k]["end_area"] += [int(area[z]) for z in ends]
                for t in SLIVER_AREAS:
                    for z in present[area[present] <= t]:
                        small[k][t][0] += 1
                        small[k][t][1] += z in ends
    rows = []
    for k in classes:
        for t in SLIVER_AREAS:
            all_, at_end = small[k][t]
            rows.append(
                {
                    "class_name": names[k],
                    "max_area_px": t,
                    "present_slices": counts[k]["present"],
                    "sliver_slices": all_,
                    "sliver_slices_at_end": at_end,
                    "organ_end_slices": counts[k]["end"],
                    "median_end_area_px": float(np.median(counts[k]["end_area"])),
                }
            )
    return rows


def mean(values) -> float:
    """NaN-aware mean; NaN if every value is NaN (as src.evaluate.mean)."""
    a = np.asarray(values, dtype=float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def aggregate(rows: list[dict], metric: str, classes: list[int]) -> dict[str, dict]:
    """variant -> {"fg": mean, <organ index>: mean, "folds": [fg per fold]}, as src.aggregate: per run
    the organ mean over patients, fg the mean of the organs, then the mean over the runs."""
    out = {}
    for variant in VARIANTS:
        per_run: dict[str, dict[int, float]] = {}
        for run in sorted({r["run"] for r in rows}):
            sel = [r for r in rows if r["variant"] == variant and r["run"] == run]
            per_run[run] = {
                k: mean([r[metric] for r in sel if r["class_idx"] == k])
                for k in classes
            }
        fg = {run: mean(list(v.values())) for run, v in per_run.items()}
        folds = sorted({run.split("/")[0] for run in per_run})
        out[variant] = {
            "fg": mean(list(fg.values())),
            "folds": [mean([fg[r] for r in fg if r.startswith(f)]) for f in folds],
        } | {k: mean([v[k] for v in per_run.values()]) for k in classes}
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
    folds = len(agg["baseline"]["folds"])
    head = [
        "variant",
        "fg",
        *(names[k] for k in classes),
        *(f"fold {i}" for i in range(folds)),
        "fg vs baseline",
    ]
    lines = [
        f"### {title}",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    for variant in VARIANTS:
        a = agg[variant]
        cells = [a["fg"], *(a[k] for k in classes), *a["folds"]]
        lines.append(
            f"| {variant} | "
            + " | ".join(format(c, fmt) for c in cells)
            + f" | {a['fg'] - agg['baseline']['fg']:+{fmt}} |"
        )
    return "\n".join(lines)


def false_positive_table(rows: list[dict], classes: list[int], names: list[str]) -> str:
    lines = [
        "### False-positive slices (organ predicted on a slice where the GT has none), summed over all runs",
        "",
    ]
    lines += [
        "| variant | " + " | ".join(names[k] for k in classes) + " |",
        "|" + "---|" * (len(classes) + 1),
    ]
    for variant in VARIANTS:
        sums = [
            sum(
                r["fp_slices"]
                for r in rows
                if r["variant"] == variant and r["class_idx"] == k
            )
            for k in classes
        ]
        lines.append(f"| {variant} | " + " | ".join(map(str, sums)) + " |")
    return "\n".join(lines)


def undefined_table(rows: list[dict], classes: list[int], names: list[str]) -> str:
    """Rows whose HD95 is NaN because a variant emptied the prediction (src.evaluate drops them)."""
    lines = [
        "### Patient x run rows with undefined HD95 (empty prediction), summed over all runs",
        "",
    ]
    lines += [
        "| variant | " + " | ".join(names[k] for k in classes) + " |",
        "|" + "---|" * (len(classes) + 1),
    ]
    for variant in VARIANTS:
        nans = [
            sum(
                r["variant"] == variant and r["class_idx"] == k and np.isnan(r["hd95"])
                for r in rows
            )
            for k in classes
        ]
        lines.append(f"| {variant} | " + " | ".join(map(str, nans)) + " |")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="CV experiment name without the _fold<k> suffix",
    )
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, default=REPO / "results" / "gating_oracle")
    args = parser.parse_args(argv)

    dirs = run_dirs(args.experiment, args.folds, args.seeds)
    cfg = read_yaml((dirs[0] / "config.yaml").read_text())
    names, classes = cfg["data"]["class_names"], cfg["eval"]["classes"]
    pattern = cfg["data"]["source_pattern"]
    tasks = [
        (str(d), pattern, p.stem.removesuffix(".nii"), names, classes)
        for d in dirs
        for p in sorted((d / "volumes" / "val").glob("*.nii.gz"))
    ]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [
            row
            for patient_rows in pool.map(score_patient, tasks)
            for row in patient_rows
        ]

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "rows.csv", rows)
    sliver_rows = slivers(
        [d for d in dirs if d.name == f"seed{args.seeds[0]}"], classes, names
    )
    write_csv(args.out / "slivers.csv", sliver_rows)
    worst = check_baseline(rows, dirs)
    summary = [
        f"# Gating oracle: {args.experiment}, {len(dirs)} runs, {len(tasks)} patient volumes",
        "",
        f"Baseline rows vs the runs' own metrics_3d.csv: max |difference| {worst:.2e} (must be ~0).",
        "Means as src.aggregate: organ mean over patients per run, fg mean of the organs, mean over runs.",
        "",
        table("fg Dice and per-organ Dice", "dice", rows, classes, names, ".4f"),
        "",
        table("fg HD95 and per-organ HD95 (mm)", "hd95", rows, classes, names, ".2f"),
        "",
        table("fg ASSD and per-organ ASSD (mm)", "assd", rows, classes, names, ".3f"),
        "",
        false_positive_table(rows, classes, names),
        "",
        undefined_table(rows, classes, names),
        "",
        "### Sliver slices (GT organ area of at most T pixels on the 256x256 training grid; slivers.csv)",
        "",
        "| organ | T px | sliver slices | at an organ end | all present slices | organ end slices | median end area px |",
        "|---|---|---|---|---|---|---|",
        *(
            f"| {r['class_name']} | {r['max_area_px']} | {r['sliver_slices']} | {r['sliver_slices_at_end']} "
            f"| {r['present_slices']} | {r['organ_end_slices']} | {r['median_end_area_px']:.0f} |"
            for r in sliver_rows
        ),
    ]
    (args.out / "summary.md").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
