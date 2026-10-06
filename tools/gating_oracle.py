"""Oracle for a slice-presence head: what would perfect per-slice organ gating, a 3D largest connected
component per organ, or both, do to the 3D predictions of finished runs?

    python -m tools.gating_oracle --experiment full_cv4_resenc_ds_unet_dice_ce --seeds 0 1 2

Reads runs/<experiment>_fold<k>/seed<s>/volumes/val (the stitched predictions) and scores them against
the source GT with src.metrics_3d, the metrics of src.evaluate. Variants:
  baseline  the prediction as is (must reproduce the run's eval/metrics_3d.csv)
  gate      organ k set to background on every z slice where the GT has no voxel of organ k
  lcc       per organ, only the largest 6-connected component of the volume is kept
  gate_lcc  gate, then lcc
Ground-truth-free variants, each applied independently to the baseline predictions (tools.postprocess_filters):
  lcc_<organ>, lcc_heart_trachea   largest component for one organ / for heart and trachea only
  esophagus_size_<r>               esophagus components below r x the largest, or below 500 voxels, removed
  heart_hull                       largest heart component, then its 3D convex hull (into background only)
  area_gate_<N>                    organ set to background on slices where it covers < N pixels of the
                                   256x256 training grid (converted to mm^2 through the run's ROI crop size)
  adjacent_gate                    organ removed on slices where it is absent from z-1 and z+1
  combo                            per organ, the ground-truth-free variant with the best mean Dice (or none)
Writes, into --out: rows.csv (per run, patient, variant, organ), means.csv (fg and per-organ means, overall
and per fold, with deltas to the baseline), summary.md and slivers.csv (slices whose GT organ area is at
most T pixels, on the 256x256 grid the network is trained on). cache/ holds per-patient results so a
timed-out job resumes; it is keyed on the source of both tools modules.
"""

import argparse
import csv
import hashlib
import json
import pickle
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage

from src.config import REPO, read_yaml
from src.metrics_3d import METRICS, volume_metrics
from tools.postprocess_filters import (
    adjacent_gate,
    area_gate,
    heart_hull,
    remove_small_components,
    take_organs,
)

ORACLE = ("gate", "gate_lcc")  # need the ground truth: excluded from the combo
SLIVER_AREAS = (1, 3, 5, 10, 25, 50)  # pixels on the training grid
AREA_GATES = (5, 10, 25, 50)  # pixels on the training grid
SIZE_FILTERS = (("0.2x", 0.2, 0), ("0.05x", 0.05, 0), ("500vox", 0.0, 500))
LONG_HD95 = 20.0  # mm, "bad" patient x organ rows


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


def variant_volumes(
    pred: np.ndarray,
    gt: np.ndarray,
    classes: list[int],
    names: list[str],
    voxels_per_px: float,
) -> Iterator[tuple[str, np.ndarray]]:
    """Yields (variant, volume) for every variant of the module docstring, baseline first.
    `voxels_per_px` is the number of voxels of a slice of `pred` that one pixel of the 256x256 training
    grid covers."""
    eso, heart, trachea = (names.index(n) for n in ("esophagus", "heart", "trachea"))
    gated = gate(pred, gt, classes)
    lcc = largest_component(pred, classes)
    yield "baseline", pred
    yield "gate", gated
    yield "lcc", lcc
    yield "gate_lcc", largest_component(gated, classes)
    for k in classes:
        yield f"lcc_{names[k]}", take_organs(pred, lcc, [k])
    yield "lcc_heart_trachea", take_organs(pred, lcc, [heart, trachea])
    for label, rel, min_voxels in SIZE_FILTERS:
        yield f"esophagus_size_{label}", remove_small_components(pred, eso, rel, min_voxels)
    yield "heart_hull", heart_hull(take_organs(pred, lcc, [heart]), heart)
    for n in AREA_GATES:
        yield f"area_gate_{n}", area_gate(pred, classes, n * voxels_per_px)
    yield "adjacent_gate", adjacent_gate(pred, classes)


def score_patient(
    task: tuple[str, str, str, list[str], list[int], float, dict[int, str] | None],
) -> list[dict]:
    """Scores the variants of one patient of one run; one row per (variant, organ). With `choices`
    ({organ: variant}) only the combo is scored, taking each organ from its chosen variant. An organ
    mask already scored for this patient (the same voxels) reuses its metrics."""
    run, pattern, patient, names, classes, px_mm2, choices = task
    run_dir = Path(run)
    ref = nib.load(REPO / pattern.format(patient=patient))
    gt = np.asarray(ref.dataobj)
    spacing = tuple(float(s) for s in ref.header.get_zooms()[:3])
    pred = np.asarray(
        nib.load(run_dir / "volumes" / "val" / f"{patient}.nii.gz").dataobj
    )
    voxels_per_px = px_mm2 / (spacing[0] * spacing[1])
    scored: dict[tuple[int, bytes], dict] = {}
    rows = []

    def score(variant: str, vol: np.ndarray) -> None:
        for k in classes:
            mask = vol == k
            absent = ~(gt == k).any(axis=(0, 1))
            false_positive = mask & absent
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

    combo = np.zeros_like(pred)
    for variant, vol in variant_volumes(pred, gt, classes, names, voxels_per_px):
        if choices is None:
            score(variant, vol)
        else:
            for k in classes:
                if choices[k] == variant:
                    combo[vol == k] = k
    if choices is not None:
        score("combo", combo)
    return rows


def train_pixel_mm2(cfg: dict) -> float:
    """Area in mm^2 of one pixel of the 256x256 training grid: the ROI crop window (data.preprocess.crop,
    in voxels of the resampled in-plane grid) is resized to that grid."""
    crop = json.loads((REPO / cfg["data"]["root"] / "roi_crop.json").read_text())
    side = crop["size"] * crop["target_spacing"][0] / cfg["data"]["preprocess"]["shape"][0]
    return side**2


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
            run: {k: mean(values[variant, run, k]) for k in classes}
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


def choose_per_organ(rows: list[dict], classes: list[int]) -> dict[int, str]:
    """Per organ, the ground-truth-free variant with the best mean Dice over all runs; the baseline
    unless a variant is strictly better."""
    agg = aggregate(
        [r for r in rows if r["variant"] not in ORACLE], "dice", classes
    )
    choices = {}
    for k in classes:
        best = "baseline"
        for variant, a in agg.items():
            if a[k, "all"] > agg[best][k, "all"]:
                best = variant
        choices[k] = best
    return choices


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


def recovery_table(rows: list[dict], classes: list[int]) -> str:
    """Share of the oracle gate's fg gain that each ground-truth-free variant recovers."""
    dice = aggregate(rows, "dice", classes)
    hd95 = aggregate(rows, "hd95", classes)
    gain = {
        m: a["gate"]["fg", "all"] - a["baseline"]["fg", "all"]
        for m, a in (("dice", dice), ("hd95", hd95))
    }
    lines = [
        f"### Share of the oracle gate's gain recovered (fg Dice {gain['dice']:+.4f}, fg HD95 {gain['hd95']:+.2f} mm)",
        "",
        "| variant | fg Dice change | share | fg HD95 change (mm) | share |",
        "|---|---|---|---|---|",
    ]
    for variant in dice:
        if variant in ("baseline", *ORACLE):
            continue
        d = dice[variant]["fg", "all"] - dice["baseline"]["fg", "all"]
        h = hd95[variant]["fg", "all"] - hd95["baseline"]["fg", "all"]
        lines.append(
            f"| {variant} | {d:+.4f} | {d / gain['dice']:.0%} | {h:+.2f} | {h / gain['hd95']:.0%} |"
        )
    return "\n".join(lines)


def count_table(
    title: str, rows: list[dict], classes: list[int], names: list[str], count
) -> str:
    """One line per variant, one column per organ and a total: `count(row, baseline_row)` summed."""
    base = {
        (r["run"], r["patient"], r["class_idx"]): r
        for r in rows
        if r["variant"] == "baseline"
    }
    head = ["variant", *(names[k] for k in classes), "all"]
    lines = [f"### {title}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for variant in variants_of(rows):
        sums = [
            sum(
                count(r, base[r["run"], r["patient"], k])
                for r in rows
                if r["variant"] == variant and r["class_idx"] == k
            )
            for k in classes
        ]
        lines.append(f"| {variant} | " + " | ".join(map(str, [*sums, sum(sums)])) + " |")
    return "\n".join(lines)


def combo_table(
    choices: dict[int, str], rows: list[dict], classes: list[int], names: list[str]
) -> str:
    dice = aggregate(rows, "dice", classes)
    hd95 = aggregate(rows, "hd95", classes)
    lines = [
        "### Combo: variant chosen per organ (best mean Dice over all runs among the ground-truth-free variants)",
        "",
        "| organ | chosen | Dice change | HD95 change (mm) |",
        "|---|---|---|---|",
    ]
    for k in classes:
        v = choices[k]
        lines.append(
            f"| {names[k]} | {v} | {dice[v][k, 'all'] - dice['baseline'][k, 'all']:+.4f} "
            f"| {hd95[v][k, 'all'] - hd95['baseline'][k, 'all']:+.2f} |"
        )
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


def emptied(row: dict, baseline: dict) -> int:
    """1 if the variant row has an undefined HD95 (empty prediction) where the baseline row has one."""
    return int(np.isnan(row["hd95"]) and not np.isnan(baseline["hd95"]))


def source_signature() -> str:
    """Hash of the code that decides a cached result: this module, the filters and the metrics."""
    digest = hashlib.blake2b()
    for path in (Path(__file__), Path(__file__).with_name("postprocess_filters.py"), REPO / "src" / "metrics_3d.py"):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def cached_score(task: tuple, cache: Path, signature: str) -> list[dict]:
    """score_patient(task), read from or written to `cache` so that a timed-out job resumes."""
    key = hashlib.blake2b((repr(task) + signature).encode()).hexdigest()[:24]
    path = cache / f"{key}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    rows = score_patient(task)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(rows))
    tmp.replace(path)
    return rows


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
    cfgs = {d: read_yaml((d / "config.yaml").read_text()) for d in dirs}
    cfg = cfgs[dirs[0]]
    names, classes = cfg["data"]["class_names"], cfg["eval"]["classes"]
    pattern = cfg["data"]["source_pattern"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cache").mkdir(exist_ok=True)
    score = partial(
        cached_score, cache=args.out / "cache", signature=source_signature()
    )

    def run_all(choices: dict[int, str] | None) -> list[dict]:
        tasks = [
            (
                str(d),
                pattern,
                p.stem.removesuffix(".nii"),
                names,
                classes,
                train_pixel_mm2(cfgs[d]),
                choices,
            )
            for d in dirs
            for p in sorted((d / "volumes" / "val").glob("*.nii.gz"))
        ]
        with ProcessPoolExecutor(args.workers) as pool:
            return [row for rows in pool.map(score, tasks) for row in rows]

    rows = run_all(None)
    choices = choose_per_organ(rows, classes)
    rows += run_all(choices)
    n_volumes = len(rows) // (len(variants_of(rows)) * len(classes))

    write_csv(args.out / "rows.csv", rows)
    write_csv(args.out / "means.csv", means_rows(rows, classes, names))
    sliver_rows = slivers(
        [d for d in dirs if d.name == f"seed{args.seeds[0]}"], classes, names
    )
    write_csv(args.out / "slivers.csv", sliver_rows)
    worst = check_baseline(rows, dirs)
    summary = [
        f"# Post-processing variants and gating oracle: {args.experiment}, {len(dirs)} runs, {n_volumes} patient volumes",
        "",
        f"Baseline rows vs the runs' own metrics_3d.csv: max |difference| {worst:.2e} (must be ~0).",
        "Means as src.aggregate: organ mean over patients per run, fg mean of the organs, mean over runs.",
        "gate and gate_lcc use the ground truth (oracle); every other variant does not.",
        "Optimistic: the area thresholds, size filters and the per-organ combo are chosen on the same validation folds that are scored.",
        f"area_gate pixel = one pixel of the 256x256 training grid: {sorted({round(train_pixel_mm2(c), 3) for c in cfgs.values()})} mm^2 (by fold crop size).",
        "",
        table("fg Dice and per-organ Dice", "dice", rows, classes, names, ".4f"),
        "",
        table("fg HD95 and per-organ HD95 (mm)", "hd95", rows, classes, names, ".2f"),
        "",
        table("fg ASSD and per-organ ASSD (mm)", "assd", rows, classes, names, ".3f"),
        "",
        recovery_table(rows, classes),
        "",
        combo_table(choices, rows, classes, names),
        "",
        count_table(
            f"Patient x run rows with HD95 above {LONG_HD95:.0f} mm (the baseline row is 'before')",
            rows,
            classes,
            names,
            lambda r, b: int(r["hd95"] > LONG_HD95),
        ),
        "",
        count_table(
            "False-positive slices (organ predicted on a slice where the GT has none), summed over all runs",
            rows,
            classes,
            names,
            lambda r, b: r["fp_slices"],
        ),
        "",
        count_table(
            "Patient x run rows emptied by the variant (HD95 undefined, src.evaluate would drop them)",
            rows,
            classes,
            names,
            emptied,
        ),
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
