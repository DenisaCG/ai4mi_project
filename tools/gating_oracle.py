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
  zrun_<L>                         organ removed on runs of fewer than L consecutive slices
  zextent_<m>                      organ kept only within the z-range of its largest component +- m slices
  slice_cc_<r>                     per slice, organ pieces smaller than r x the largest piece removed
  contiguous_z_<g>                 organ kept only on its longest run of slices, runs joined across gaps of <= g slices
  trachea_anchor_<R>               esophagus components whose centroid is more than R training-grid pixels from the
                                   trachea centroid of that slice (nearest slice with trachea) removed
  combo                            per organ, the ground-truth-free variant with the best mean Dice (or none)
  combo_hd95                       same by lowest HD95, among variants losing at most 0.002 Dice
--variants restricts the run to the named variants (baseline is always scored).
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
    contiguous_z,
    heart_hull,
    min_run_length,
    remove_small_components,
    slice_components,
    take_organs,
    trachea_anchor,
    z_extent,
)

ORACLE = ("gate", "gate_lcc")  # need the ground truth: excluded from the combo
SLIVER_AREAS = (1, 3, 5, 10, 25, 50)  # pixels on the training grid
AREA_GATES = (5, 10, 25, 50)  # pixels on the training grid
SIZE_FILTERS = (("0.2x", 0.2, 0), ("0.05x", 0.05, 0), ("500vox", 0.0, 500))
RUN_LENGTHS = (3, 5)  # slices
Z_MARGINS = (0, 5)  # slices
SLICE_COMPONENTS = (0.1, 0.25)  # of the largest piece on the slice
CONTIGUOUS_GAPS = (0, 2, 5)  # slices
ANCHOR_RADII = (20, 40, 60)  # pixels of the 256x256 training grid
HD95_DICE_TOLERANCE = 0.002  # combo_hd95 may lose this much Dice per organ
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
    only: tuple[str, ...] | None = None,
) -> Iterator[tuple[str, np.ndarray]]:
    """Yields (variant, volume) for every variant of the module docstring, baseline first; with `only`,
    just those (and the baseline). `voxels_per_px` is the number of voxels of a slice of `pred` that one
    pixel of the 256x256 training grid covers."""
    eso, heart, trachea = (names.index(n) for n in ("esophagus", "heart", "trachea"))
    lcc = largest_component(pred, classes)
    candidates = {
        "gate": lambda: gate(pred, gt, classes),
        "lcc": lambda: lcc,
        "gate_lcc": lambda: largest_component(gate(pred, gt, classes), classes),
        **{f"lcc_{names[k]}": partial(take_organs, pred, lcc, [k]) for k in classes},
        "lcc_heart_trachea": lambda: take_organs(pred, lcc, [heart, trachea]),
        **{
            f"esophagus_size_{label}": partial(remove_small_components, pred, eso, rel, min_voxels)
            for label, rel, min_voxels in SIZE_FILTERS
        },
        "heart_hull": lambda: heart_hull(take_organs(pred, lcc, [heart]), heart),
        **{f"area_gate_{n}": partial(area_gate, pred, classes, n * voxels_per_px) for n in AREA_GATES},
        "adjacent_gate": lambda: adjacent_gate(pred, classes),
        **{f"zrun_{n}": partial(min_run_length, pred, classes, n) for n in RUN_LENGTHS},
        **{f"zextent_{m}": partial(z_extent, pred, classes, m) for m in Z_MARGINS},
        **{f"slice_cc_{r}": partial(slice_components, pred, classes, r) for r in SLICE_COMPONENTS},
        **{f"contiguous_z_{g}": partial(contiguous_z, pred, classes, g) for g in CONTIGUOUS_GAPS},
        **{
            f"trachea_anchor_{r}": partial(trachea_anchor, pred, eso, trachea, r * voxels_per_px**0.5)
            for r in ANCHOR_RADII
        },
    }
    yield "baseline", pred
    for name, build in candidates.items():
        if only is None or name in only:
            yield name, build()


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
    reference: np.ndarray | None = None,
) -> list[dict]:
    """One row per organ for the label volume `vol` of a patient. `scored` caches the metrics of organ
    masks already seen for this patient, so identical masks are scored once. With `reference` (the
    baseline prediction) each row also has gt_voxels_lost: GT voxels the reference had right and `vol` dropped."""
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
            | (
                {}
                if reference is None
                else {"gt_voxels_lost": int(((gt == k) & (reference == k) & ~mask).sum())}
            )
        )
    return rows


def score_patient(
    task: tuple[str, str, str, list[str], list[int], float, dict[str, dict[int, str]] | None, tuple[str, ...] | None],
) -> list[dict]:
    """Scores the variants (`only`, default all) of one patient of one run; one row per (variant, organ).
    With `choices` ({combo name: {organ: variant}}) only the combos are scored, taking each organ from its
    chosen variant. An organ mask already scored for this patient (the same voxels) reuses its metrics."""
    run, pattern, patient, names, classes, px_mm2, choices, only = task
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
        rows.extend(
            volume_rows(run_dir, patient, variant, vol, gt, spacing, names, classes, scored, pred)
        )

    combos = {name: np.zeros_like(pred) for name in choices or {}}
    for variant, vol in variant_volumes(pred, gt, classes, names, voxels_per_px, only):
        if choices is None:
            score(variant, vol)
        else:
            for name, chosen in choices.items():
                for k in classes:
                    if chosen[k] == variant:
                        combos[name][vol == k] = k
    for name, vol in combos.items():
        score(name, vol)
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


def choose_per_organ(
    rows: list[dict], classes: list[int], by: str = "dice"
) -> dict[int, str]:
    """Per organ, the ground-truth-free variant that is best over all runs; the baseline unless a variant
    is strictly better. by="dice": highest mean Dice. by="hd95": lowest mean HD95 among the variants that
    lose at most HD95_DICE_TOLERANCE Dice."""
    free = [r for r in rows if r["variant"] not in ORACLE]
    dice, hd95 = aggregate(free, "dice", classes), aggregate(free, "hd95", classes)
    choices = {}
    for k in classes:
        best = "baseline"
        for variant in dice:
            if by == "dice":
                better = dice[variant][k, "all"] > dice[best][k, "all"]
            else:
                ok = (
                    dice[variant][k, "all"]
                    >= dice["baseline"][k, "all"] - HD95_DICE_TOLERANCE
                )
                better = ok and hd95[variant][k, "all"] < hd95[best][k, "all"]
            if better:
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
    choices: dict[str, dict[int, str]],
    rows: list[dict],
    classes: list[int],
    names: list[str],
) -> str:
    dice = aggregate(rows, "dice", classes)
    hd95 = aggregate(rows, "hd95", classes)
    lines = [
        "### Combos: variant chosen per organ among the ground-truth-free variants (combo by mean Dice, combo_hd95 by mean HD95)",
        "",
        "| combo | organ | chosen | Dice change | HD95 change (mm) |",
        "|---|---|---|---|---|",
    ]
    for combo, chosen in choices.items():
        for k in classes:
            v = chosen[k]
            lines.append(
                f"| {combo} | {names[k]} | {v} | {dice[v][k, 'all'] - dice['baseline'][k, 'all']:+.4f} "
                f"| {hd95[v][k, 'all'] - hd95['baseline'][k, 'all']:+.2f} |"
            )
    return "\n".join(lines)


def esophagus_table(rows: list[dict], classes: list[int], names: list[str]) -> str:
    """The esophagus alone (its pieces may be real): metric changes against the baseline, the GT voxels
    the variant removed from the baseline prediction (summed over all runs) and the rows it emptied."""
    k = names.index("esophagus")
    agg = {m: aggregate(rows, m, classes) for m in METRICS}
    base = {
        (r["run"], r["patient"]): r for r in rows if r["variant"] == "baseline" and r["class_idx"] == k
    }
    lines = [
        "### Esophagus alone",
        "",
        "| variant | Dice | Dice change | HD95 mm | HD95 change | ASSD mm | ASSD change | GT voxels removed | rows emptied |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for variant in agg["dice"]:
        sel = [r for r in rows if r["variant"] == variant and r["class_idx"] == k]
        cells = []
        for m, fmt in (("dice", ".4f"), ("hd95", ".2f"), ("assd", ".3f")):
            v, b = agg[m][variant][k, "all"], agg[m]["baseline"][k, "all"]
            cells += [format(v, fmt), format(v - b, "+" + fmt)]
        lost = sum(r.get("gt_voxels_lost", 0) for r in sel)
        gone = sum(r["pred_voxels"] == 0 and base[r["run"], r["patient"]]["pred_voxels"] > 0 for r in sel)
        lines.append(f"| {variant} | " + " | ".join(cells) + f" | {lost} | {gone} |")
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
    parser.add_argument(
        "--variants", nargs="+", help="score only these variants (baseline always)"
    )
    parser.add_argument("--out", type=Path, default=REPO / "results" / "gating_oracle")
    args = parser.parse_args(argv)

    dirs = run_dirs(args.experiment, args.folds, args.seeds)
    cfgs = {d: read_yaml((d / "config.yaml").read_text()) for d in dirs}
    cfg = cfgs[dirs[0]]
    only = tuple(args.variants) if args.variants else None
    names, classes = cfg["data"]["class_names"], cfg["eval"]["classes"]
    pattern = cfg["data"]["source_pattern"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cache").mkdir(exist_ok=True)
    score = partial(
        cached_score, cache=args.out / "cache", signature=source_signature()
    )

    def run_all(choices: dict[str, dict[int, str]] | None) -> list[dict]:
        tasks = [
            (
                str(d),
                pattern,
                p.stem.removesuffix(".nii"),
                names,
                classes,
                train_pixel_mm2(cfgs[d]),
                choices,
                only,
            )
            for d in dirs
            for p in sorted((d / "volumes" / "val").glob("*.nii.gz"))
        ]
        with ProcessPoolExecutor(args.workers) as pool:
            return [row for rows in pool.map(score, tasks) for row in rows]

    rows = run_all(None)
    choices = {
        "combo": choose_per_organ(rows, classes),
        "combo_hd95": choose_per_organ(rows, classes, by="hd95"),
    }
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
        esophagus_table(rows, classes, names),
        "",
        count_table(
            "GT voxels of the baseline prediction removed by the variant (true positives lost), summed over all runs",
            rows,
            classes,
            names,
            lambda r, b: r["gt_voxels_lost"],
        ),
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
