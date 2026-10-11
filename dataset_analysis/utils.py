"""Shared I/O and explicit measurement conventions for SegTHOR exploration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone

import nibabel as nib
import numpy as np
from PIL import Image

CLASSES = {1: "esophagus", 2: "heart", 3: "trachea", 4: "aorta"}
NAMES = {k: name.capitalize() for k, name in CLASSES.items()}
REPO = Path(__file__).resolve().parents[1]
N_FOLDS, N_SEEDS = 4, 3  # cross-validation protocol, as in src.aggregate
METRICS = ("dice", "hd95", "assd")
INVENTORY = REPO / "deck/snellius_inventory/igardner1/arch_summary.csv"
PER_PATIENT = REPO / "deck/snellius_inventory/scur0049/arch_per_patient.csv"

# Figures share the repo-wide theme in tools/plot_style.py, with its earthy organ colors.
os.environ.setdefault("MPLCONFIGDIR", str(REPO / "dataset_analysis/results/.matplotlib"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
sys.path.insert(0, str(REPO / "tools"))
from plot_style import (BACKGROUND_COLOR as BACKGROUND, EARTH_LABEL_COLORS as COLORS,  # noqa: E402,F401
                        FOOTNOTE_COLOR as MUTED, GRID_COLOR, apply_style, decorate, tint)

INK = "#333333"
FIGURES = ("class_distribution.png", "shape_descriptors_3d.png",
           "shape_descriptors_summary.png", "target_area_through_scan.png",
           "baseline_3d_dice_by_class.png", "baseline_dice_vs_target_size.png",
           "baseline_dice_by_organ_position.png")
OLD_FIGURES = ("original_class_frequency.png", "original_patient_volume_extent.png",
               "original_normalized_extent.png", "processed_positive_area_distribution.png",
               "processed_area_vs_z.png", "baseline_positive_dice_distribution.png",
               "baseline_dice_vs_area_scatter.png", "baseline_dice_vs_area_binned.png",
               "baseline_dice_vs_scan_z.png", "baseline_dice_vs_organ_z.png",
               "baseline_patient_performance.png")


def pyplot():
    """matplotlib.pyplot with the shared theme applied."""
    apply_style()
    import matplotlib.pyplot as plt
    return plt


def clean_axis(ax, grid_axis: str = "y"):
    """Only light reference lines along one axis; no ticks, box or competing color coding."""
    ax.set_axisbelow(True)
    ax.grid(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=.8)
    ax.tick_params(length=0, pad=6)


def save(fig, output: Path, name: str, plt):
    if name not in FIGURES:
        raise ValueError(f"Uncurated figure name: {name}")
    fig.savefig(output / "plots" / name)
    plt.close(fig)


def remove_superseded(output: Path):
    """Delete only named v1 generated graphics inside the selected result folder."""
    for name in OLD_FIGURES:
        (output / "plots" / name).unlink(missing_ok=True)
    for name in ("esophagus_examples.png", "heart_examples.png", "trachea_examples.png"):
        (output / "examples" / name).unlink(missing_ok=True)
    (output / "tables/baseline_examples.csv").unlink(missing_ok=True)


def parser(description: str) -> argparse.ArgumentParser:
    """Paths default to this checkout, independent of the current directory."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--repo-root", type=Path, default=REPO)
    p.add_argument("--original-data", type=Path)
    p.add_argument("--processed-data", type=Path)
    p.add_argument("--output-dir", type=Path)
    return p


def paths(args):
    """Restrict writes to an analysis subdirectory, never an input dataset."""
    root = args.repo_root.resolve()
    original = (args.original_data or root / "data/segthor_part1/train").resolve()
    processed = (args.processed_data or root / "data/SEGTHOR").resolve()
    output = (args.output_dir or root / "dataset_analysis/results").resolve()
    analysis = root / "dataset_analysis/results"
    if not output.is_relative_to(analysis):
        raise ValueError("--output-dir must be inside dataset_analysis/results")
    for source in (original, processed):
        if not source.is_dir():
            raise FileNotFoundError(source)
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Output and input directories must not overlap")
    for sub in ("tables", "plots", "examples"):
        (output / sub).mkdir(parents=True, exist_ok=True)
    return root, original, processed, output


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write missing or undefined numeric values as blank CSV cells."""
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if v is None or isinstance(v, (float, np.floating))
                             and not np.isfinite(v) else v for k, v in row.items()})


def load_cv_runs(metrics_dir: Path, experiment: str, keys=("dice",)) -> list[dict]:
    """Read the validation metrics of every run of a 4-fold x 3-seed experiment.

    Args:
        metrics_dir: Folder holding one `<experiment>_fold<k>/seed<s>/metrics_3d.csv` per run.
        experiment: Experiment name without the fold suffix.
        keys: Columns of metrics_3d.csv to read, as floats.

    Returns:
        One row per fold, seed, patient and organ with the requested metrics.
    """
    rows = []
    for fold in range(N_FOLDS):
        for seed in range(N_SEEDS):
            path = metrics_dir / f"{experiment}_fold{fold}" / f"seed{seed}" / "metrics_3d.csv"
            for r in read_csv(path):
                if r["split"] == "val":
                    rows.append({"fold": fold, "seed": seed, "patient": r["patient"], "organ": r["class_name"],
                                 **{k: float(r[k]) for k in keys}})
    return rows


def read_csv(path: Path) -> list[dict]:
    """Load generated tables, retaining strings until explicitly converted."""
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_arch_runs(metrics_dir: Path, experiment: str) -> tuple[list[dict], str]:
    """Read the validation metrics of every run of an experiment, from the local metrics or else the Snellius inventory.

    Args:
        metrics_dir: Folder holding the local runs, as for load_cv_runs.
        experiment: Experiment name without the fold suffix.

    Returns:
        The rows of load_cv_runs (all of METRICS) and the name of the source they came from.
    """
    if (metrics_dir / f"{experiment}_fold0").exists():
        return load_cv_runs(metrics_dir, experiment, METRICS), "metrics"
    rows = [
        {"fold": int(r["fold"]), "seed": int(r["seed"]), "patient": r["patient"], "organ": r["organ"],
         **{k: float(r[k]) for k in METRICS}}
        for r in read_csv(PER_PATIENT) if r["experiment"] == experiment
    ]
    return rows, "snellius inventory"


def spreads(runs: dict[tuple[int, int], float]) -> tuple[float, float]:
    """Standard deviation of a per-run value over seeds and over folds.

    Args:
        runs: Value of every run, keyed by (fold, seed).

    Returns:
        Sample std over seeds of the fold-averaged value, and over folds of the seed-averaged value; NaN where
        fewer than two seeds or folds are present.
    """
    folds = sorted({f for f, _ in runs})
    seeds = sorted({s for _, s in runs})
    by_seed = [statistics.fmean(runs[f, s] for f in folds) for s in seeds]
    by_fold = [statistics.fmean(runs[f, s] for s in seeds) for f in folds]
    return tuple(statistics.stdev(v) if len(v) > 1 else float("nan") for v in (by_seed, by_fold))


def summarise(rows: list[dict]) -> dict:
    """Mean over the runs of the per-run patient means, for the mean over organs and the esophagus.

    Args:
        rows: Validation metrics of one experiment, from load_cv_runs (all of METRICS read).

    Returns:
        The metrics of the mean over organs, plus the esophagus Dice and the seed and fold std of the Dice, the HD95
        and the esophagus Dice.
    """
    cells = defaultdict(list)
    for r in rows:
        for key in METRICS:
            cells[key, r["fold"], r["seed"], r["organ"]].append(r[key])
    runs = {key: defaultdict(dict) for key in METRICS}
    for (key, fold, seed, organ), values in cells.items():
        runs[key][fold, seed][organ] = statistics.fmean(values)
    mean_runs = {key: {run: statistics.fmean(v.values()) for run, v in runs[key].items()} for key in METRICS}
    esophagus_runs = {run: v["esophagus"] for run, v in runs["dice"].items()}
    out = {key: statistics.fmean(mean_runs[key].values()) for key in METRICS}
    out["n_runs"] = len(mean_runs["dice"])
    out["esophagus_dice"] = statistics.fmean(esophagus_runs.values())
    out["dice_seed_std"], out["dice_fold_std"] = spreads(mean_runs["dice"])
    out["hd95_seed_std"], out["hd95_fold_std"] = spreads(mean_runs["hd95"])
    out["esophagus_seed_std"], out["esophagus_fold_std"] = spreads(esophagus_runs)
    return out


def inventory_row(experiment: str) -> dict:
    """Mean metrics of one experiment from the Snellius inventory, in the layout of summarise()."""
    rows = [r for r in read_csv(INVENTORY) if r["experiment"] == experiment]
    mean = next(r for r in rows if r["organ"] == "mean")
    esophagus = next(r for r in rows if r["organ"] == "esophagus")
    return {
        **{key: float(mean[key]) for key in METRICS},
        "n_runs": int(mean["n_runs"]),
        "esophagus_dice": float(esophagus["dice"]),
        "dice_seed_std": float(mean["seed_std"] or "nan"),
        "dice_fold_std": float(mean["fold_std"] or "nan"),
        "hd95_seed_std": None,
        "hd95_fold_std": None,
        "esophagus_seed_std": None,
        "esophagus_fold_std": None,
    }


def identity(path: Path) -> tuple[str, int]:
    match = re.fullmatch(r"(Patient_\d+)_(\d{4})", path.stem)
    if match is None:
        raise ValueError(f"Unexpected slice filename: {path}")
    return match[1], int(match[2])


def discover(processed: Path) -> dict:
    """Pair by exact filename and reject patient overlap or missing image/mask pairs."""
    patients = {}
    for split in ("train", "val"):
        images = {p.name: p for p in (processed / split / "img").glob("*.png")}
        masks = {p.name: p for p in (processed / split / "gt").glob("*.png")}
        if not images or images.keys() != masks.keys():
            raise ValueError(f"Missing/mismatched image and GT files in {split}")
        for name in sorted(masks):
            patient, z = identity(masks[name])
            entry = patients.setdefault(patient, {"split": split, "slices": {}})
            if entry["split"] != split or z in entry["slices"]:
                raise ValueError(f"Patient leakage or duplicate index: {patient}, {z}")
            entry["slices"][z] = (images[name], masks[name])
    return dict(sorted(patients.items()))


def load_png(path: Path) -> np.ndarray:
    """Decode exact 63-spaced labels, one per class in CLASSES plus background."""
    with Image.open(path) as img:
        a = np.asarray(img)
    allowed = np.array([0] + [63 * k for k in CLASSES])
    if a.ndim != 2 or a.dtype != np.uint8 or not np.isin(a, allowed).all():
        raise ValueError(f"Invalid PNG label encoding: {path}")
    return a // 63


def load_original(folder: Path, patient: str):
    """Validate CT/GT geometry and units before interpreting physical quantities."""
    ct = nib.load(folder / patient / f"{patient}.nii.gz")
    gt = nib.load(folder / patient / "GT.nii.gz")
    if len(gt.shape) != 3 or ct.shape != gt.shape or not np.allclose(ct.affine, gt.affine):
        raise ValueError(f"CT/GT geometry mismatch for {patient}")
    spacing = np.asarray(gt.header.get_zooms()[:3], dtype=float)
    if (gt.header.get_xyzt_units()[0] != "mm" or not np.all(spacing > 0)
            or not np.allclose(spacing, np.linalg.norm(gt.affine[:3, :3], axis=0))):
        raise ValueError(f"Unusable physical geometry for {patient}")
    data = np.asanyarray(gt.dataobj)
    if not np.issubdtype(data.dtype, np.integer) or data.min() < 0 or data.max() > max(CLASSES):
        raise ValueError(f"Expected supplied annotations 0–{max(CLASSES)} for {patient}")
    return gt, data


def normalized_z(index: int, count: int) -> float:
    return index / (count - 1) if count > 1 else 0.0


def extent(positive: list[int], z: int | None = None) -> dict:
    """An absent class has undefined extent; a one-slice class is at relative z=0.5."""
    first, last = (positive[0], positive[-1]) if positive else (None, None)
    present = z is not None and z in positive
    return {"first_positive_slice": first, "last_positive_slice": last,
            "distance_from_first": z - first if present else None,
            "distance_from_last": last - z if present else None,
            "organ_relative_z": ((z - first) / (last - first) if last != first else 0.5)
            if present else None}


def overlap(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Exact hard Dice; joint-empty Dice is NaN, not a successful segmentation."""
    if gt.shape != pred.shape:
        raise ValueError("Mask shapes differ")
    g, p = int(np.count_nonzero(gt)), int(np.count_nonzero(pred))
    intersection = int(np.count_nonzero(gt & pred))
    return {"gt_present": bool(g), "pred_present": bool(p), "joint_empty": not (g or p),
            "gt_area": g, "pred_area": p, "intersection": intersection,
            "dice": 2 * intersection / (g + p) if g + p else np.nan,
            "fp_pixels": p - intersection, "fn_pixels": g - intersection}


def distribution(values) -> dict:
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict.fromkeys(("mean", "min", "max", "p05", "p25", "median", "p75", "p95"), np.nan)
    qs = np.percentile(a, [5, 25, 50, 75, 95])
    return {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max()),
            **dict(zip(("p05", "p25", "median", "p75", "p95"), map(float, qs)))}


def provenance(output: Path, stage: str, args, inputs: list[Path], extra: dict) -> None:
    """Record versions, code hashes and source file size/mtime without modifying inputs."""
    rows = [{"path": str(p.resolve()), "size_bytes": p.stat().st_size,
             "mtime_ns": p.stat().st_mtime_ns} for p in sorted(set(inputs))]
    write_csv(output / "tables" / f"{stage}_inputs.csv", rows)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo_root,
                                         text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    info = {"utc": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "git_commit": commit, "versions": {p: importlib.metadata.version(p)
            for p in ("numpy", "nibabel", "matplotlib", "pillow")},
            "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in Path(__file__).parent.glob("*.py")}, **extra}
    (output / f"{stage}_run.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
