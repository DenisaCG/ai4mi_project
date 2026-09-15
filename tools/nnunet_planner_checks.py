#!/usr/bin/env python3
"""
nnU-Net planner/fingerprint checks NOT already covered by explore_data.py,
ported as closely as possible to nnU-Net's own source so the numbers here are
literally the same computation nnU-Net would run, not a re-derivation.

Reference source (cloned to a scratch dir while writing this,
github.com/MIC-DKFZ/nnUNet, nnunetv2 branch):

  1. Dataset integrity check
     nnunetv2/experiment_planning/verify_dataset_integrity.py
     (check_cases: shape/spacing/affine match; verify_labels: unexpected values)

  2. crop_to_nonzero
     nnunetv2/preprocessing/cropping/cropping.py
     (create_nonzero_mask: per-channel !=0, OR'd, binary_fill_holes;
      crop_to_nonzero: bbox of that mask)
     consumed in nnunetv2/experiment_planning/dataset_fingerprint/fingerprint_extractor.py
     (relative_size_after_cropping = prod(shape_after) / prod(shape_before))
     and in nnunetv2/experiment_planning/experiment_planners/default_experiment_planner.py
     (median_relative_size_after_cropping < 0.75 -> mask-restricted normalization)

  3. Anisotropy / target spacing
     nnunetv2/experiment_planning/experiment_planners/default_experiment_planner.py
     (determine_fullres_target_spacing)
     nnunetv2/configuration.py (ANISO_THRESHOLD = 3)

  4. Connected-component-per-class rule
     Described in the 2018 MICCAI-BraTS-workshop nnU-Net paper, Sec. 2.5
     "Postprocessing": if a class lies within a single connected component in
     ALL training cases, that is treated as a property of the dataset and all
     but the largest component are removed at inference. (Not literally present
     in nnunetv2, which instead does a CV-Dice-driven search in
     postprocessing/remove_connected_components.py -- reproduced here from the
     paper's stated rule, using the same scipy.ndimage.label primitive.)

Usage:
    python tools/nnunet_planner_checks.py \
        --data-dir data/segthor_part1/train --out-dir figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes, label

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import PALETTE, apply_style, decorate, legend_below

# Same label-mapping correction as explore_data.py: label 1 is the aorta, not
# the esophagus, and class 4 (esophagus) has no voxels in this release.
CLASS_NAMES = {0: "background", 1: "aorta", 2: "heart", 3: "trachea", 4: "esophagus"}
EXPECTED_LABELS = [0, 1, 2, 3, 4]  # what a SegTHOR dataset.json would declare
ANISO_THRESHOLD = 3  # nnunetv2/configuration.py: ANISO_THRESHOLD = 3
MASK_NORM_RELATIVE_SIZE_THRESHOLD = 3 / 4.0  # default_experiment_planner.py:205


# ---------------------------------------------------------------------------
# 1. verify_dataset_integrity.py -- check_cases() + verify_labels()
# ---------------------------------------------------------------------------
def check_case(img: nib.Nifti1Image, seg: nib.Nifti1Image) -> dict:
    shape_image = img.shape
    shape_seg = seg.shape
    shape_match = shape_image == shape_seg

    spacing_images = np.array(img.header.get_zooms()[:3])
    spacing_seg = np.array(seg.header.get_zooms()[:3])
    spacing_match = bool(np.allclose(spacing_images, spacing_seg))

    affine_match = bool(np.allclose(img.affine, seg.affine))

    return {
        "shape_image": shape_image,
        "shape_seg": shape_seg,
        "shape_match": shape_match,
        "spacing_image": spacing_images.tolist(),
        "spacing_seg": spacing_seg.tolist(),
        "spacing_match": spacing_match,
        "affine_match": affine_match,
        "affine_diag_image": np.diag(img.affine)[:3].tolist(),
        "affine_diag_seg": np.diag(seg.affine)[:3].tolist(),
        "affine_translation_image": img.affine[:3, 3].tolist(),
        "affine_translation_seg": seg.affine[:3, 3].tolist(),
    }


def verify_labels(seg_data: np.ndarray, expected_labels: list[int]) -> dict:
    found_labels = sorted(int(x) for x in np.unique(seg_data))
    unexpected = [l for l in found_labels if l not in expected_labels]
    missing = [l for l in expected_labels if l not in found_labels]
    return {"found_labels": found_labels, "unexpected_labels": unexpected, "missing_labels": missing}


# ---------------------------------------------------------------------------
# 2. cropping.py -- create_nonzero_mask() + crop_to_nonzero()
# ---------------------------------------------------------------------------
def create_nonzero_mask(data: np.ndarray) -> np.ndarray:
    """data must be (C, X, Y, Z). Verbatim port of cropping.py's version."""
    assert data.ndim == 4
    nonzero_mask = data[0] != 0
    for c in range(1, data.shape[0]):
        nonzero_mask |= data[c] != 0
    return binary_fill_holes(nonzero_mask)


def crop_to_nonzero(data: np.ndarray) -> dict:
    """data must be (C, X, Y, Z). Returns bbox + before/after shapes, mirroring
    what fingerprint_extractor.py:analyze_case records per case."""
    shape_before_crop = data.shape[1:]
    mask = create_nonzero_mask(data)
    if not mask.any():
        return {"shape_before_crop": shape_before_crop, "shape_after_crop": (0, 0, 0),
                "relative_size_after_cropping": 0.0}
    idx = np.nonzero(mask)
    mins = tuple(int(a.min()) for a in idx)
    maxs = tuple(int(a.max()) for a in idx)
    shape_after_crop = tuple(hi - lo + 1 for lo, hi in zip(mins, maxs))
    relative_size = float(np.prod(shape_after_crop) / np.prod(shape_before_crop))
    return {"shape_before_crop": shape_before_crop, "shape_after_crop": shape_after_crop,
            "relative_size_after_cropping": relative_size}


# ---------------------------------------------------------------------------
# 3. default_experiment_planner.py -- determine_fullres_target_spacing()
# ---------------------------------------------------------------------------
def determine_fullres_target_spacing(spacings: np.ndarray, sizes: np.ndarray,
                                     anisotropy_threshold: int = ANISO_THRESHOLD) -> dict:
    """spacings, sizes: (N, 3) arrays, one row per case. Verbatim port of
    default_experiment_planner.py:determine_fullres_target_spacing (median
    target, 10th-percentile override on the anisotropic axis)."""
    target = np.percentile(spacings, 50, axis=0)
    target_size = np.percentile(sizes, 50, axis=0)

    worst_spacing_axis = int(np.argmax(target))
    other_axes = [i for i in range(len(target)) if i != worst_spacing_axis]
    other_spacings = [target[i] for i in other_axes]
    other_sizes = [target_size[i] for i in other_axes]

    has_aniso_spacing = bool(target[worst_spacing_axis] > (anisotropy_threshold * max(other_spacings)))
    has_aniso_voxels = bool(target_size[worst_spacing_axis] * anisotropy_threshold < min(other_sizes))

    overridden = False
    final_target = target.copy()
    if has_aniso_spacing and has_aniso_voxels:
        spacings_of_that_axis = spacings[:, worst_spacing_axis]
        target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 10)
        if target_spacing_of_that_axis < max(other_spacings):
            target_spacing_of_that_axis = max(max(other_spacings), target_spacing_of_that_axis) + 1e-5
        final_target[worst_spacing_axis] = target_spacing_of_that_axis
        overridden = True

    return {
        "median_spacing": target.tolist(),
        "median_shape": target_size.tolist(),
        "worst_spacing_axis": worst_spacing_axis,
        "has_aniso_spacing": has_aniso_spacing,
        "has_aniso_voxels": has_aniso_voxels,
        "cascade_override_applied": overridden,
        "final_target_spacing": final_target.tolist(),
    }


# ---------------------------------------------------------------------------
# 4. Connected-component-per-class rule (paper Sec. 2.5)
# ---------------------------------------------------------------------------
STRUCT_26 = np.ones((3, 3, 3), dtype=int)


def connected_components_per_class(seg: np.ndarray, classes: list[int]) -> dict:
    """Reports both connectivity definitions. scipy's default (structure=None)
    is face-only (6-connectivity in 3D); nnU-Net's real remove_all_but_largest_component
    (acvl_utils, skimage-based) uses full connectivity by default -- see the
    aorta case study this surfaced (P02/P09/P16: 6-conn=2, 26-conn=1)."""
    out = {}
    for c in classes:
        mask = seg == c
        if not mask.any():
            out[c] = {"n6": 0, "n26": 0}
            continue
        _, n6 = label(mask)
        _, n26 = label(mask, structure=STRUCT_26)
        out[c] = {"n6": int(n6), "n26": int(n26)}
    return out


def find_patients(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("Patient_"))


CACHE_JSON = "_checks_cache.json"
CACHE_NPZ = "_checks_cache.npz"


def compute(data_dir: Path) -> dict:
    """Runs the full per-patient pass (the ~2min part: loading 20 CT+GT
    volumes, cropping, labeling connected components, pooling foreground
    intensities). Returns everything every figure function needs, plain
    enough to be cached to disk so a style-only change never re-triggers
    this."""
    patients = find_patients(data_dir)
    print(f"Found {len(patients)} patients in {data_dir}")

    integrity_rows = []
    crop_rows = []
    spacings, sizes = [], []
    cc_rows = []
    affine_diffs = []
    fg_intensities_per_class = {c: [] for c in [1, 2, 3, 4]}

    rs = np.random.RandomState(1234)  # matches fingerprint_extractor.py's seed
    for p in patients:
        pid = p.name
        img = nib.load(str(p / f"{pid}.nii.gz"))
        seg = nib.load(str(p / "GT.nii.gz"))

        # 1. integrity
        check = check_case(img, seg)
        seg_data = np.asarray(seg.dataobj).astype(np.int16)
        labels_info = verify_labels(seg_data, EXPECTED_LABELS)
        integrity_rows.append({"pid": pid, **check, **labels_info})
        affine_diffs.append(float(np.max(np.abs(img.affine - seg.affine))))

        # 2. crop_to_nonzero (image only, 1 channel -> add channel axis)
        ct_data = np.asarray(img.dataobj)[None].astype(np.float32)
        crop = crop_to_nonzero(ct_data)
        crop_rows.append({"pid": pid, **crop})

        # 3. collect for anisotropy (native spacing/shape, pre-crop, matches
        # what fingerprint_extractor.py records per case)
        spacings.append(list(img.header.get_zooms()[:3]))
        sizes.append(list(img.shape))

        # 4. connected components per organ class (both connectivity defs)
        cc = connected_components_per_class(seg_data, [1, 2, 3, 4])
        cc_rows.append({"pid": pid, **cc})

        # 5. foreground intensities per class, for the CT normalization figure.
        # nnU-Net subsamples (fingerprint_extractor.py) because Decathlon-scale
        # datasets can have huge foreground counts; with 20 patients we can
        # just keep every foreground voxel per class -- superset of what
        # nnU-Net's subsample would see, no sampling noise.
        ct_data_flat = ct_data[0]
        for c in [1, 2, 3, 4]:
            vals = ct_data_flat[seg_data == c]
            if len(vals):
                fg_intensities_per_class[c].append(vals.astype(np.float32))

        print(f"  {pid}: shape_match={check['shape_match']} spacing_match={check['spacing_match']} "
              f"affine_match={check['affine_match']} unexpected_labels={labels_info['unexpected_labels']} "
              f"rel_size_after_crop={crop['relative_size_after_cropping']:.4f} cc={cc}")

    spacings = np.array(spacings)
    sizes = np.array(sizes)
    aniso = determine_fullres_target_spacing(spacings, sizes)

    median_rel_size = float(np.median([r["relative_size_after_cropping"] for r in crop_rows]))
    triggers_mask_norm = median_rel_size < MASK_NORM_RELATIVE_SIZE_THRESHOLD

    all_shape_ok = all(r["shape_match"] for r in integrity_rows)
    all_spacing_ok = all(r["spacing_match"] for r in integrity_rows)
    all_affine_ok = all(r["affine_match"] for r in integrity_rows)
    any_unexpected = any(r["unexpected_labels"] for r in integrity_rows)
    always_missing = sorted(set.intersection(*[set(r["missing_labels"]) for r in integrity_rows]))

    cc_summary = {}
    for c in [1, 2, 3, 4]:
        counts6 = [r[c]["n6"] for r in cc_rows if r[c]["n6"] > 0]
        counts26 = [r[c]["n26"] for r in cc_rows if r[c]["n26"] > 0]
        cc_summary[c] = {
            "n_patients_present": len(counts6),
            "always_single_component_6conn": all(n == 1 for n in counts6) if counts6 else None,
            "always_single_component_26conn": all(n == 1 for n in counts26) if counts26 else None,
            "max_components_seen_6conn": max(counts6) if counts6 else 0,
            "max_components_seen_26conn": max(counts26) if counts26 else 0,
        }

    # pooled foreground intensities, all classes combined (this is what
    # nnU-Net's CTNormalization actually clips/z-scores on: ALL foreground,
    # not per-organ)
    all_fg = np.concatenate([v for vals in fg_intensities_per_class.values() for v in vals])
    p00_5, p99_5 = np.percentile(all_fg, [0.5, 99.5])
    fg_mean, fg_std = float(all_fg.mean()), float(all_fg.std())

    results = {
        "integrity": integrity_rows,
        "integrity_summary": {
            "all_shape_match": all_shape_ok,
            "all_spacing_match": all_spacing_ok,
            "all_affine_match": all_affine_ok,
            "any_unexpected_labels": any_unexpected,
            "labels_missing_in_every_patient": always_missing,
            "max_affine_abs_diff_per_patient": affine_diffs,
        },
        "crop_to_nonzero": crop_rows,
        "median_relative_size_after_cropping": median_rel_size,
        "mask_restricted_normalization_would_trigger": triggers_mask_norm,
        "anisotropy": aniso,
        "connected_components": cc_rows,
        "connected_components_summary": cc_summary,
        "ct_normalization": {
            "clip_lower_p0_5": float(p00_5), "clip_upper_p99_5": float(p99_5),
            "mean": fg_mean, "std": fg_std, "n_foreground_voxels_pooled": int(len(all_fg)),
        },
    }

    return {
        "results": results,
        "integrity_rows": integrity_rows,
        "crop_rows": crop_rows,
        "cc_rows": cc_rows,
        "cc_summary": cc_summary,
        "affine_diffs": affine_diffs,
        "spacings": spacings,
        "sizes": sizes,
        "aniso": aniso,
        "median_rel_size": median_rel_size,
        "all_fg": all_fg,
        "p00_5": p00_5, "p99_5": p99_5, "fg_mean": fg_mean, "fg_std": fg_std,
    }


def save_cache(data: dict, out_dir: Path) -> None:
    np.savez_compressed(out_dir / CACHE_NPZ, spacings=data["spacings"], sizes=data["sizes"],
                        all_fg=data["all_fg"])
    json_part = {k: v for k, v in data.items() if k not in ("spacings", "sizes", "all_fg")}
    with open(out_dir / CACHE_JSON, "w") as f:
        json.dump(json_part, f, default=str)
    print(f"Cached per-patient computation to {out_dir / CACHE_JSON} and {out_dir / CACHE_NPZ}")


def load_cache(out_dir: Path) -> dict | None:
    json_path, npz_path = out_dir / CACHE_JSON, out_dir / CACHE_NPZ
    if not json_path.is_file() or not npz_path.is_file():
        return None
    with open(json_path) as f:
        data = json.load(f)
    npz = np.load(npz_path)
    data["spacings"], data["sizes"], data["all_fg"] = npz["spacings"], npz["sizes"], npz["all_fg"]
    # JSON has no int keys -- cc_rows/cc_summary use class ids (1-4) as dict
    # keys, which round-trip through json.dump/load as strings. Undo that.
    for row in data["cc_rows"]:
        for k in list(row):
            if k != "pid":
                row[int(k)] = row.pop(k)
    data["cc_summary"] = {int(k): v for k, v in data["cc_summary"].items()}
    data["results"]["connected_components_summary"] = {
        int(k): v for k, v in data["results"]["connected_components_summary"].items()
    }
    print(f"Loaded cached per-patient computation from {json_path} (skip with --recompute)")
    return data


# key -> (figure fn, args it needs from the `data` dict returned by compute())
FIGURES = {
    "integrity": lambda d, o: _fig_integrity(d["integrity_rows"], o),
    "crop": lambda d, o: _fig_crop(d["crop_rows"], d["median_rel_size"], o),
    "anisotropy": lambda d, o: _fig_anisotropy(d["aniso"], d["spacings"], o),
    "connected_components": lambda d, o: _fig_connected_components(d["cc_rows"], d["cc_summary"], o),
    "shape_match": lambda d, o: _fig_shape_match(d["integrity_rows"], o),
    "spacing_match": lambda d, o: _fig_spacing_match(d["integrity_rows"], o),
    "affine": lambda d, o: _fig_affine(d["integrity_rows"], d["affine_diffs"], o),
    "ct_normalization": lambda d, o: _fig_ct_normalization(d["all_fg"], d["p00_5"], d["p99_5"], d["fg_mean"],
                                                            d["fg_std"], o),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/segthor_part1/train"))
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    ap.add_argument("--only", type=str, default=None,
                    help=f"Comma-separated figure keys to (re)generate, skipping the rest. "
                         f"Choices: {','.join(FIGURES)}. Default: all.")
    ap.add_argument("--recompute", action="store_true",
                    help="Ignore any cached per-patient data and reload/recompute from --data-dir "
                         "(slow, ~2min for 20 patients). Default: reuse the cache if present.")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    data = None if args.recompute else load_cache(args.out_dir)
    if data is None:
        data = compute(args.data_dir)
        save_cache(data, args.out_dir)

    with open(args.out_dir / "nnunet_checks.json", "w") as f:
        json.dump(data["results"], f, indent=2, default=str)

    keys = [k.strip() for k in args.only.split(",")] if args.only else list(FIGURES)
    for k in keys:
        if k not in FIGURES:
            print(f"  skip unknown figure key {k!r} (choices: {', '.join(FIGURES)})")
            continue
        FIGURES[k](data, args.out_dir)
        print(f"  wrote figure: {k}")

    _write_markdown(data["results"], args.out_dir)
    print(f"\nWrote nnunet_checks.json, nnunet_checks.md and {len(keys)} figure(s) to {args.out_dir}")


def _fig_integrity(rows, out_dir):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.axis("off")
    col_labels = ["patient", "shape\nmatch", "spacing\nmatch", "affine\nmatch", "unexpected\nlabels",
                  "missing\nlabels"]
    table_data = [[r["pid"], "OK" if r["shape_match"] else "FAIL", "OK" if r["spacing_match"] else "FAIL",
                   "OK" if r["affine_match"] else "FAIL",
                   ",".join(map(str, r["unexpected_labels"])) or "-",
                   ",".join(map(str, r["missing_labels"])) or "-"] for r in rows]
    tbl = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center",
                   bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor(PALETTE[0])
            cell.set_text_props(color="white", fontweight="bold", linespacing=1.6)
            cell.set_height(cell.get_height() * 2.0)
        elif col == 5 and table_data[row - 1][5] != "-":
            cell.set_facecolor("#F7DCD4")
    decorate(fig, "Every Patient Passes Structural Checks -- But One Label Is Missing",
             subtitle="Each row is one patient. Image and label files line up perfectly on shape, spacing, and "
                       "geometry (all OK) -- but the esophagus (class 4) has zero voxels in every single one.",
             footnote_text="nnU-Net source: verify_dataset_integrity.py's check_cases() + verify_labels(), "
                            "checked against the 5 labels {0..4} this dataset should have. Note: this check only "
                            "flags values outside the expected set, so it cannot catch a missing class on its own "
                            "-- it needed the 5-class expectation added here to surface it.")
    fig.savefig(out_dir / "integrity_table.png")
    plt.close(fig)


def _fig_crop(rows, median_rel_size, out_dir):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    pids = [r["pid"].replace("Patient_", "P") for r in rows]
    vals = [r["relative_size_after_cropping"] for r in rows]
    ax.bar(pids, vals, color=PALETTE[2])
    ax.axhline(MASK_NORM_RELATIVE_SIZE_THRESHOLD, color=PALETTE[5], linestyle="--", linewidth=1.5,
               label=f"nnU-Net mask-norm threshold (0.75)")
    ax.axhline(median_rel_size, color=PALETTE[0], linestyle=":", linewidth=1.5,
               label=f"dataset median ({median_rel_size:.3f})")
    ax.set_ylabel("relative size after crop_to_nonzero")
    ax.set_ylim(0, 1.05)
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=8)
    legend_below(ax, ncol=2)
    decorate(fig, "Cropping to Non-Empty Space Barely Shrinks These Scans",
             subtitle="Each bar is one patient: how much of the 3D volume is left after trimming away completely "
                       "empty (zero-value) space. Since it's ~100% for everyone, nnU-Net's rule for restricting "
                       "normalization stats to just the cropped region never kicks in for this dataset.",
             footnote_text=f"nnU-Net source: cropping.py's crop_to_nonzero(); the rule (default_experiment_planner.py) "
                            f"triggers below a 0.75 median -- this dataset's {median_rel_size:.3f} is far above it.",
             has_legend=True)
    fig.savefig(out_dir / "crop_to_nonzero.png")
    plt.close(fig)


def _fig_anisotropy(aniso, spacings, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5))
    axis_names = ["x", "y", "z"]

    ax = axes[0]
    ax.boxplot([spacings[:, i] for i in range(3)], tick_labels=axis_names)
    worst = aniso["worst_spacing_axis"]
    ax.scatter([worst + 1], [aniso["median_spacing"][worst]], color=PALETTE[5], zorder=5, s=60,
               label="worst_spacing_axis")
    ax.set_ylabel("spacing (mm)")
    ax.set_title("Per-axis spacing across 20 patients", fontsize=11)
    ax.legend(fontsize=9, frameon=False)

    ax2 = axes[1]
    ax2.axis("off")
    ms = [round(v, 3) for v in aniso["median_spacing"]]
    ft = [round(v, 3) for v in aniso["final_target_spacing"]]
    lines = [
        ("Median spacing (x, y, z)", f"{ms} mm"),
        ("Coarsest axis", f"{axis_names[worst]}"),
        ("Spacing on that axis >3x the others?", "Yes" if aniso["has_aniso_spacing"] else "No"),
        ("Also has >3x fewer voxels?", "Yes" if aniso["has_aniso_voxels"] else "No"),
        ("-> Resampling override applied?", "Yes" if aniso["cascade_override_applied"] else "No"),
        ("Target spacing to resample to", f"{ft} mm"),
    ]
    y = 0.95
    for label, value in lines:
        ax2.text(0.0, y, f"{label}:", fontsize=10.5, va="top", color="#444444")
        ax2.text(0.0, y - 0.075, value, fontsize=12, va="top", fontweight="bold")
        y -= 0.17

    verdict = "Coarser Enough to Change nnU-Net's Resampling Plan" if aniso["cascade_override_applied"] else \
        "Coarser, But Not Extreme Enough to Change nnU-Net's Resampling Plan"
    decorate(fig, f"Z-Axis Spacing Is {verdict}",
             subtitle="Left: how much each of the 3 scan directions varies in physical spacing (mm) across your "
                       "20 patients -- z is clearly coarser. Right: the exact numbers nnU-Net's rule checks to "
                       "decide whether that coarseness is extreme enough to need special handling (it decides no).",
             footnote_text="nnU-Net source: default_experiment_planner.py's determine_fullres_target_spacing(), "
                            "threshold=3x (configuration.py). Both conditions shown at right must be True to "
                            "trigger the override.")
    fig.savefig(out_dir / "anisotropy.png")
    plt.close(fig)


def _fig_connected_components(rows, summary, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    classes = [1, 2, 3, 4]
    pids = [r["pid"].replace("Patient_", "P") for r in rows]
    x = np.arange(len(pids))
    width = 0.2

    for ax, key, conn_label in [(axes[0], "n6", "6-connectivity (scipy default)"),
                                 (axes[1], "n26", "26-connectivity (nnU-Net's likely intent)")]:
        for i, c in enumerate(classes):
            vals = [r[c][key] for r in rows]
            ax.bar(x + (i - 1.5) * width, vals, width, label=CLASS_NAMES[c], color=PALETTE[i])
        ax.set_xticks(x)
        ax.set_xticklabels(pids, rotation=90, fontsize=8)
        ax.set_ylabel("# connected components")
        ax.axhline(1, color="#999999", linewidth=0.8, zorder=0)
        ax.set_title(conn_label, fontsize=11)
    legend_below(axes[1], ncol=4)

    always_single_26 = [CLASS_NAMES[c] for c, s in summary.items() if s["always_single_component_26conn"]]
    decorate(fig, "The Aorta's Apparent Split Is Just How 'Touching' Gets Defined",
             subtitle="Each bar counts separate, disconnected blobs of one organ in one patient. Under a strict "
                       "definition of 'touching' (left), the aorta looks split into 2 pieces in 3 patients. Under "
                       "a looser, more anatomically sensible definition (right), every organ is one solid piece "
                       "in every patient.",
             footnote_text=f"nnU-Net source: scipy.ndimage.label, 6- vs 26-connectivity (nnU-Net's own "
                            f"postprocessing library likely uses the looser definition). Always single-piece "
                            f"under the looser definition: {', '.join(always_single_26) or 'none'}. See "
                            f"aorta_component_case_study_Patient_02.png for why the aorta looks split at all.",
             has_legend=True)
    fig.savefig(out_dir / "connected_components.png")
    plt.close(fig)


def _fig_shape_match(integrity_rows, out_dir):
    """R2: image shape vs. seg shape, per patient per axis. A match means the
    point sits exactly on the y=x diagonal; this is the actual per-case
    computation check_cases() runs (shape_image == shape_seg), shown as a
    distribution instead of a single pass/fail boolean."""
    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    axis_names, colors = ["x", "y", "z"], PALETTE[:3]
    for i, (name, color) in enumerate(zip(axis_names, colors)):
        img_vals = [r["shape_image"][i] for r in integrity_rows]
        seg_vals = [r["shape_seg"][i] for r in integrity_rows]
        ax.scatter(img_vals, seg_vals, s=50, color=color, alpha=0.75, label=f"axis {name}", edgecolor="white")
    lims = ax.get_xlim()
    ax.plot(lims, lims, color="#999999", linestyle="--", linewidth=1, zorder=0, label="y = x (match)")
    ax.set_xlabel("image shape (voxels)")
    ax.set_ylabel("segmentation shape (voxels)")
    legend_below(ax, ncol=2)
    decorate(fig, "The Scan and Its Label Map Are Always Exactly the Same Size",
             subtitle="Each dot compares one patient's scan size to its label map's size, on one axis (x, y, or "
                       "z). Every dot sits exactly on the diagonal line -- meaning they always match, with no "
                       "resizing errors anywhere in this dataset.",
             footnote_text="nnU-Net source: verify_dataset_integrity.py's check_cases(), shape_image == shape_seg. "
                            "60 points shown (20 patients x 3 axes).",
             has_legend=True)
    fig.savefig(out_dir / "shape_match.png")
    plt.close(fig)


def _fig_spacing_match(integrity_rows, out_dir):
    """R3: image spacing vs. seg spacing, per patient per axis -- same
    treatment as R2, for check_cases()'s np.allclose(spacing_images, spacing_seg)."""
    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    axis_names, colors = ["x", "y", "z"], PALETTE[:3]
    for i, (name, color) in enumerate(zip(axis_names, colors)):
        img_vals = [r["spacing_image"][i] for r in integrity_rows]
        seg_vals = [r["spacing_seg"][i] for r in integrity_rows]
        ax.scatter(img_vals, seg_vals, s=50, color=color, alpha=0.75, label=f"axis {name}", edgecolor="white")
    lims = ax.get_xlim()
    ax.plot(lims, lims, color="#999999", linestyle="--", linewidth=1, zorder=0, label="y = x (match)")
    ax.set_xlabel("image spacing (mm)")
    ax.set_ylabel("segmentation spacing (mm)")
    legend_below(ax, ncol=2)
    decorate(fig, "The Scan and Its Label Map Are Always at the Same Physical Scale",
             subtitle="Same idea as the shape check: each dot compares one patient's scan spacing (mm) to its "
                       "label map's spacing, on one axis. Every dot lands exactly on the diagonal -- physical "
                       "scale is never mismatched between the two files.",
             footnote_text="nnU-Net source: verify_dataset_integrity.py's check_cases(), "
                            "np.allclose(spacing_images, spacing_seg). Points cluster at only 4 distinct values "
                            "because that's how many acquisition protocols this dataset actually used.",
             has_legend=True)
    fig.savefig(out_dir / "spacing_match.png")
    plt.close(fig)


def _fig_affine(integrity_rows, affine_diffs, out_dir):
    """R4: not just the (zero) difference -- the actual affine values, so the
    per-patient geometry itself is visible. Two panels: the diagonal (voxel
    spacing with orientation sign) and the translation/origin (where the
    voxel grid sits in patient space), image vs. seg overlaid per patient."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5))
    pids = [r["pid"].replace("Patient_", "P") for r in integrity_rows]
    x = np.arange(len(pids))
    axis_names, colors = ["x", "y", "z"], PALETTE[:3]

    ax = axes[0]
    for i, (name, color) in enumerate(zip(axis_names, colors)):
        img_vals = [r["affine_diag_image"][i] for r in integrity_rows]
        seg_vals = [r["affine_diag_seg"][i] for r in integrity_rows]
        ax.scatter(x, img_vals, s=60, color=color, marker="o", label=f"{name} (image)", zorder=3)
        ax.scatter(x, seg_vals, s=110, facecolors="none", edgecolors=color, marker="o", linewidth=1.3,
                   label=f"{name} (seg, ring)", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(pids, rotation=90, fontsize=8)
    ax.set_ylabel("affine diagonal (mm, signed voxel scale)")
    ax.set_title("Voxel scale + orientation", fontsize=11)

    ax2 = axes[1]
    for i, (name, color) in enumerate(zip(axis_names, colors)):
        img_vals = [r["affine_translation_image"][i] for r in integrity_rows]
        seg_vals = [r["affine_translation_seg"][i] for r in integrity_rows]
        ax2.scatter(x, img_vals, s=60, color=color, marker="o", label=f"{name} (image)", zorder=3)
        ax2.scatter(x, seg_vals, s=110, facecolors="none", edgecolors=color, marker="o", linewidth=1.3,
                   label=f"{name} (seg, ring)", zorder=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(pids, rotation=90, fontsize=8)
    ax2.set_ylabel("affine translation (mm, scanner-space origin)")
    ax2.set_title("Volume origin", fontsize=11)
    legend_below(ax2, ncol=3)

    decorate(fig, "Scan and Label Sit in Exactly the Same 3D Space, Every Patient",
             subtitle="Left: how the scan is scaled and oriented in physical space. Right: where it sits in the "
                       "scanner's coordinate system (this varies by patient -- everyone was positioned slightly "
                       "differently on the table). In both panels, the label's values (rings) land exactly on "
                       "top of the scan's values (dots): the two files were never misaligned.",
             footnote_text="nnU-Net source: verify_dataset_integrity.py's check_cases(), "
                            "np.allclose(img.affine, seg.affine). Voxel scale (left) clusters at the same "
                            "handful of acquisition-protocol values as shape_spacing_boxplots.png.",
             has_legend=True)
    fig.savefig(out_dir / "affine_check.png")
    plt.close(fig)


def _fig_ct_normalization(all_fg, p00_5, p99_5, mean, std, out_dir):
    fig, ax = plt.subplots(figsize=(9, 7.5))
    clipped = np.clip(all_fg, -1200, 600)
    ax.hist(clipped, bins=200, color=PALETTE[2], edgecolor="none")
    ax.axvline(p00_5, color=PALETTE[5], linestyle="--", label=f"clip p0.5 = {p00_5:.0f} HU")
    ax.axvline(p99_5, color=PALETTE[5], linestyle="--", label=f"clip p99.5 = {p99_5:.0f} HU")
    ax.axvline(mean, color=PALETTE[0], linestyle="-", label=f"mean = {mean:.1f} HU")
    ax.axvspan(mean - std, mean + std, color=PALETTE[0], alpha=0.12, label=f"±1 std = {std:.1f} HU")
    ax.set_xlabel("Hounsfield Units")
    ax.set_ylabel("voxel count (all foreground classes pooled)")
    legend_below(ax, ncol=2)
    decorate(fig, "Most Labeled Tissue Sits in a Narrow Brightness Band Near 0-50 HU",
             subtitle="This histogram pools the CT brightness (Hounsfield Units) of every labeled-organ voxel "
                       "across all 20 patients. The dashed lines are the cutoffs nnU-Net would clip extreme "
                       "values to before training; the solid line and shaded band are the average and typical "
                       "spread it would then normalize everything around.",
             footnote_text="nnU-Net source: preprocessing/normalization's CTNormalization.run() -- clip to "
                            "[p0.5, p99.5], then z-score. Every foreground voxel is used here (not nnU-Net's "
                            "per-case subsample) since this dataset is small enough that no subsampling is needed.",
             has_legend=True)
    fig.savefig(out_dir / "ct_normalization.png")
    plt.close(fig)


def _write_markdown(results, out_dir):
    lines = ["# nnU-Net planner/fingerprint checks -- ported from source\n"]

    lines.append("## 1. Dataset integrity (verify_dataset_integrity.py)\n")
    s = results["integrity_summary"]
    lines.append(f"- All image/label shapes match: **{s['all_shape_match']}**")
    lines.append(f"- All image/label spacings match: **{s['all_spacing_match']}**")
    lines.append(f"- All image/label affines match: **{s['all_affine_match']}**")
    lines.append(f"- Any unexpected label values found (outside {{0..4}}): **{s['any_unexpected_labels']}**")
    lines.append(f"- Label(s) missing in EVERY patient: **{s['labels_missing_in_every_patient']}** "
                  f"(= esophagus; note this is only detectable because we know 5 classes were expected -- "
                  f"nnU-Net's own verify_labels() does not check for missing expected labels, only unexpected ones)\n")

    lines.append("## 2. crop_to_nonzero (cropping.py)\n")
    lines.append(f"- Dataset median relative size after crop: **{results['median_relative_size_after_cropping']:.4f}**")
    lines.append(f"- Triggers mask-restricted normalization (< 0.75)? **{results['mask_restricted_normalization_would_trigger']}**\n")

    lines.append("## 3. Anisotropy / target spacing (default_experiment_planner.py)\n")
    a = results["anisotropy"]
    lines.append(f"- median spacing (x,y,z): {[round(v, 4) for v in a['median_spacing']]}")
    lines.append(f"- worst_spacing_axis: {a['worst_spacing_axis']}")
    lines.append(f"- has_aniso_spacing: **{a['has_aniso_spacing']}**")
    lines.append(f"- has_aniso_voxels: **{a['has_aniso_voxels']}**")
    lines.append(f"- cascade override applied: **{a['cascade_override_applied']}**")
    lines.append(f"- final target spacing: {[round(v, 4) for v in a['final_target_spacing']]}\n")

    lines.append("## 4. Connected components per class (paper Sec. 2.5 rule)\n")
    lines.append("| class | n patients present | always single (6-conn) | always single (26-conn) | max seen (6-conn) | max seen (26-conn) |")
    lines.append("|---|---|---|---|---|---|")
    for c, cs in results["connected_components_summary"].items():
        lines.append(f"| {CLASS_NAMES[c]} | {cs['n_patients_present']} | {cs['always_single_component_6conn']} | "
                      f"{cs['always_single_component_26conn']} | {cs['max_components_seen_6conn']} | {cs['max_components_seen_26conn']} |")
    lines.append("\nAorta's 6-conn≠26-conn mismatch is a connectivity-definition artifact (arch curvature), "
                  "not a real split -- see `aorta_component_case_study_Patient_02.png`.\n")

    lines.append("## 5. Affine consistency, per patient\n")
    lines.append(f"- Max |image_affine - seg_affine| across all 20 patients: "
                  f"**{max(results['integrity_summary']['max_affine_abs_diff_per_patient']):.6f}**\n")

    lines.append("## 6. CT normalization parameters (CTNormalization.run(), pooled foreground)\n")
    n = results["ct_normalization"]
    lines.append(f"- clip lower (p0.5): **{n['clip_lower_p0_5']:.1f} HU**")
    lines.append(f"- clip upper (p99.5): **{n['clip_upper_p99_5']:.1f} HU**")
    lines.append(f"- mean: **{n['mean']:.1f} HU**, std: **{n['std']:.1f} HU**")
    lines.append(f"- n foreground voxels pooled: {n['n_foreground_voxels_pooled']:,}\n")

    (out_dir / "nnunet_checks.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
