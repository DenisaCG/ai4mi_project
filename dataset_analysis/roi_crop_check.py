"""Feasibility check for an in-plane ROI crop + pad to a fixed T x T window (candidate preprocessing step 3).

Read-only on the raw data. For each patient (corrected SEGTHOR, resampled in-plane to the median TRAIN spacing
like slice_segthor.py --resample median) it finds a window centre from the IMAGE only, with one of several rules,
then uses the labels only to MEASURE how large a window that centre needs to contain every organ voxel.

Per rule it reports, from the TRAIN patients only, the window T that is needed (+ margin, rounded up), then how
much of each VAL patient's labels that T retains. It also evaluates the naive T (largest organ bounding box,
ignoring where the window is centred) to show what that shortcut would clip. z is untouched: only the in-plane
(x, y) window is analysed, so volumes are resampled in-plane only.

Run from the repo root, on a compute node:  python dataset_analysis/roi_crop_check.py [--limit 2 for a dry run]
"""
import argparse
import csv
import json
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, REPO.as_posix())  # repo-root utils.py (needed by slice_segthor), not dataset_analysis/utils.py
from slice_segthor import (ROI_MASK_STEP, body_mask, get_splits, inplane_extent, largest_components,  # noqa: E402
                           median_target_spacing, resample_image, resample_label)

CLASS_NAMES = ["esophagus", "heart", "trachea", "aorta"]  # labels 1..4 (corrected SEGTHOR)
NUM_LABELS = len(CLASS_NAMES) + 1
RULES = ("fov_center", "body_bbox", "body_centroid", "lung_bbox")
LUNG_HU = -320  # air-like voxels inside the body (lungs, trachea); the body mask itself is slice_segthor.body_mask
T_SWEEP = (224, 256, 288, 320, 352, 384)


def lung_mask(ct: np.ndarray, body: np.ndarray) -> np.ndarray:
    """Air-like voxels enclosed by the body (per-slice hole filling), largest components only (lungs, not bowel gas)."""
    filled = np.stack([ndimage.binary_fill_holes(body[:, :, z]) for z in range(body.shape[2])], axis=2)
    return largest_components(filled & (ct < LUNG_HU), min_fraction=0.2)


def window_centres(ct: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """IMAGE-ONLY in-plane window centres (voxels) per rule, and the body/lung bbox widths (voxels)."""
    small = ct[::ROI_MASK_STEP, ::ROI_MASK_STEP, ::ROI_MASK_STEP]
    body = body_mask(small)
    lungs = lung_mask(small, body)
    b_lo, b_hi = inplane_extent(body, ROI_MASK_STEP)
    l_lo, l_hi = inplane_extent(lungs, ROI_MASK_STEP)
    xs, ys, _ = np.nonzero(body)
    centres = {"fov_center": np.array(ct.shape[:2], dtype=float) / 2,
               "body_bbox": (b_lo + b_hi) / 2,
               "body_centroid": np.array([xs.mean(), ys.mean()]) * ROI_MASK_STEP + ROI_MASK_STEP / 2,
               "lung_bbox": (l_lo + l_hi) / 2}
    return centres, {"body": b_hi - b_lo, "lungs": l_hi - l_lo}


def window(centre: float, size: int, n: int) -> tuple[int, int]:
    """[start, stop) of a `size` window around `centre`, clipped to the array (the rest would be padding)."""
    start = int(round(centre - size / 2))
    return max(start, 0), min(start + size, n)


def window_stats(proj: np.ndarray, centre: np.ndarray, size: int, num_slices: int) -> dict:
    """Label voxels inside the window (proj = per-class voxel counts summed over z, shape [K, X, Y])."""
    (x0, x1), (y0, y1) = window(centre[0], size, proj.shape[1]), window(centre[1], size, proj.shape[2])
    inside = proj[:, x0:x1, y0:y1].sum(axis=(1, 2))
    total = proj.sum(axis=(1, 2))
    per_class = [float(inside[k] / total[k]) for k in range(1, NUM_LABELS)]
    return {"retained": float(inside[1:].sum() / total[1:].sum()), "worst_class": min(per_class),
            "clipped_voxels": int(total[1:].sum() - inside[1:].sum()),
            "fg_fraction": float(inside[1:].sum() / (size * size * num_slices))}


def analyse(id_: str, split: str, source_dir: Path, target: tuple[float, float]) -> dict:
    try:
        ct_nib = nib.load(str(source_dir / "train" / id_ / f"{id_}.nii.gz"))
        gt_nib = nib.load(str(source_dir / "train" / id_ / "GT.nii.gz"))
        spacing = ct_nib.header.get_zooms()[:3]
        ct, gt = np.asarray(ct_nib.dataobj), np.asarray(gt_nib.dataobj)
        assert ct.shape == gt.shape, (ct.shape, gt.shape)
        labels = set(np.unique(gt).tolist())
        assert labels == set(range(NUM_LABELS)), f"expected the corrected labels 0..4, got {sorted(labels)}"

        # in-plane resampling only (z factor 1): cubic image, nearest-neighbour label, as in slice_segthor.py
        grid = (target[0], target[1], spacing[2])
        ct = resample_image(ct, spacing, grid)
        gt = resample_label(gt, spacing, grid)
        assert ct.shape == gt.shape, (ct.shape, gt.shape)
        nx, ny, nz = ct.shape

        proj = np.stack([(gt == k).sum(axis=2) for k in range(NUM_LABELS)]).astype(np.int64)
        union = proj[1:].sum(axis=0)
        lo, hi = inplane_extent(union[:, :, None] > 0)
        centres, widths = window_centres(ct)
        heart_z = int(np.argmax((gt == 2).sum(axis=(0, 1))))

        rules = {}
        for rule, c in centres.items():
            half = np.maximum(c - lo, hi - c)  # distance from the centre to the farthest label edge, per axis
            rules[rule] = {"centre": c.tolist(), "required_T": (2 * half).tolist()}
        return {"id": id_, "split": split, "shape_native": list(ct_nib.shape), "spacing_native": [float(s) for s in spacing],
                "axcodes": "".join(nib.aff2axcodes(ct_nib.affine)), "shape_resampled": [nx, ny, nz],
                "hu_min_max": [float(ct.min()), float(ct.max())], "num_slices": nz,
                "fg_fraction_frame": float(union.sum() / (nx * ny * nz)),
                "label_bbox_px": [(hi - lo).tolist(), lo.tolist()], "body_px": widths["body"].tolist(),
                "lungs_px": widths["lungs"].tolist(), "rules": rules, "proj": proj,
                "preview": {"z": heart_z, "ct": ct[:, :, heart_z], "gt": gt[:, :, heart_z]}}
    except Exception:
        return {"id": id_, "split": split, "error": traceback.format_exc()}


def round_up(value: float, multiple: int) -> int:
    return int(np.ceil(value / multiple) * multiple)


def aggregate(results: list[dict], margin: int, multiple: int) -> dict:
    train = [r for r in results if r["split"] == "train"]
    naive_extent = max(max(r["label_bbox_px"][0]) for r in train)
    out = {"margin_px": margin, "multiple": multiple, "naive_T": round_up(naive_extent + 2 * margin, multiple),
           "naive_extent_px": naive_extent, "rules": {}}
    for rule in RULES:
        req = np.array([r["rules"][rule]["required_T"] for r in train])
        worst = train[int(req.max(axis=1).argmax())]["id"]
        needed = float(req.max())
        t_rule = round_up(needed + 2 * margin, multiple)
        sweep = {}
        for size in sorted(set(T_SWEEP) | {t_rule, out["naive_T"]}):
            rows = {}
            for split in ("train", "val"):
                st = [window_stats(r["proj"], np.array(r["rules"][rule]["centre"]), size, r["num_slices"])
                      for r in results if r["split"] == split]
                if st:
                    rows[split] = {"min_retained": min(s["retained"] for s in st),
                                   "min_worst_class": min(s["worst_class"] for s in st),
                                   "patients_clipped": sum(s["clipped_voxels"] > 0 for s in st), "n": len(st),
                                   "mean_fg_fraction": float(np.mean([s["fg_fraction"] for s in st]))}
            sweep[size] = rows
        out["rules"][rule] = {"train_required_T_max_xy": req.max(axis=0).tolist(), "train_required_T_max": needed,
                              "worst_train_patient": worst, "T": t_rule, "sweep": sweep}
    out["best_rule"] = min(RULES, key=lambda k: (out["rules"][k]["T"], out["rules"][k]["train_required_T_max"]))
    out["fg_fraction_frame_mean"] = float(np.mean([r["fg_fraction_frame"] for r in results]))
    return out


def make_figure(results: list[dict], rule: str, size: int, path: Path, n: int = 6) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    picks = sorted(results, key=lambda r: -max(r["rules"][rule]["required_T"]))[:n]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, r in zip(axes.ravel(), picks):
        ax.imshow(np.clip(r["preview"]["ct"], -1000, 400).T, cmap="gray")  # array axes (x, y) transposed, not re-oriented
        ax.imshow(np.ma.masked_equal(r["preview"]["gt"], 0).T, cmap=ListedColormap(["#c0392b", "#2980b9", "#27ae60", "#f39c12"]),
                  vmin=1, vmax=4, alpha=.6, interpolation="nearest")
        cx, cy = r["rules"][rule]["centre"]
        for s, style, color in ((size, "-", "#7CFC00"), (256, "--", "white")):
            ax.add_patch(Rectangle((round(cx - s / 2), round(cy - s / 2)), s, s, fill=False, ls=style, ec=color, lw=2))
        ax.plot([cx], [cy], "+", color="#7CFC00", ms=14)
        ax.set_title(f"{r['id']} ({r['split']}): needs T={max(r['rules'][rule]['required_T']):.0f} px", fontsize=11)
        ax.axis("off")
    fig.suptitle(f"Centre rule '{rule}': green = T={size} window, dashed white = 256; the six patients that need the largest T\n"
                 "esophagus red, heart blue, trachea green, aorta orange (heart slice, array axes)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_tables(results: list[dict], summary: dict, out_dir: Path) -> None:
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "tables" / "per_patient.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient", "split", "shape_native", "spacing_x_native", "axcodes", "shape_resampled", "fg_fraction_frame",
                    "label_bbox_x_px", "label_bbox_y_px", "body_x_px", "body_y_px", "lungs_x_px", "lungs_y_px"]
                   + [f"{rule}_required_T_{ax}" for rule in RULES for ax in "xy"])
        for r in results:
            w.writerow([r["id"], r["split"], r["shape_native"], round(r["spacing_native"][0], 4), r["axcodes"],
                        r["shape_resampled"], f"{r['fg_fraction_frame']:.5f}", *np.round(r["label_bbox_px"][0], 1),
                        *np.round(r["body_px"], 1), *np.round(r["lungs_px"], 1),
                        *[round(v, 1) for rule in RULES for v in r["rules"][rule]["required_T"]]])
    with open(out_dir / "tables" / "retention_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rule", "T", "split", "n_patients", "min_retained", "min_worst_class", "patients_clipped", "mean_fg_fraction"])
        for rule, info in summary["rules"].items():
            for size, rows in info["sweep"].items():
                for split, s in rows.items():
                    w.writerow([rule, size, split, s["n"], f"{s['min_retained']:.5f}", f"{s['min_worst_class']:.5f}",
                                s["patients_clipped"], f"{s['mean_fg_fraction']:.5f}"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source_dir", type=Path, default=REPO / "data/segthor_part1_corrected")
    p.add_argument("--out_dir", type=Path, default=REPO / "dataset_analysis/results/roi_crop_check")
    p.add_argument("--retains", type=int, default=5, help="val patients, same split as the configs (fold 0, seed 0)")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--margin", type=int, default=16, help="voxels added on each side of the largest required half-window")
    p.add_argument("--multiple", type=int, default=32)
    p.add_argument("--process", "-p", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="dry run: only the first N train patients and 1 val patient")
    args = p.parse_args()
    started = datetime.now().isoformat(timespec="seconds")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    print(f"[{started}] roi_crop_check, commit {commit}, {vars(args)}", flush=True)

    random.seed(args.seed)  # get_splits shuffles with the global RNG
    train_ids, val_ids, _ = get_splits(args.source_dir, args.retains, args.fold)
    train_zooms = [nib.load(str(args.source_dir / "train" / i / f"{i}.nii.gz")).header.get_zooms()[:3] for i in train_ids]
    target = median_target_spacing(train_zooms)  # from train patients only, as in slice_segthor.py
    print(f"corrected SEGTHOR, median train spacing {np.round(target, 4).tolist()} mm; train {train_ids}; val {val_ids}", flush=True)
    jobs = [(i, "train") for i in train_ids] + [(i, "val") for i in val_ids]
    if args.limit:
        jobs = [j for j in jobs if j[1] == "train"][:args.limit] + [j for j in jobs if j[1] == "val"][:1]

    t0 = time.time()
    task_args = [(i, s, args.source_dir, target[:2]) for i, s in jobs]
    results = [analyse(*a) for a in task_args] if args.process == 1 else Pool(args.process).starmap(analyse, task_args)
    failed = [r for r in results if "error" in r]
    for r in failed:
        print(f"FAILED {r['id']}:\n{r['error']}", flush=True)
    if failed:
        sys.exit(f"{len(failed)} patient(s) failed, no summary written (T would be wrong): {[r['id'] for r in failed]}")
    assert len({r["axcodes"] for r in results}) == 1, "mixed orientations, offsets are not comparable"

    for r in results:
        req = ", ".join(f"{rule} {max(r['rules'][rule]['required_T']):.0f}" for rule in RULES)
        print(f"{r['id']} {r['split']:5s} {r['axcodes']} native {r['shape_native']} @ {r['spacing_native'][0]:.3f} mm -> "
              f"{r['shape_resampled']}, HU {r['hu_min_max'][0]:.0f}/{r['hu_min_max'][1]:.0f}, organs {np.round(r['label_bbox_px'][0]).tolist()} px, "
              f"body {np.round(r['body_px']).tolist()}, lungs {np.round(r['lungs_px']).tolist()} | required T: {req}", flush=True)

    summary = aggregate(results, args.margin, args.multiple)
    best = summary["best_rule"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_tables(results, summary, args.out_dir)
    make_figure(results, best, summary["rules"][best]["T"], args.out_dir / "roi_crop_examples.png")

    print(f"\nnaive T (largest single-patient organ box {summary['naive_extent_px']:.0f} px + 2x{args.margin}): {summary['naive_T']}")
    print(f"foreground fraction of the full frame (= the current 256 squish), mean over patients: {summary['fg_fraction_frame_mean']:.4f}")
    for rule in RULES:
        info = summary["rules"][rule]
        print(f"\n[{rule}] train needs T={info['train_required_T_max']:.0f} px (x,y {np.round(info['train_required_T_max_xy']).tolist()}, "
              f"worst {info['worst_train_patient']}) -> T={info['T']} with margin")
        for size, rows in info["sweep"].items():
            tag = " <- rule T" if size == info["T"] else (" <- naive T" if size == summary["naive_T"] else "")
            print(f"   T={size}: " + "; ".join(f"{sp}: min retained {s['min_retained']:.4f}, worst class {s['min_worst_class']:.3f}, "
                  f"{s['patients_clipped']}/{s['n']} clipped, fg {s['mean_fg_fraction']:.3f}" for sp, s in rows.items()) + tag)
    print(f"\nbest rule: {best}, T={summary['rules'][best]['T']}")

    (args.out_dir / "run.json").write_text(json.dumps({
        "args": {k: str(v) for k, v in vars(args).items()}, "git_commit": commit, "started": started,
        "finished": datetime.now().isoformat(timespec="seconds"), "seconds": round(time.time() - t0, 1),
        "segthor_version": "corrected", "target_spacing_mm": target, "train": train_ids, "val": val_ids,
        "summary": summary, "patients": [{k: v for k, v in r.items() if k not in ("proj", "preview")} for r in results]},
        indent=2, default=str))
    print(f"[{datetime.now().isoformat(timespec='seconds')}] wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
