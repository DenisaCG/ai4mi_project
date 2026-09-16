"""Analyze existing validation hard predictions for annotated classes 1–3 only."""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from utils import (CLASSES, discover, distribution, identity, load_original,
                   load_png, overlap, parser, paths, provenance, read_csv, write_csv)

from figures import baseline_figures


def class_summary(rows):
    """GT-positive slice summaries, with joint-empty and false-positive-only counts separate."""
    result = []
    for k, name in CLASSES.items():
        rs = [r for r in rows if r["class_id"] == k]
        positive = [r for r in rs if r["gt_present"]]
        patient_means = [np.mean([r["dice"] for r in positive if r["patient_id"] == p])
                         for p in sorted({r["patient_id"] for r in positive})]
        result.append({"class_id": k, "class_name": name, "source_grid": "processed_png",
                       "total_slices": len(rs), "gt_positive_slices": len(positive),
                       "joint_empty_slices": sum(r["joint_empty"] for r in rs),
                       "fp_only_slices": sum(not r["gt_present"] and r["pred_present"] for r in rs),
                       "fp_pixels_on_gt_empty": sum(r["fp_pixels"] for r in rs if not r["gt_present"]),
                       "num_positive_patients": len(patient_means),
                       "equal_patient_mean_positive_slice_dice": float(np.mean(patient_means)) if patient_means else np.nan,
                       **{f"positive_slice_dice_{key}": value for key, value in
                          distribution(r["dice"] for r in positive).items()}})
    return result


def bin_index(value: float, edges: np.ndarray) -> int:
    """Left-closed bins; the final bin includes its upper endpoint."""
    return int(np.searchsorted(edges[1:-1], value, side="right"))


def bin_summaries(rows):
    """Report pooled slices AND equal-patient summaries; no independence assumption."""
    output, patient_output = [], []
    for k, name in CLASSES.items():
        positive = [r for r in rows if r["class_id"] == k and r["gt_present"]]
        if not positive:
            continue
        areas = np.array([r["gt_area"] for r in positive])
        area_edges = np.unique(np.quantile(areas, np.linspace(0, 1, 6)))
        if len(area_edges) == 1:
            area_edges = np.array([areas[0], areas[0] + 1])
        for kind, field, edges in (("area_quantile", "gt_area", area_edges),
                                   ("scan_z", "normalized_z", np.linspace(0, 1, 11)),
                                   ("organ_z", "organ_relative_z", np.array([0, .2, .8, 1.]))):
            for b in range(len(edges) - 1):
                rs = [r for r in positive if bin_index(r[field], edges) == b]
                if not rs:
                    continue
                ids = sorted({r["patient_id"] for r in rs})
                patient_means = []
                label = ("first_20_percent", "middle_60_percent", "last_20_percent")[b] if kind == "organ_z" else str(b)
                for patient in ids:
                    pr = [r for r in rs if r["patient_id"] == patient]
                    mean = float(np.mean([r["dice"] for r in pr]))
                    patient_means.append(mean)
                    patient_output.append({"patient_id": patient, "class_id": k, "class_name": name,
                        "source_grid": "processed_png", "bin_type": kind, "bin_id": b,
                        "bin_label": label, "lower": edges[b], "upper": edges[b + 1],
                        "num_slices": len(pr), "mean_dice": mean,
                        "median_dice": float(np.median([r["dice"] for r in pr]))})
                stats = distribution(r["dice"] for r in rs)
                patient_stats = distribution(patient_means)
                output.append({"class_id": k, "class_name": name, "source_grid": "processed_png",
                    "bin_type": kind, "bin_id": b, "bin_label": label,
                    "lower": edges[b], "upper": edges[b + 1], "num_slices": len(rs), "num_patients": len(ids),
                    "median_gt_area": float(np.median([r["gt_area"] for r in rs])),
                    **{f"slice_dice_{key}": v for key, v in stats.items()},
                    **{f"patient_mean_dice_{key}": v for key, v in patient_stats.items()}})
    return output, patient_output


def patient_metrics(rows, original, volumes, processed_predictions):
    """Compare original-grid reconstructed masks, never mean slice Dice, for 3D Dice.

    Verify reconstructed labels against the current PNG predictions so a stale
    reconstruction cannot silently be attached to a newer best checkpoint.
    """
    results, inputs = [], []
    for patient in sorted({r["patient_id"] for r in rows}):
        prediction_path = volumes / f"{patient}.nii.gz"
        scores = {}
        if prediction_path.exists():
            gt_nii, gt = load_original(original, patient)
            pred_nii = nib.load(prediction_path)
            if (pred_nii.shape != gt.shape or not np.allclose(pred_nii.affine, gt_nii.affine)
                    or pred_nii.header.get_xyzt_units()[0] != "mm"
                    or not np.allclose(pred_nii.header.get_zooms(), gt_nii.header.get_zooms())):
                raise ValueError(f"Reconstructed geometry mismatch: {patient}")
            pred = np.asanyarray(pred_nii.dataobj)
            if not np.issubdtype(pred.dtype, np.integer) or pred.min() < 0 or pred.max() > 4:
                raise ValueError(f"Unexpected reconstructed labels: {patient}")
            for z, image in sorted(processed_predictions[patient].items()):
                # This analysis intentionally supports the repository's exact 2x grid change.
                enlarged = image.repeat(2, axis=0).repeat(2, axis=1)
                if not np.array_equal(enlarged, pred[:, :, z]):
                    raise ValueError(f"Reconstruction differs from current best PNG: {patient}, {z}")
            for k in CLASSES:
                m = overlap(gt == k, pred == k)
                scores[k] = {"dice_3d": m["dice"], "gt_voxels_3d": m["gt_area"],
                             "pred_voxels_3d": m["pred_area"], "intersection_voxels_3d": m["intersection"],
                             "fp_voxels_3d": m["fp_pixels"], "fn_voxels_3d": m["fn_pixels"]}
            inputs.extend([prediction_path, original / patient / "GT.nii.gz",
                           original / patient / f"{patient}.nii.gz"])
            print(f"3D verified: {patient}", flush=True)
        for k, name in CLASSES.items():
            rs = [r for r in rows if r["patient_id"] == patient and r["class_id"] == k]
            positive = [r["dice"] for r in rs if r["gt_present"]]
            stats = distribution(positive)
            results.append({"patient_id": patient, "split": "val", "class_id": k, "class_name": name,
                            "num_gt_positive_slices": len(positive), "mean_positive_slice_dice": stats["mean"],
                            "median_positive_slice_dice": stats["median"],
                            "joint_empty_slices": sum(r["joint_empty"] for r in rs),
                            "fp_only_slices": sum(not r["gt_present"] and r["pred_present"] for r in rs),
                            "reconstruction_available": bool(scores),
                            "slice_metric_grid": "processed_png", "volume_metric_grid": "original_nifti",
                            **scores.get(k, dict.fromkeys(("dice_3d", "gt_voxels_3d", "pred_voxels_3d",
                                "intersection_voxels_3d", "fp_voxels_3d", "fn_voxels_3d"), np.nan))})
    return results, inputs


def main():
    p = parser(__doc__)
    p.add_argument("--predictions", type=Path)
    p.add_argument("--reconstructed-volumes", type=Path)
    args = p.parse_args()
    root, original, processed, output = paths(args)
    predictions = (args.predictions or root / "results/segthor/ce/best_epoch/val").resolve()
    volumes = (args.reconstructed_volumes or root / "volumes/segthor/ce").resolve()
    if not predictions.is_dir():
        raise FileNotFoundError(predictions)
    for source in (predictions, volumes):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Output cannot overlap baseline input directories")
    run_path = output / "dataset_run.json"
    run = json.loads(run_path.read_text())
    feature_path = output / "tables/slice_class_stats.csv"
    features = read_csv(feature_path)
    patients = discover(processed)
    selected_ids = set(run["patient_ids"])
    expected = {gt.name for patient, entry in patients.items()
                if entry["split"] == "val" and patient in selected_ids for _, gt in entry["slices"].values()}
    actual = {p.name for p in predictions.glob("*.png")}
    if not expected or not expected <= actual or (not run["subset"] and expected != actual):
        raise ValueError("Prediction names do not match validation GT names")
    feature_map = {(r["stem"], int(r["class_id"])): r for r in features if r["split"] == "val"}
    if len(feature_map) != sum(r["split"] == "val" for r in features) or set(feature_map) != {
            (Path(name).stem, k) for name in expected for k in CLASSES}:
        raise ValueError("Duplicate or missing validation feature rows")
    rows, inputs, decoded = [], [run_path, feature_path], {}
    for name in sorted(expected):
        gt_path, pred_path = processed / "val/gt" / name, predictions / name
        gt, pred = load_png(gt_path), load_png(pred_path, prediction=True)
        patient, z = identity(gt_path)
        decoded.setdefault(patient, {})[z] = pred
        for k, class_name in CLASSES.items():
            feature = feature_map[(gt_path.stem, k)]
            m = overlap(gt == k, pred == k)
            if int(feature["pixel_area"]) != m["gt_area"]:
                raise ValueError(f"Stale area features: {name}, {k}")
            rows.append({"patient_id": patient, "split": "val", "stem": gt_path.stem,
                         "slice_index": z, "class_id": k, "class_name": class_name,
                         "source_grid": "processed_png", "normalized_z": float(feature["normalized_z"]),
                         "organ_relative_z": float(feature["organ_relative_z"]) if feature["organ_relative_z"] else np.nan,
                         "distance_from_first": int(feature["distance_from_first"]) if feature["distance_from_first"] else None,
                         "distance_from_last": int(feature["distance_from_last"]) if feature["distance_from_last"] else None,
                         "relative_area": float(feature["relative_area"]), **m})
        inputs.extend([gt_path, pred_path])
    print(f"Baseline: paired {len(expected)} validation slices by exact stem", flush=True)
    summary = class_summary(rows)
    bins, patient_bins = bin_summaries(rows)
    patient_rows, volume_inputs = patient_metrics(rows, original, volumes, decoded)
    inputs.extend(volume_inputs)
    for name, records in (("baseline_slice_metrics", rows), ("baseline_class_summary", summary),
                          ("baseline_binned_summary", bins), ("baseline_patient_bins", patient_bins),
                          ("baseline_patient_metrics", patient_rows)):
        write_csv(output / "tables" / f"{name}.csv", records)
    baseline_figures(output, bins, patient_rows)
    checkpoint_note = predictions.parent.parent / "best_epoch.txt"
    if checkpoint_note.exists():
        inputs.append(checkpoint_note)
    provenance(output, "baseline", args, inputs, {"num_slice_class_rows": len(rows),
               "checkpoint_note": checkpoint_note.read_text() if checkpoint_note.exists() else None,
               "annotated_classes": CLASSES, "joint_empty_dice": "undefined (blank CSV cell)",
               "main_summary_population": "GT-positive processed slices only",
               "dice_3d": "full original-grid reconstructed prediction vs original GT",
               "area_bins": "within-class 5 quantile bins, duplicate edges collapsed",
               "position_bins": "left-closed; last includes upper endpoint",
               "num_available_reconstructions": sum(r["reconstruction_available"] for r in patient_rows) // 3})
    print(f"Baseline complete -> {output}", flush=True)


if __name__ == "__main__":
    main()
