"""Isolated, single-pair comparison of native and pre-crop resampled registration inputs.

Run only from the dedicated CPU Slurm job. All generated files stay under OUTPUT;
the real fold cache and production registration dataset are read-only inputs.
"""

import argparse
import json
import os
import resource
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from src.config import REPO, load_config
from src.folds import run_configs
from src.registration import (check_pair, estimate_transform, estimation_image,
                              fold_plan, geometry, parameters, save_transform, warp_pair)
from src.registration_smoketest import (CONFIG, ORGAN_NAMES, deformation_qc,
                                        image_geometry, preprocessing_qc, sha256)


OUTPUT = REPO / "metrics/augmentation_metrics/registration/smoketest/preprocessing_comparison"
SID = "Synthetic_f0_00"
STAGES = ("initial", "rigid", "final")


def checked_output() -> None:
    expected = REPO / "metrics/augmentation_metrics/registration/smoketest"
    if OUTPUT.parent.resolve() != expected.resolve() or OUTPUT.is_symlink():
        raise ValueError("diagnostic output path is outside the approved smoketest directory")
    OUTPUT.mkdir(parents=True, exist_ok=True)


def inputs():
    cfg = run_configs(load_config(CONFIG), fold=0)[0]
    train, val, pairs = fold_plan(cfg)
    if (pairs[0]["source_patient"], pairs[0]["target_patient"]) != ("Patient_26", "Patient_19"):
        raise ValueError("fold-0 first pair is not Patient_26 -> Patient_19")
    if not {"Patient_26", "Patient_19"} <= set(train) or {"Patient_26", "Patient_19"} & set(val):
        raise ValueError("the diagnostic pair is not wholly in fold-0 training")
    root = REPO / cfg["data"]["root"]
    stats = json.loads((root / "ct_norm_stats.json").read_text())
    roi = json.loads((root / "roi_crop.json").read_text())
    if set(stats["train_patients"]) != set(train) or set(roi["required_train"]) != set(train):
        raise ValueError("fold-0 cache statistics do not match the training patients")
    native = REPO / cfg["data"]["preprocess"]["source_dir"] / "train"
    resampled = root / "resampled" / "train"
    paths = {
        "native_source_ct": native / "Patient_26/Patient_26.nii.gz",
        "native_source_gt": native / "Patient_26/GT.nii.gz",
        "native_target_ct": native / "Patient_19/Patient_19.nii.gz",
        "native_target_gt": native / "Patient_19/GT.nii.gz",
        "resampled_source_ct": resampled / "Patient_26/Patient_26.nii.gz",
        "resampled_source_gt": resampled / "Patient_26/GT.nii.gz",
        "resampled_target_ct": resampled / "Patient_19/Patient_19.nii.gz",
        "resampled_target_gt": resampled / "Patient_19/GT.nii.gz",
    }
    if any(not p.is_file() for p in paths.values()):
        raise FileNotFoundError("a required native or existing resampled NIfTI is missing")
    return cfg, paths


def point(image: sitk.Image, mask: np.ndarray):
    if not mask.any():
        return None
    z, y, x = ndimage.center_of_mass(mask)
    return list(image.TransformContinuousIndexToPhysicalPoint((float(x), float(y), float(z))))


def organ_qc(warped: sitk.Image, target: sitk.Image, source: sitk.Image) -> dict:
    w, t, s = (sitk.GetArrayViewFromImage(image) for image in (warped, target, source))
    w_voxel = float(np.prod(warped.GetSpacing()))
    s_voxel = float(np.prod(source.GetSpacing()))
    result = {}
    for label, name in ORGAN_NAMES.items():
        wm, tm, sm = w == label, t == label, s == label
        nw, nt, ns = int(wm.sum()), int(tm.sum()), int(sm.sum())
        pw, pt = point(warped, wm), point(target, tm)
        result[name] = {
            "warped_volume_mm3": nw * w_voxel,
            "source_volume_mm3": ns * s_voxel,
            "target_volume_mm3": nt * w_voxel,
            "warped_vs_source_volume_change_pct":
                100 * (nw * w_voxel / (ns * s_voxel) - 1) if ns else None,
            "warped_vs_target_dice_qc_only":
                2 * int(np.count_nonzero(wm & tm)) / (nw + nt) if nw + nt else None,
            "warped_vs_target_centroid_mm_qc_only":
                float(np.linalg.norm(np.subtract(pw, pt))) if pw and pt else None,
            "warped_centroid_mm": pw,
            "target_centroid_mm": pt,
        }
    return result


def image_qc(warped: sitk.Image, target: sitk.Image) -> dict:
    w = sitk.GetArrayViewFromImage(warped)
    t = sitk.GetArrayViewFromImage(target)
    fixed_body = t > -500
    moving_body = w > -500
    overlap = int(np.count_nonzero(fixed_body & moving_body))
    n_fixed, n_moving = int(fixed_body.sum()), int(moving_body.sum())
    a = np.clip(w[fixed_body].astype(np.float32), -1000, 400)
    b = np.clip(t[fixed_body].astype(np.float32), -1000, 400)
    diff = a - b
    return {
        "body_proxy": "native target-grid CT > -500 HU; no GT or morphological cleanup",
        "fixed_body_voxels": n_fixed,
        "warped_body_voxels": n_moving,
        "body_dice": 2 * overlap / (n_fixed + n_moving),
        "fixed_body_covered_fraction": overlap / n_fixed,
        "ct_similarity_mask": "same fixed-target CT > -500 HU voxels for every stage and run",
        "ct_similarity_window_hu": [-1000, 400],
        "ct_mae_hu": float(np.mean(np.abs(diff))),
        "ct_rmse_hu": float(np.sqrt(np.mean(np.square(diff)))),
        "ct_pearson": float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else None,
    }


def save_image(image: sitk.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.nii.gz")
    try:
        sitk.WriteImage(image, str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def comparison_figure(target_ct, target_gt, stage_pairs: dict, folder: Path) -> str:
    target_labels = sitk.GetArrayViewFromImage(target_gt)
    images = {stage: (sitk.GetArrayViewFromImage(pair[0]),
                      sitk.GetArrayViewFromImage(pair[1])) for stage, pair in stage_pairs.items()}
    fixed = sitk.GetArrayViewFromImage(target_ct)
    fig, axes = plt.subplots(4, 4, figsize=(19, 18), dpi=150)
    for row, (label, name) in enumerate(ORGAN_NAMES.items()):
        z = int(np.median(np.flatnonzero((target_labels == label).any(axis=(1, 2)))))
        for col, stage in enumerate(("target",) + STAGES):
            ct, gt = (fixed, target_labels) if stage == "target" else images[stage]
            ax = axes[row, col]
            ax.imshow(ct[z], cmap="gray", vmin=-1000, vmax=400, origin="lower")
            if np.any(gt[z] == label):
                ax.contour(gt[z] == label, levels=[0.5], colors=["#ffb703"], linewidths=0.8)
            ax.set_title(f"{name}: {stage} CT + its own GT\ntarget-grid z={z}")
            ax.axis("off")
    fig.suptitle("Same physical target slices; GT contours belong to the CT in each panel")
    fig.tight_layout()
    path = folder / "initial_rigid_final_organs.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def run(condition: str, repeat: int) -> None:
    checked_output()
    run_dir = OUTPUT / f"{condition}_{repeat}"
    run_dir.mkdir(exist_ok=False)
    figure_dir, report_dir = run_dir / "figures", run_dir / "reports"
    figure_dir.mkdir()
    report_dir.mkdir()
    started = time.monotonic()
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    cfg, paths = inputs()
    fingerprints = {key: sha256(path) for key, path in paths.items()}
    images = {key: sitk.ReadImage(str(path), sitk.sitkUInt8 if key.endswith("gt") else sitk.sitkFloat32)
              for key, path in paths.items()}
    for prefix in ("native_source", "native_target", "resampled_source", "resampled_target"):
        if geometry(images[prefix + "_ct"]) != geometry(images[prefix + "_gt"]):
            raise ValueError(f"{prefix} CT/GT geometry mismatch")
    native_source, source_gt = images["native_source_ct"], images["native_source_gt"]
    native_target, target_gt = images["native_target_ct"], images["native_target_gt"]
    moving = images[f"{condition}_source_ct"]
    fixed = images[f"{condition}_target_ct"]
    params = parameters(cfg["data"]["registration"].get("params"))
    params["threads"] = 1  # only controlled execution change from production defaults
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    if sitk.ProcessObject.GetGlobalDefaultNumberOfThreads() != 1:
        raise RuntimeError("SimpleITK did not accept the one-thread setting")
    initial = sitk.CenteredTransformInitializer(
        estimation_image(fixed, params), estimation_image(moving, params),
        sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    transform, optimizer = estimate_transform(moving, fixed, params)
    composite = sitk.CompositeTransform(transform)
    if composite.GetNumberOfTransforms() != 2:
        raise ValueError("expected rigid and B-spline transform components")
    transforms = {"initial": initial, "rigid": composite.GetNthTransform(0),
                  "final": transform}
    stage_pairs = {}
    stage_reports = {}
    for stage in STAGES:
        tx = transforms[stage]
        save_transform(tx, report_dir / f"{stage}_transform.h5", native_target)
        ct, gt = warp_pair(native_source, source_gt, native_target, tx)
        if geometry(ct) != geometry(native_target) or geometry(gt) != geometry(native_target):
            raise ValueError(f"{stage}: warped CT/GT do not match native target geometry")
        if stage == "final":
            check_pair(ct, gt)
        volume_dir = run_dir / "volumes" / stage / "train" / SID
        save_image(ct, volume_dir / f"{SID}.nii.gz")
        save_image(gt, volume_dir / "GT.nii.gz")
        stage_pairs[stage] = ct, gt
        stage_reports[stage] = {"organs": organ_qc(gt, target_gt, source_gt),
                                "images": image_qc(ct, native_target),
                                "ct_geometry": image_geometry(ct),
                                "gt_geometry": image_geometry(gt)}
    figure = comparison_figure(native_target, target_gt, stage_pairs, figure_dir)
    deformation = deformation_qc(report_dir / "final_transform.h5", native_target, figure_dir)
    retention = preprocessing_qc(cfg, SID, run_dir / "volumes" / "final", report_dir)
    if {key: sha256(path) for key, path in paths.items()} != fingerprints:
        raise ValueError("a native or cached registration input changed during the diagnostic")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report = {
        "condition": condition, "repeat": repeat,
        "pair": {"fold": 0, "source": "Patient_26", "target": "Patient_19",
                 "pairing_seed": cfg["data"]["registration"]["pairing_seed"]},
        "registration_input_paths": {key: str(paths[key]) for key in
                                     (f"{condition}_source_ct", f"{condition}_target_ct")},
        "all_input_sha256_unchanged": fingerprints,
        "geometry": {key: image_geometry(image) for key, image in images.items()},
        "simpleitk_version": sitk.Version_VersionString(),
        "threads": {"simpleitk": sitk.ProcessObject.GetGlobalDefaultNumberOfThreads(),
                    "omp": os.environ.get("OMP_NUM_THREADS"),
                    "itk_global": os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS")},
        "registration_parameters": params,
        "optimizer": optimizer,
        "stages": stage_reports,
        "deformation": deformation,
        "preprocessing_qc": retention,
        "figure": figure,
        "runtime": {"wall_seconds": time.monotonic() - started,
                    "user_cpu_seconds": usage.ru_utime - usage_before.ru_utime,
                    "system_cpu_seconds": usage.ru_stime - usage_before.ru_stime,
                    "peak_rss_kib": usage.ru_maxrss},
    }
    (report_dir / "qc.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"condition": condition, "repeat": repeat,
                      "rigid_seconds": optimizer["rigid"]["runtime_seconds"],
                      "bspline_seconds": optimizer["bspline"]["runtime_seconds"],
                      "wall_seconds": report["runtime"]["wall_seconds"]}), flush=True)


def repeat_qc(one: dict, two: dict) -> dict:
    condition = one["condition"]
    target = sitk.ReadImage(one["registration_input_paths"][f"{condition}_target_ct"])
    size = target.GetSize()
    coords = np.linspace(0, 1, 9)
    points = [target.TransformContinuousIndexToPhysicalPoint(
        tuple(float(v * (n - 1)) for v, n in zip((x, y, z), size)))
        for x in coords for y in coords for z in coords]
    stages = {}
    for stage in STAGES:
        tx = [sitk.ReadTransform(str(OUTPUT / f"{condition}_{i}/reports/{stage}_transform.h5"))
              for i in (1, 2)]
        distances = [float(np.linalg.norm(np.subtract(tx[0].TransformPoint(p),
                                                       tx[1].TransformPoint(p)))) for p in points]
        cts, gts = [], []
        for i in (1, 2):
            root = OUTPUT / f"{condition}_{i}/volumes/{stage}/train/{SID}"
            cts.append(sitk.ReadImage(str(root / f"{SID}.nii.gz")))
            gts.append(sitk.ReadImage(str(root / "GT.nii.gz")))
        if geometry(cts[0]) != geometry(cts[1]) or geometry(gts[0]) != geometry(gts[1]):
            raise ValueError(f"repeat {stage} CT/GT geometries differ")
        a, b = (sitk.GetArrayViewFromImage(image) for image in cts)
        diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
        ga, gb = (sitk.GetArrayViewFromImage(image) for image in gts)
        per_organ = {}
        for label, name in ORGAN_NAMES.items():
            ma, mb = ga == label, gb == label
            pa, pb = point(gts[0], ma), point(gts[1], mb)
            denom = int(ma.sum() + mb.sum())
            per_organ[name] = {"dice": 2 * int(np.count_nonzero(ma & mb)) / denom if denom else None,
                               "centroid_shift_mm": float(np.linalg.norm(np.subtract(pa, pb)))
                               if pa and pb else None,
                               "volume_difference_mm3": float((int(mb.sum()) - int(ma.sum())) *
                                                              np.prod(gts[0].GetSpacing()))}
        stages[stage] = {"transform_point_distance_mm": {"median": float(np.median(distances)),
                                                          "p95": float(np.percentile(distances, 95)),
                                                          "max": float(np.max(distances))},
                         "warped_ct_mae_hu": float(diff.mean()),
                         "warped_ct_abs_difference_hu_percentiles":
                             {str(p): float(np.percentile(diff, p)) for p in (50, 90, 95, 99)},
                         "warped_gt": per_organ}
    return {"transform_sample_grid": "9x9x9 points across native target physical FOV",
            "stages": stages}


def summarize() -> None:
    checked_output()
    reports = {}
    for condition in ("native", "resampled"):
        for repeat in (1, 2):
            path = OUTPUT / f"{condition}_{repeat}/reports/qc.json"
            reports[f"{condition}_{repeat}"] = json.loads(path.read_text())
    repeats = {condition: repeat_qc(reports[f"{condition}_1"], reports[f"{condition}_2"])
               for condition in ("native", "resampled")}
    summary = {"runs": {key: {"optimizer": report["optimizer"],
                             "stages": report["stages"], "runtime": report["runtime"]}
                        for key, report in reports.items()},
               "repeatability": repeats,
               "interpretation": "QC measures only; human anatomical review required before bulk generation"}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (OUTPUT / "README.md").write_text(
        "# Fold-0 preprocessing comparison\n\n"
        "Four isolated, one-thread registrations of Patient_26 -> Patient_19: "
        "native_1/2 and resampled_1/2. Estimation uses native HU or the existing "
        "fold-0 pre-crop resampled HU CTs. Every initial, rigid and final transform "
        "warps the same native source CT/GT onto the same native target grid. "
        "Per-run `reports/qc.json` records all inputs and settings; `summary.json` "
        "compares the two repeats. The PNG figures show target and stage-specific "
        "warped CT with its own GT contour on the same physical target planes. "
        "Stage NIfTIs and verified transforms are under each run's `volumes/` "
        "and `reports/`. Final-stage CT/GT are passed through the unchanged "
        "production slicer in an isolated temporary directory for ROI retention QC; "
        "that temporary data is removed. The real fold's statistics and all "
        "production caches remain read-only. Target GT is used only for QC. "
        "Body overlap uses a simple CT > -500 HU proxy; CT MAE/RMSE/Pearson "
        "use the same target-body mask and [-1000,400] HU window throughout. "
        "Deformation Jacobians are sampled on a coarse 4x4x5 mm grid; positive "
        "samples do not rule out local folding.\n")
    print(json.dumps({"output": str(OUTPUT), "repeatability": repeats}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("native", "resampled"))
    parser.add_argument("--repeat", type=int, choices=(1, 2))
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        if args.condition or args.repeat:
            parser.error("--summarize takes no run arguments")
        summarize()
    elif args.condition and args.repeat:
        run(args.condition, args.repeat)
    else:
        parser.error("provide --condition and --repeat, or --summarize")


if __name__ == "__main__":
    main()
