"""Isolated fold-0 CT-body-centroid initialization diagnostic; run on a CPU node."""

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

from src.config import REPO
from src.registration import (check_pair, estimation_image, geometry, metric_mask,
                              parameters, registration_method, save_transform, warp_pair)
from src.registration_preprocessing_comparison import (SID, image_qc, inputs, organ_qc,
                                                       save_image)
from src.registration_smoketest import (ORGAN_NAMES, deformation_qc, image_geometry,
                                        preprocessing_qc, sha256)


OUTPUT = REPO / "metrics/augmentation_metrics/registration/smoketest/anatomical_initialization"
BASE = REPO / "metrics/augmentation_metrics/registration/smoketest/preprocessing_comparison/native_1"


def output_paths(stage):
    if OUTPUT.parent.resolve() != (REPO / "metrics/augmentation_metrics/registration/smoketest").resolve():
        raise ValueError("output parent changed")
    if OUTPUT.is_symlink() or not OUTPUT.is_dir():
        raise ValueError("isolated diagnostic output directory must exist and not be a symlink")
    for name in ("reports", "figures", "volumes"):
        (OUTPUT / name).mkdir(exist_ok=True)
    if (OUTPUT / "reports" / f"{stage}.json").exists():
        raise FileExistsError(f"{stage} result already exists; refusing to overwrite")


def contact_and_components(gt):
    labels = sitk.GetArrayViewFromImage(gt)
    spacing = gt.GetSpacing()
    structure = ndimage.generate_binary_structure(3, 1)
    result = {}
    for label, name in ORGAN_NAMES.items():
        mask = labels == label
        _, count = ndimage.label(mask, structure=structure)
        result[name] = {"present": bool(mask.any()), "components_6_connected": int(count),
                        "volume_mm3": int(mask.sum()) * float(np.prod(spacing))}
    heart, aorta = labels == 2, labels == 4
    face_area = (spacing[0] * spacing[1], spacing[0] * spacing[2],
                 spacing[1] * spacing[2])  # z, y, x array axes
    area = 0.0
    for axis, mm2 in enumerate(face_area):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        area += int(np.count_nonzero(heart[tuple(lo)] & aorta[tuple(hi)]) +
                    np.count_nonzero(aorta[tuple(lo)] & heart[tuple(hi)])) * mm2
    return {"organs": result, "heart_aorta_face_contact_mm2": area,
            "heart_aorta_touch_6_connected": bool(area > 0),
            "connectivity": "6-neighbor voxel faces; areas use physical face sizes"}


def figure(target_ct, target_gt, stages):
    fixed = sitk.GetArrayViewFromImage(target_ct)
    target_labels = sitk.GetArrayViewFromImage(target_gt)
    arrays = {name: (sitk.GetArrayViewFromImage(pair[0]),
                     sitk.GetArrayViewFromImage(pair[1])) for name, pair in stages.items()}
    columns = ("target",) + tuple(stages)
    fig, axes = plt.subplots(4, len(columns), figsize=(5 * len(columns), 17), dpi=150)
    for row, (label, name) in enumerate(ORGAN_NAMES.items()):
        z = int(np.median(np.flatnonzero((target_labels == label).any(axis=(1, 2)))))
        for col, stage in enumerate(columns):
            ct, gt = (fixed, target_labels) if stage == "target" else arrays[stage]
            ax = axes[row, col]
            ax.imshow(ct[z], cmap="gray", vmin=-1000, vmax=400, origin="lower")
            if np.any(gt[z] == label):
                ax.contour(gt[z] == label, levels=[0.5], colors=["#ffb703"], linewidths=0.8)
            ax.set_title(f"{name}: {stage} CT + own GT\ntarget z={z}")
            ax.axis("off")
    fig.suptitle("Identical target physical slices; contours belong to each panel's CT")
    fig.tight_layout()
    path = OUTPUT / "figures" / ("rigid_comparison.png" if "final" not in stages else "final_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def stage_report(name, source_ct, source_gt, target_ct, target_gt, transform, cfg):
    ct, gt = warp_pair(source_ct, source_gt, target_ct, transform)
    if geometry(ct) != geometry(target_ct) or geometry(gt) != geometry(target_ct):
        raise ValueError(f"{name}: target geometry mismatch")
    check_pair(ct, gt)
    directory = OUTPUT / "volumes" / name / "train" / SID
    save_image(ct, directory / f"{SID}.nii.gz")
    save_image(gt, directory / "GT.nii.gz")
    report = {"images": image_qc(ct, target_ct),
              "organs": organ_qc(gt, target_gt, source_gt),
              "topology": contact_and_components(gt),
              "ct_geometry": image_geometry(ct), "gt_geometry": image_geometry(gt)}
    if name != "initial":
        report["preprocessing_qc"] = preprocessing_qc(cfg, SID, OUTPUT / "volumes" / name,
                                                      OUTPUT / "reports")
    return (ct, gt), report


def rigid_run():
    output_paths("rigid")
    started = time.monotonic()
    cfg, paths = inputs()
    hashes = {key: sha256(path) for key, path in paths.items()}
    source_ct = sitk.ReadImage(str(paths["native_source_ct"]), sitk.sitkFloat32)
    source_gt = sitk.ReadImage(str(paths["native_source_gt"]), sitk.sitkUInt8)
    target_ct = sitk.ReadImage(str(paths["native_target_ct"]), sitk.sitkFloat32)
    target_gt = sitk.ReadImage(str(paths["native_target_gt"]), sitk.sitkUInt8)
    if geometry(source_ct) != geometry(source_gt) or geometry(target_ct) != geometry(target_gt):
        raise ValueError("native CT/GT geometry mismatch")
    params = parameters(cfg["data"]["registration"].get("params"))
    params["threads"] = 1
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    fixed, moving = estimation_image(target_ct, params), estimation_image(source_ct, params)
    fixed_body, moving_body = metric_mask(fixed, params), metric_mask(moving, params)
    initial = sitk.CenteredTransformInitializer(
        fixed_body, moving_body, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS)
    method = registration_method(params, fixed_body)
    method.SetInitialTransform(initial, inPlace=False)
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=0.01, numberOfIterations=params["rigid_iterations"])
    method.SetOptimizerScalesFromPhysicalShift()
    stage_started = time.monotonic()
    try:
        rigid = method.Execute(fixed, moving)
    except Exception:
        print(f"rigid optimizer failed after {time.monotonic() - stage_started:.2f}s", flush=True)
        raise
    optimizer = {"metric": method.GetMetricValue(),
                 "iterations": method.GetOptimizerIteration(),
                 "stop": method.GetOptimizerStopConditionDescription(),
                 "runtime_seconds": time.monotonic() - stage_started}
    print(f"rigid optimizer: {optimizer}", flush=True)
    stages, reports = {}, {}
    for name, transform in (("initial", initial), ("rigid", rigid)):
        save_transform(transform, OUTPUT / "reports" / f"{name}_transform.h5", target_ct)
        stages[name], reports[name] = stage_report(
            name, source_ct, source_gt, target_ct, target_gt, transform, cfg)
    baseline_stages = {}
    for name in ("initial", "rigid"):
        root = BASE / "volumes" / name / "train" / SID
        baseline_stages[f"baseline_{name}"] = (
            sitk.ReadImage(str(root / f"{SID}.nii.gz")), sitk.ReadImage(str(root / "GT.nii.gz")))
    plots = figure(target_ct, target_gt, baseline_stages | stages)
    if {key: sha256(path) for key, path in paths.items()} != hashes:
        raise ValueError("source/cache input changed during diagnostic")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report = {"pair": {"fold": 0, "source": "Patient_26", "target": "Patient_19",
                       "pairing_seed": cfg["data"]["registration"]["pairing_seed"]},
              "initializer": "MOMENTS of CT-derived body masks from estimation images",
              "initializer_fixed_ct_body_voxels": int(sitk.GetArrayViewFromImage(fixed_body).sum()),
              "initializer_moving_ct_body_voxels": int(sitk.GetArrayViewFromImage(moving_body).sum()),
              "input_paths": {k: str(v) for k, v in paths.items()}, "input_sha256": hashes,
              "geometry": {"source_ct": image_geometry(source_ct), "target_ct": image_geometry(target_ct),
                           "fixed_estimation": image_geometry(fixed),
                           "moving_estimation": image_geometry(moving)},
              "params": params, "simpleitk_version": sitk.Version_VersionString(),
              "thread_environment": {key: os.environ.get(key) for key in
                                     ("OMP_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
                                      "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
              "simpleitk_threads": sitk.ProcessObject.GetGlobalDefaultNumberOfThreads(),
              "optimizer": optimizer, "source_topology": contact_and_components(source_gt),
              "target_topology": contact_and_components(target_gt),
              "stages": reports, "figure": plots,
              "runtime": {"wall_seconds": time.monotonic() - started,
                          "user_cpu_seconds": usage.ru_utime,
                          "system_cpu_seconds": usage.ru_stime,
                          "peak_rss_kib": usage.ru_maxrss}}
    (OUTPUT / "reports" / "rigid.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"rigid_stage": {name: {"body_dice": r["images"]["body_dice"],
                                             "ct_mae_hu": r["images"]["ct_mae_hu"]}
                                      for name, r in reports.items()},
                      "report": str(OUTPUT / "reports/rigid.json")}), flush=True)


def final_run():
    output_paths("final")
    started = time.monotonic()
    cfg, paths = inputs()
    hashes = {key: sha256(path) for key, path in paths.items()}
    source_ct = sitk.ReadImage(str(paths["native_source_ct"]), sitk.sitkFloat32)
    source_gt = sitk.ReadImage(str(paths["native_source_gt"]), sitk.sitkUInt8)
    target_ct = sitk.ReadImage(str(paths["native_target_ct"]), sitk.sitkFloat32)
    target_gt = sitk.ReadImage(str(paths["native_target_gt"]), sitk.sitkUInt8)
    params = parameters(cfg["data"]["registration"].get("params"))
    params["threads"] = 1
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    fixed, moving = estimation_image(target_ct, params), estimation_image(source_ct, params)
    mask = metric_mask(fixed, params)
    rigid = sitk.ReadTransform(str(OUTPUT / "reports/rigid_transform.h5"))
    extent = [(n - 1) * s for n, s in zip(fixed.GetSize(), fixed.GetSpacing())]
    mesh = [max(1, round(length / params["bspline_spacing_mm"])) for length in extent]
    bspline = sitk.BSplineTransformInitializer(fixed, mesh, order=3)
    method = registration_method(params, mask)
    method.SetMovingInitialTransform(rigid)
    method.SetInitialTransform(bspline, inPlace=True)
    bound = params["bspline_coefficient_bound_mm"]
    method.SetOptimizerAsLBFGSB(numberOfIterations=params["bspline_iterations"],
                                maximumNumberOfFunctionEvaluations=500,
                                lowerBound=-bound, upperBound=bound)
    stage_started = time.monotonic()
    try:
        fitted_bspline = method.Execute(fixed, moving)
    except Exception:
        print(f"B-spline optimizer failed after {time.monotonic() - stage_started:.2f}s", flush=True)
        raise
    optimizer = {"metric": method.GetMetricValue(), "iterations": method.GetOptimizerIteration(),
                 "stop": method.GetOptimizerStopConditionDescription(), "mesh_size": mesh,
                 "runtime_seconds": time.monotonic() - stage_started}
    final = sitk.CompositeTransform(3)
    final.AddTransform(rigid)
    final.AddTransform(fitted_bspline)
    save_transform(final, OUTPUT / "reports/final_transform.h5", target_ct)
    pair, report = stage_report("final", source_ct, source_gt, target_ct, target_gt, final, cfg)
    stage_pairs = {}
    for name in ("initial", "rigid"):
        root = OUTPUT / "volumes" / name / "train" / SID
        stage_pairs[name] = (sitk.ReadImage(str(root / f"{SID}.nii.gz")),
                             sitk.ReadImage(str(root / "GT.nii.gz")))
    stage_pairs["final"] = pair
    plot = figure(target_ct, target_gt, stage_pairs)
    deformation = deformation_qc(OUTPUT / "reports/final_transform.h5", target_ct,
                                 OUTPUT / "figures")
    if {key: sha256(path) for key, path in paths.items()} != hashes:
        raise ValueError("source/cache input changed during diagnostic")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = {"optimizer": optimizer, "stage": report, "deformation": deformation,
              "figure": plot, "input_sha256": hashes,
              "runtime": {"wall_seconds": time.monotonic() - started,
                          "user_cpu_seconds": usage.ru_utime,
                          "system_cpu_seconds": usage.ru_stime,
                          "peak_rss_kib": usage.ru_maxrss}}
    (OUTPUT / "reports/final.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"final_body_dice": report["images"]["body_dice"],
                      "final_ct_mae_hu": report["images"]["ct_mae_hu"],
                      "report": str(OUTPUT / "reports/final.json")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("rigid", "final"), required=True)
    stage = parser.parse_args().stage
    (rigid_run if stage == "rigid" else final_run)()
