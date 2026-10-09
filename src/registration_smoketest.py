"""One isolated fold-0 registration pair with reproducible numerical and visual QC."""

import hashlib
import json
import resource
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from PIL import Image

from slice_segthor import slice_patient
from src.config import REPO, load_config
from src.folds import run_configs
from src.registration import (check_pair, fold_plan, geometry, parameters,
                              register_one)


OUTPUT = REPO / "metrics/augmentation_metrics/registration/smoketest"
CONFIG = REPO / "configs/full_cv4_enet_dice_ce_registration.yaml"
ORGAN_NAMES = {1: "Esophagus", 2: "Heart", 3: "Trachea", 4: "Aorta"}
ORGAN_COLORS = ["#ffbc42", "#ef476f", "#06d6a0", "#118ab2"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_geometry(image: sitk.Image) -> dict:
    size, spacing, origin, direction = geometry(image)
    return {"size_xyz": size, "spacing_mm": spacing, "origin_mm": origin,
            "direction": direction}


def organ_counts(image: sitk.Image) -> dict:
    array = sitk.GetArrayViewFromImage(image)
    voxel_mm3 = float(np.prod(image.GetSpacing()))
    return {ORGAN_NAMES[label]: {"voxels": int(np.count_nonzero(array == label)),
                                 "volume_mm3": float(np.count_nonzero(array == label) * voxel_mm3)}
            for label in ORGAN_NAMES}


def intensity(image: sitk.Image) -> dict:
    array = sitk.GetArrayViewFromImage(image)
    return {"min_hu": float(array.min()), "max_hu": float(array.max()),
            "mean_hu": float(array.mean()), "all_finite": bool(np.isfinite(array).all())}


def preprocessing_qc(cfg: dict, sid: str, volumes: Path, reports: Path) -> dict:
    """Run the production slicer in temporary pilot space with real fold-0 statistics."""
    cache = REPO / cfg["data"]["root"]
    if not cache.is_dir():
        raise FileNotFoundError(f"real fold cache missing: {cache}")
    stats = json.loads((cache / "ct_norm_stats.json").read_text())
    roi = json.loads((cache / "roi_crop.json").read_text())
    train = set(fold_plan(cfg)[0])
    if set(stats["train_patients"]) != train or set(roi["required_train"]) != train:
        raise ValueError("fold-0 preprocessing metadata is not derived from its 30 real training patients")
    spacing = tuple(stats["target_spacing"])
    if spacing != tuple(roi["target_spacing"]):
        raise ValueError("fold-0 normalization and ROI target spacings differ")
    side = int(roi["size"])
    with tempfile.TemporaryDirectory(prefix="preprocessing_", dir=reports) as directory:
        temporary = Path(directory)
        slice_patient(sid, temporary / "train", volumes, (side, side),
                      gt_version="corrected", target_spacing=spacing,
                      norm_stats=stats, crop_size=side)
        resampled_gt = sitk.ReadImage(str(temporary / "resampled/train" / sid / "GT.nii.gz"))
        before = organ_counts(resampled_gt)
        cropped = json.loads((temporary / "roi_crop" / f"{sid}.json").read_text())
        counts = {name: 0 for name in ORGAN_NAMES.values()}
        files = sorted((temporary / "train/gt").glob(f"{sid}_*.png"))
        if not files:
            raise ValueError("production slicer generated no pilot GT slices")
        for file in files:
            with Image.open(file) as png:
                labels = np.asarray(png)
            if not np.isin(np.unique(labels), [0, 63, 126, 189, 252]).all():
                raise ValueError(f"unexpected label values in {file}")
            for label, name in ORGAN_NAMES.items():
                counts[name] += int(np.count_nonzero(labels == 63 * label))
        voxel_mm3 = float(np.prod(resampled_gt.GetSpacing()))
        after = {name: {"voxels": count, "volume_mm3": count * voxel_mm3,
                        "retained_fraction_of_resampled":
                        count / before[name]["voxels"] if before[name]["voxels"] else None}
                 for name, count in counts.items()}
        return {"real_cache": str(cache), "fold_train_patients": sorted(train),
                "normalization_stats": stats, "roi_metadata": roi,
                "resampled_gt_geometry": image_geometry(resampled_gt),
                "resampled_gt_organs": before, "roi_window": cropped,
                "post_roi_png_slices": len(files), "post_roi_organs": after}


def overlay(ax, image: np.ndarray, labels: np.ndarray | None, title: str) -> None:
    ax.imshow(image, cmap="gray", vmin=-1000, vmax=400)
    if labels is not None:
        colors = ListedColormap(ORGAN_COLORS)
        norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], colors.N)
        ax.imshow(np.ma.masked_equal(labels, 0), cmap=colors, norm=norm, alpha=0.5)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def figures(source_ct: sitk.Image, source_gt: sitk.Image, target_ct: sitk.Image,
            warped_ct: sitk.Image, warped_gt: sitk.Image, folder: Path,
            prefix: str = "") -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    source_on_target = sitk.Resample(source_ct, target_ct, sitk.Transform(3, sitk.sitkIdentity),
                                     sitk.sitkLinear, -1000., sitk.sitkFloat32)
    source_view = sitk.GetArrayFromImage(source_on_target)
    target_view = sitk.GetArrayFromImage(target_ct)
    warped_view = sitk.GetArrayFromImage(warped_ct)
    warped_labels = sitk.GetArrayFromImage(warped_gt)
    levels = np.quantile(np.flatnonzero((warped_labels > 0).any(axis=(1, 2))),
                         [0.25, 0.5, 0.75]).astype(int)
    fig, axes = plt.subplots(3, 3, figsize=(14, 13), dpi=200)
    for row, z in enumerate(levels):
        point = target_ct.TransformIndexToPhysicalPoint((0, 0, int(z)))
        caption = f"target plane {z}, origin {tuple(round(v, 1) for v in point)} mm"
        for col, (array, title) in enumerate(((source_view, "Source CT, same physical grid"),
                                               (target_view, "Target CT"),
                                               (warped_view, "Warped CT"))):
            overlay(axes[row, col], array[z], None, f"{title}\n{caption}")
    fig.suptitle("Registration overview; shared target-grid physical planes, HU window [-1000, 400]")
    fig.tight_layout()
    overview = folder / f"{prefix}registration_overview.png"
    fig.savefig(overview)
    plt.close(fig)

    source_view = sitk.GetArrayFromImage(source_ct)
    source_labels = sitk.GetArrayFromImage(source_gt)
    fig, axes = plt.subplots(4, 3, figsize=(15, 17), dpi=200)
    for row, (label, name) in enumerate(ORGAN_NAMES.items()):
        source_z = int(np.median(np.flatnonzero((source_labels == label).any(axis=(1, 2)))))
        warped_z = int(np.median(np.flatnonzero((warped_labels == label).any(axis=(1, 2)))))
        overlay(axes[row, 0], source_view[source_z], source_labels[source_z],
                f"{name}: source CT + source GT, z={source_z}")
        overlay(axes[row, 1], warped_view[warped_z], warped_labels[warped_z],
                f"{name}: warped CT + warped GT, z={warped_z}")
        overlay(axes[row, 2], target_view[warped_z], warped_labels[warped_z],
                f"{name}: target CT + warped GT, z={warped_z}")
    fig.legend(handles=[Patch(color=color, label=name) for name, color in
                        zip(ORGAN_NAMES.values(), ORGAN_COLORS)], loc="lower center", ncol=4)
    fig.suptitle("Organ-present slices; CT window [-1000, 400]; source and target use native grids")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    organs = folder / f"{prefix}organ_overlays.png"
    fig.savefig(organs)
    plt.close(fig)
    return [str(overview), str(organs)]


def deformation_qc(transform_path: Path, target: sitk.Image, folder: Path) -> dict:
    """Inspect full target-to-source mapping and B-spline on a 4x4x5 mm target grid."""
    transform = sitk.CompositeTransform(sitk.ReadTransform(str(transform_path)))
    if transform.GetNumberOfTransforms() != 2:
        raise ValueError("expected a rigid plus B-spline composite transform")
    spacing = (4., 4., 5.)
    size = [max(2, round((n - 1) * old / new) + 1)
            for n, old, new in zip(target.GetSize(), target.GetSpacing(), spacing)]
    grid = sitk.Image(size, sitk.sitkUInt8)
    grid.SetOrigin(target.GetOrigin())
    grid.SetDirection(target.GetDirection())
    grid.SetSpacing(spacing)
    field_filter = sitk.TransformToDisplacementFieldFilter()
    field_filter.SetReferenceImage(grid)
    field_filter.SetOutputPixelType(sitk.sitkVectorFloat32)
    full_field = field_filter.Execute(transform)
    bspline_field = field_filter.Execute(transform.GetNthTransform(1))
    full_mag = np.linalg.norm(sitk.GetArrayFromImage(full_field), axis=-1)
    bspline_mag = np.linalg.norm(sitk.GetArrayFromImage(bspline_field), axis=-1)
    jac = sitk.GetArrayFromImage(sitk.DisplacementFieldJacobianDeterminant(full_field))
    z = jac.shape[0] // 2
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), dpi=200)
    for ax, array, title, color in zip(axes, (bspline_mag, full_mag, jac),
                                        ("B-spline displacement (mm)",
                                         "Full target-to-source displacement (mm)",
                                         "Full-transform Jacobian determinant"),
                                        ("magma", "magma", "coolwarm")):
        display = ax.imshow(array[z], cmap=color)
        fig.colorbar(display, ax=ax, fraction=0.046)
        ax.set_title(f"{title}\ncoarse target-grid plane {z}")
        ax.axis("off")
    fig.tight_layout()
    path = folder / "deformation.png"
    fig.savefig(path)
    plt.close(fig)
    def summary(array):
        finite = array[np.isfinite(array)]
        return {"min": float(finite.min()) if finite.size else None,
                "median": float(np.median(finite)) if finite.size else None,
                "max": float(finite.max()) if finite.size else None,
                "nonfinite_voxels": int(array.size - finite.size)}
    return {"grid": image_geometry(grid), "mapping": "target physical point to source physical point",
            "bspline_displacement_mm": summary(bspline_mag),
            "full_displacement_mm": summary(full_mag),
            "full_jacobian_determinant": summary(jac) |
            {"nonpositive_voxels": int(np.count_nonzero(jac <= 0)),
             "nonpositive_fraction": float(np.count_nonzero(jac <= 0) / jac.size)},
            "figure": str(path)}


def main() -> None:
    started = time.monotonic()
    cfg = run_configs(load_config(CONFIG), fold=0)[0]
    _, _, pairs = fold_plan(cfg)
    entry = dict(pairs[0])
    if (entry["source_patient"], entry["target_patient"]) != ("Patient_26", "Patient_19"):
        raise ValueError(f"unexpected first fold-0 pair: {entry}")
    volumes, figure_dir, reports = (OUTPUT / name for name in ("volumes", "figures", "reports"))
    for directory in (volumes, figure_dir, reports):
        directory.mkdir(parents=True, exist_ok=True)
    sid = entry["synthetic_id"]
    entry["output_ct"] = str(volumes / "train" / sid / f"{sid}.nii.gz")
    entry["output_gt"] = str(volumes / "train" / sid / "GT.nii.gz")
    if not Path(entry["output_ct"]).is_file() or not Path(entry["output_gt"]).is_file():
        raise FileNotFoundError("previously saved smoketest CT/GT pair is missing")
    source_dir = REPO / cfg["data"]["preprocess"]["source_dir"] / "train"
    source = source_dir / entry["source_patient"]
    target = source_dir / entry["target_patient"]
    inputs = [source / f"{entry['source_patient']}.nii.gz", source / "GT.nii.gz",
              target / f"{entry['target_patient']}.nii.gz"]
    hashes = {str(path): sha256(path) for path in inputs}
    params = parameters(cfg["data"]["registration"].get("params"))
    print("QC of previously saved CT/GT begins before any new registration", flush=True)
    source_ct, source_gt, target_ct = (sitk.ReadImage(str(path)) for path in inputs)
    warped_ct = sitk.ReadImage(entry["output_ct"])
    warped_gt = sitk.ReadImage(entry["output_gt"])
    check_pair(warped_ct, warped_gt)
    if geometry(warped_ct) != geometry(warped_gt) or geometry(warped_ct) != geometry(target_ct):
        raise ValueError("warped CT/GT do not exactly match the target reference grid")
    retention = preprocessing_qc(cfg, sid, volumes, reports)
    image_figures = figures(source_ct, source_gt, target_ct, warped_ct, warped_gt, figure_dir)
    existing_qc = {"fold": 0, "source": entry["source_patient"],
                   "target": entry["target_patient"], "pairing_seed": 42,
                   "source_ct_geometry": image_geometry(source_ct),
                   "source_gt_geometry": image_geometry(source_gt),
                   "target_ct_geometry": image_geometry(target_ct),
                   "warped_ct_geometry": image_geometry(warped_ct),
                   "warped_gt_geometry": image_geometry(warped_gt),
                   "source_organs": organ_counts(source_gt),
                   "warped_organs": organ_counts(warped_gt),
                   "source_ct_intensity": intensity(source_ct),
                   "target_ct_intensity": intensity(target_ct),
                   "warped_ct_intensity": intensity(warped_ct),
                   "preprocessing": retention, "figures": image_figures,
                   "input_sha256_before": hashes}
    (reports / "qc_existing.json").write_text(json.dumps(existing_qc, indent=2, allow_nan=False) + "\n")
    (OUTPUT / "README.md").write_text(
        "# Fold-0 registration smoketest\n\n"
        "`reports/qc_existing.json` and the overview/organ-overlay PNGs are QC of the CT/GT "
        "saved by job 27793262. They were read back from disk and processed in an isolated "
        "temporary directory using the real fold-0 training statistics. The earlier transform "
        "export failed; deformation diagnostics are pending until a verified transform exists.\n"
    )
    print("saved-volume geometry, labels, intensity, ROI retention and figures complete", flush=True)

    # Preserve the known incomplete HDF5 from job 27793262, so the public path
    # remains absent until safe serialization and point-mapping validation succeed.
    transform_path = reports / "transform.h5"
    failed_transform = reports / "transform.failed_27793262.h5"
    if transform_path.exists():
        if failed_transform.exists():
            raise FileExistsError(f"cannot quarantine old partial transform: {failed_transform}")
        transform_path.rename(failed_transform)
    rerun_root = volumes / "rerun"
    rerun_entry = dict(entry)
    rerun_entry["output_ct"] = str(rerun_root / "train" / sid / f"{sid}.nii.gz")
    rerun_entry["output_gt"] = str(rerun_root / "train" / sid / "GT.nii.gz")
    if Path(rerun_entry["output_ct"]).exists() or Path(rerun_entry["output_gt"]).exists():
        raise FileExistsError("rerun CT/GT already exists; refusing to overwrite it")
    print("rerunning exactly the same fold-0 pair into separate volumes", flush=True)
    info = register_one(cfg, rerun_entry, params, transform_output=transform_path)
    repeated_ct = sitk.ReadImage(rerun_entry["output_ct"])
    repeated_gt = sitk.ReadImage(rerun_entry["output_gt"])
    check_pair(repeated_ct, repeated_gt)
    if geometry(repeated_ct) != geometry(target_ct):
        raise ValueError("rerun CT/GT do not match the target physical grid")
    old_ct, new_ct = sitk.GetArrayViewFromImage(warped_ct), sitk.GetArrayViewFromImage(repeated_ct)
    old_gt, new_gt = sitk.GetArrayViewFromImage(warped_gt), sitk.GetArrayViewFromImage(repeated_gt)
    ct_different = int(np.count_nonzero(old_ct != new_ct))
    gt_different = int(np.count_nonzero(old_gt != new_gt))
    comparison = {"ct_geometry_equal": geometry(warped_ct) == geometry(repeated_ct),
                  "gt_geometry_equal": geometry(warped_gt) == geometry(repeated_gt),
                  "ct_differing_voxels": ct_different,
                  "gt_differing_voxels": gt_different,
                  "ct_max_absolute_hu_difference":
                  int(np.max(np.abs(old_ct[old_ct != new_ct].astype(np.int64) -
                                    new_ct[old_ct != new_ct].astype(np.int64)))) if ct_different else 0}
    if not comparison["ct_geometry_equal"] or not comparison["gt_geometry_equal"]:
        raise ValueError("original and rerun pairs have different spatial geometry")
    if ct_different or gt_different:
        comparison["rerun_figures"] = figures(source_ct, source_gt, target_ct,
                                               repeated_ct, repeated_gt, figure_dir,
                                               prefix="rerun_")
        comparison["rerun_preprocessing"] = preprocessing_qc(cfg, sid, rerun_root, reports)
    (reports / "rerun_comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n")
    print(f"rerun comparison: {comparison['ct_differing_voxels']} CT and "
          f"{comparison['gt_differing_voxels']} GT voxels differ", flush=True)
    deformation = deformation_qc(transform_path, target_ct, figure_dir)
    if {str(path): sha256(path) for path in inputs} != hashes:
        raise ValueError("a source CT/GT or target CT changed during the smoketest")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    runtime = float(info["runtime_seconds"])
    report = existing_qc | {"synthetic_id": sid, "config": str(CONFIG),
              "registration_parameters": params,
              "simpleitk_version": sitk.Version_VersionString(), "registration": info,
              "rerun_comparison": comparison, "deformation": deformation,
              "input_sha256_unchanged": hashes, "figures": image_figures + [deformation["figure"]],
              "pilot_wall_seconds_before_report_write": round(time.monotonic() - started, 2),
              "process_user_cpu_seconds": round(usage.ru_utime, 2),
              "process_system_cpu_seconds": round(usage.ru_stime, 2),
              "process_peak_rss_kib": usage.ru_maxrss,
              "estimated_120_registration_seconds": round(120 * runtime, 2),
              "estimated_120_registration_hours": round(120 * runtime / 3600, 2)}
    (reports / "qc.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (OUTPUT / "README.md").write_text(
        "# Fold-0 registration smoketest\n\n"
        "One pair from the production cyclic plan: Patient_26 -> Patient_19, pairing seed 42. "
        "`volumes/train/Synthetic_f0_00/` preserves the CT/GT produced by job 27793262. "
        "`reports/qc_existing.json` records their disk-readback geometry, intensity, organ volumes "
        "and actual real-fold ROI retention. No production registration manifest or cache was touched.\n\n"
        "A second execution of the same pair writes CT/GT to `volumes/rerun/train/Synthetic_f0_00/` "
        "without overwriting the original. `reports/rerun_comparison.json` counts any differences. "
        "`reports/transform.h5` is the safely flattened, reread and point-verified transform "
        "for the rerun pair; the old incomplete file is `reports/transform.failed_27793262.h5`. "
        "If the pairs differ, use the rerun figures and preprocessing report to interpret the transform.\n\n"
        "`reports/qc.json` adds exact settings, optimizer outcomes, stage runtimes, source checksums, "
        "and deformation statistics.\n\n"
        "`figures/registration_overview.png` compares CTs on the same target physical grid; "
        "`figures/organ_overlays.png` selects organ-present slices on each native grid; "
        "`figures/deformation.png` shows the B-spline component, full mapping, and full Jacobian "
        "on a coarse 4 x 4 x 5 mm target grid. Nonpositive Jacobians there indicate possible folding; "
        "a positive coarse-grid result cannot exclude folding between samples.\n\n"
        "The 120-pair extrapolation multiplies this pair's registration-and-write runtime by 120. "
        "It excludes this pilot's figure and QC overhead; patient-to-patient runtime can vary. "
        "Automated checks do not establish anatomical plausibility. Inspect all figures before reuse.\n\n"
        "This QC completion was run with `sbatch "
        "jobsAndOutputs/augmentation/jobs/registration/smoketest.job` from the repository root "
        "after the original CT/GT pair from job 27793262 was present. "
        "The completed no-augmentation ENet and ResEnc + DS U-Net fold-0 control logs both "
        "recorded NVIDIA H100 GPUs; this registration generation uses only CPU resources.\n"
    )
    print(json.dumps({"output": str(OUTPUT), "registration_seconds": runtime,
                      "pilot_wall_seconds": report["pilot_wall_seconds_before_report_write"],
                      "estimated_120_hours": report["estimated_120_registration_hours"]}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
