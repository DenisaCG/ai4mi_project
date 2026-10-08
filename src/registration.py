"""Offline, fold-specific 3D registration augmentation for corrected Full40 SegTHOR.

Build with both arm configs so one set of synthetic NIfTIs is sliced twice:
python -m src.registration --configs configs/full_cv4_enet_ce_registration.yaml \
    configs/full_cv4_enet_dice_ce_registration.yaml --fold 0 --limit 1 --visualize
Omit --limit to complete a fold; completed pairs are reused on a restart.
"""

import argparse
import json
import os
import random
import shutil
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.config import REPO, load_config
from src.folds import run_configs
from src.splits import validate_split


DEFAULT_PARAMETERS = {
    "estimation_spacing": [2.0, 2.0, 2.5],
    "hu_window": [-1000.0, 400.0],
    "body_threshold_hu": -500.0,
    "mi_bins": 50,
    "sampling_fraction": 0.05,
    "metric_seed": 4242,  # zero means wall-clock seeding in SimpleITK
    "shrink_factors": [4, 2, 1],
    "smoothing_sigmas_mm": [4.0, 2.0, 0.0],
    "bspline_spacing_mm": 80.0,
    "bspline_coefficient_bound_mm": 10.0,
    "rigid_iterations": 150,
    "bspline_iterations": 75,
    "threads": 4,
}
SYNTHETIC_PREFIX = "Synthetic_f"


def parameters(overrides: dict | None = None) -> dict:
    result = deepcopy(DEFAULT_PARAMETERS)
    unknown = set(overrides or {}) - set(result)
    if unknown:
        raise ValueError(f"unknown registration parameters: {sorted(unknown)}")
    result.update(overrides or {})
    if (len(result["estimation_spacing"]) != 3 or
            any(float(x) <= 0 for x in result["estimation_spacing"])):
        raise ValueError("registration estimation_spacing must contain three positive values")
    if len(result["hu_window"]) != 2 or result["hu_window"][0] >= result["hu_window"][1]:
        raise ValueError("registration hu_window must be [low, high]")
    if (len(result["shrink_factors"]) != len(result["smoothing_sigmas_mm"]) or
            any(x < 1 for x in result["shrink_factors"])):
        raise ValueError("registration pyramid settings must have matching positive levels")
    for key in ("mi_bins", "metric_seed", "rigid_iterations", "bspline_iterations", "threads"):
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError(f"registration {key} must be a positive integer")
    for key in ("sampling_fraction", "bspline_spacing_mm", "bspline_coefficient_bound_mm"):
        if not 0 < result[key] <= (1 if key == "sampling_fraction" else float("inf")):
            raise ValueError(f"registration {key} must be positive")
    return result


def fold_patients(source_dir: Path, retains: int, fold: int,
                  num_folds: int, split_seed: int) -> tuple[list[str], list[str]]:
    """Replicate slice_segthor.get_splits without altering the process-wide RNG."""
    ids = sorted(p.name for p in (source_dir / "train").glob("Patient_*"))
    if len(ids) != 40:
        raise ValueError(f"registration requires corrected Full40: found {len(ids)} patients")
    validate_split(len(ids), retains, fold, num_folds)
    random.Random(split_seed).shuffle(ids)
    val = ids[fold * retains:(fold + 1) * retains]
    train = sorted(set(ids) - set(val))
    if len(train) != 30 or len(val) != 10:
        raise ValueError(f"fold {fold}: expected 30 train / 10 val, got {len(train)} / {len(val)}")
    return train, sorted(val)


def cyclic_pairs(train: list[str], fold: int, pairing_seed: int) -> list[dict]:
    order = sorted(train)
    random.Random(pairing_seed).shuffle(order)
    return [
        {"synthetic_id": f"{SYNTHETIC_PREFIX}{fold}_{i:02d}",
         "source_patient": source, "target_patient": order[(i + 1) % len(order)]}
        for i, source in enumerate(order)
    ]


def fold_root(cfg: dict) -> Path:
    registration = cfg["data"]["registration"]
    root = Path(registration["root"])
    return (REPO / root) / f"fold{cfg['data']['preprocess']['fold']}"


def ensure_storage(cfg: dict) -> None:
    """Keep generated volumes on scratch, matching the pipeline's sliced-cache layout."""
    root = (REPO / cfg["data"]["registration"]["root"])
    if root.is_symlink() and not root.exists():
        raise FileNotFoundError(f"{root} points to missing scratch storage")
    if root.exists():
        if not root.is_symlink():
            raise ValueError(f"{root} is a real directory; registration data need a scratch symlink")
        return
    scratch = Path(os.environ.get(
        "SCRATCH_ROOT", f"/scratch-shared/{os.environ['USER']}/ai4mi_project"
    )) / root.name
    scratch.mkdir(parents=True, exist_ok=True)
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        root.symlink_to(scratch, target_is_directory=True)
    except FileExistsError:
        if not root.is_symlink() or not root.exists():
            raise


def processed_root(cfg: dict) -> Path:
    return fold_root(cfg) / "processed" / Path(cfg["data"]["root"]).name


def fold_plan(cfg: dict) -> tuple[list[str], list[str], list[dict]]:
    p = cfg["data"]["preprocess"]
    if p.get("gt_version") != "corrected" or p.get("num_folds") != 4 or p["retains"] != 10:
        raise ValueError("registration needs a resolved corrected Full40 four-fold config")
    train, val = fold_patients(REPO / p["source_dir"], p["retains"], p["fold"],
                               p["num_folds"], p["seed"])
    pairs = cyclic_pairs(train, p["fold"], cfg["data"]["registration"]["pairing_seed"])
    if set(train) & set(val) or any(
        e["source_patient"] in val or e["target_patient"] in val for e in pairs
    ):
        raise ValueError(f"fold {p['fold']}: validation patient selected for registration")
    return train, val, pairs


def expected_manifest(cfg: dict) -> dict:
    train, val, pairs = fold_plan(cfg)
    reg = cfg["data"]["registration"]
    params = parameters(reg.get("params"))
    root = fold_root(cfg)
    entries = []
    for pair in pairs:
        sid = pair["synthetic_id"]
        out = root / "train" / sid
        entries.append(pair | {
            "fold": cfg["data"]["preprocess"]["fold"],
            "pairing_seed": reg["pairing_seed"],
            "registration_parameters": params,
            "simpleitk_version": sitk.Version_VersionString(),
            "output_ct": str(out / f"{sid}.nii.gz"),
            "output_gt": str(out / "GT.nii.gz"),
            "registration_status": "pending",
            "processed": {},
        })
    return {
        "schema_version": 1,
        "source_dir": cfg["data"]["preprocess"]["source_dir"],
        "fold": cfg["data"]["preprocess"]["fold"],
        "split_seed": cfg["data"]["preprocess"]["seed"],
        "pairing_seed": reg["pairing_seed"],
        "parameters": params,
        "simpleitk_version": sitk.Version_VersionString(),
        "train_patients": train,
        "validation_patients": val,
        "pairs": entries,
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.replace(path)


def load_manifest(cfg: dict, create: bool = False) -> dict:
    path = fold_root(cfg) / "manifest.json"
    expected = expected_manifest(cfg)
    if not path.exists():
        if not create:
            raise FileNotFoundError(f"{path} missing: build registration data first")
        write_manifest(path, expected)
        return expected
    saved = json.loads(path.read_text())
    for key in ("schema_version", "source_dir", "fold", "split_seed", "pairing_seed",
                "parameters", "simpleitk_version", "train_patients", "validation_patients"):
        if saved.get(key) != expected[key]:
            raise ValueError(f"{path}: {key} differs from the requested registration plan")
    if [(x["synthetic_id"], x["source_patient"], x["target_patient"]) for x in saved["pairs"]] != [
        (x["synthetic_id"], x["source_patient"], x["target_patient"]) for x in expected["pairs"]
    ]:
        raise ValueError(f"{path}: stored pairing differs from the requested plan")
    return saved


def geometry(image: sitk.Image) -> tuple:
    return (image.GetSize(), image.GetSpacing(), image.GetOrigin(), image.GetDirection())


def estimation_image(image: sitk.Image, params: dict) -> sitk.Image:
    spacing = params["estimation_spacing"]
    size = [max(1, round((n - 1) * old / new) + 1)
            for n, old, new in zip(image.GetSize(), image.GetSpacing(), spacing)]
    ref = sitk.Image(size, sitk.sitkFloat32)
    ref.SetOrigin(image.GetOrigin())
    ref.SetDirection(image.GetDirection())
    ref.SetSpacing(spacing)
    low, high = params["hu_window"]
    clipped = sitk.Clamp(image, sitk.sitkFloat32, low, high)
    return sitk.Resample(clipped, ref, sitk.Transform(3, sitk.sitkIdentity),
                         sitk.sitkLinear, low, sitk.sitkFloat32)


def metric_mask(fixed: sitk.Image, params: dict) -> sitk.Image:
    mask = sitk.Cast(fixed > params["body_threshold_hu"], sitk.sitkUInt8)
    mask = sitk.BinaryMorphologicalClosing(mask, [2, 2, 1])
    mask = sitk.Cast(sitk.RelabelComponent(sitk.ConnectedComponent(mask)) == 1, sitk.sitkUInt8)
    mask = sitk.BinaryFillhole(mask)
    if not np.any(sitk.GetArrayViewFromImage(mask)):
        raise ValueError("fixed CT produced an empty registration body mask")
    return mask


def registration_method(params: dict, mask: sitk.Image) -> sitk.ImageRegistrationMethod:
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(params["mi_bins"])
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(params["sampling_fraction"], params["metric_seed"])
    method.SetMetricFixedMask(mask)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetShrinkFactorsPerLevel(params["shrink_factors"])
    method.SetSmoothingSigmasPerLevel(params["smoothing_sigmas_mm"])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return method


def estimate_transform(source: sitk.Image, target: sitk.Image,
                       params: dict) -> tuple[sitk.Transform, dict]:
    moving, fixed = estimation_image(source, params), estimation_image(target, params)
    mask = metric_mask(fixed, params)
    rigid0 = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    rigid_method = registration_method(params, mask)
    rigid_method.SetInitialTransform(rigid0, inPlace=False)
    rigid_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=0.01, numberOfIterations=params["rigid_iterations"]
    )
    rigid_method.SetOptimizerScalesFromPhysicalShift()
    rigid = rigid_method.Execute(fixed, moving)
    rigid_info = {"metric": rigid_method.GetMetricValue(),
                  "iterations": rigid_method.GetOptimizerIteration(),
                  "stop": rigid_method.GetOptimizerStopConditionDescription()}

    extent = [(n - 1) * s for n, s in zip(fixed.GetSize(), fixed.GetSpacing())]
    mesh = [max(1, round(length / params["bspline_spacing_mm"])) for length in extent]
    bspline = sitk.BSplineTransformInitializer(fixed, mesh, order=3)
    # CompositeTransform applies its back transform first. Only the back B-spline
    # parameters are exposed to the optimizer; the fitted rigid stays fixed.
    composite = sitk.CompositeTransform(3)
    composite.AddTransform(rigid)
    composite.AddTransform(bspline)
    deform_method = registration_method(params, mask)
    deform_method.SetInitialTransform(composite, inPlace=True)
    bound = params["bspline_coefficient_bound_mm"]
    deform_method.SetOptimizerAsLBFGSB(
        numberOfIterations=params["bspline_iterations"],
        maximumNumberOfFunctionEvaluations=500,
        lowerBound=-bound, upperBound=bound
    )
    final = deform_method.Execute(fixed, moving)
    deform_info = {"metric": deform_method.GetMetricValue(),
                   "iterations": deform_method.GetOptimizerIteration(),
                   "stop": deform_method.GetOptimizerStopConditionDescription(),
                   "mesh_size": mesh}
    return final, {"rigid": rigid_info, "bspline": deform_info}


def warp_pair(source_ct: sitk.Image, source_gt: sitk.Image, target_ct: sitk.Image,
              transform: sitk.Transform) -> tuple[sitk.Image, sitk.Image]:
    """The transform maps fixed/target points to moving/source points for Resample."""
    ct = sitk.Resample(source_ct, target_ct, transform, sitk.sitkLinear, -1000.0,
                       sitk.sitkFloat32)
    if not np.isfinite(sitk.GetArrayViewFromImage(ct)).all():
        raise ValueError("warped CT contains non-finite values")
    ct = sitk.Cast(sitk.Round(ct), sitk.sitkInt32)
    gt = sitk.Resample(source_gt, target_ct, transform, sitk.sitkNearestNeighbor,
                       0, sitk.sitkUInt8)
    return ct, gt


def check_pair(ct: sitk.Image, gt: sitk.Image) -> None:
    if geometry(ct) != geometry(gt):
        raise ValueError("warped CT and GT geometry differ")
    image = sitk.GetArrayViewFromImage(ct)
    if not np.isfinite(image).all():
        raise ValueError("warped CT contains non-finite values")
    if image.min() < -1000 or image.max() > 31743:
        raise ValueError(f"warped CT HU range {image.min()}..{image.max()} violates slicer bounds")
    labels = set(int(x) for x in np.unique(sitk.GetArrayViewFromImage(gt)))
    if labels != {0, 1, 2, 3, 4}:
        raise ValueError(f"warped GT labels {sorted(labels)}; expected all of 0..4")


def register_one(cfg: dict, entry: dict, params: dict) -> dict:
    source_dir = REPO / cfg["data"]["preprocess"]["source_dir"] / "train"
    source_id, target_id = entry["source_patient"], entry["target_patient"]
    source_ct = sitk.ReadImage(str(source_dir / source_id / f"{source_id}.nii.gz"), sitk.sitkFloat32)
    source_gt = sitk.ReadImage(str(source_dir / source_id / "GT.nii.gz"), sitk.sitkUInt8)
    target_ct = sitk.ReadImage(str(source_dir / target_id / f"{target_id}.nii.gz"), sitk.sitkFloat32)
    print(f"{source_id} CT {geometry(source_ct)}; GT {geometry(source_gt)}", flush=True)
    print(f"{target_id} CT {geometry(target_ct)}", flush=True)
    if geometry(source_ct) != geometry(source_gt):
        raise ValueError(f"{source_id}: CT and GT physical geometry differ")
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(params["threads"])
    started = time.monotonic()
    transform, info = estimate_transform(source_ct, target_ct, params)
    ct, gt = warp_pair(source_ct, source_gt, target_ct, transform)
    check_pair(ct, gt)
    out_ct, out_gt = Path(entry["output_ct"]), Path(entry["output_gt"])
    out_ct.parent.mkdir(parents=True, exist_ok=True)
    tmp_ct = out_ct.with_name("ct_tmp.nii.gz")
    tmp_gt = out_gt.with_name("gt_tmp.nii.gz")
    sitk.WriteImage(ct, str(tmp_ct))
    sitk.WriteImage(gt, str(tmp_gt))
    os.replace(tmp_ct, out_ct)
    os.replace(tmp_gt, out_gt)
    info["runtime_seconds"] = round(time.monotonic() - started, 2)
    info["source_geometry"] = geometry(source_ct)
    info["target_geometry"] = geometry(target_ct)
    return info


def preprocess_one(cfg: dict, entry: dict) -> int:
    """Slice a generated NIfTI with the real fold's unchanged preprocessing statistics."""
    from slice_segthor import slice_patient

    p = cfg["data"]["preprocess"]
    real_cache = REPO / cfg["data"]["root"]
    if not real_cache.is_dir():
        raise FileNotFoundError(f"{real_cache} missing: build the reference fold cache first")
    stats = json.loads((real_cache / "ct_norm_stats.json").read_text()) if p.get("normalize") else None
    roi = json.loads((real_cache / "roi_crop.json").read_text()) if p.get("crop") else None
    if stats and set(stats["train_patients"]) != set(fold_plan(cfg)[0]):
        raise ValueError(f"{real_cache}: normalization statistics are not from real fold training patients")
    if roi and set(roi["required_train"]) != set(fold_plan(cfg)[0]):
        raise ValueError(f"{real_cache}: ROI size is not from real fold training patients")
    spacing = (tuple(stats["target_spacing"]) if stats else
               tuple(roi["target_spacing"]) if roi else None)
    if p.get("resample") and spacing is None:
        raise ValueError(f"{real_cache}: missing real-fold target spacing")
    side = roi["size"] if roi else None
    cache = processed_root(cfg)
    sid = entry["synthetic_id"]
    tmp = cache / f"tmp_{sid}"
    if tmp.exists():
        shutil.rmtree(tmp)
    slice_patient(sid, tmp / "train", fold_root(cfg), (side, side) if side else tuple(p["shape"]),
                  gt_version="corrected", target_spacing=spacing, norm_stats=stats, crop_size=side)
    img = sorted((tmp / "train" / "img").glob(f"{sid}_*.png"))
    gt = sorted((tmp / "train" / "gt").glob(f"{sid}_*.png"))
    if not img or [x.name for x in img] != [x.name for x in gt]:
        raise ValueError(f"{sid}: synthetic image/GT slices differ")
    for kind, files in (("img", img), ("gt", gt)):
        destination = cache / "train" / kind
        destination.mkdir(parents=True, exist_ok=True)
        for stale in destination.glob(f"{sid}_*.png"):
            stale.unlink()
        for file in files:
            file.replace(destination / file.name)
    shutil.rmtree(tmp)
    return len(img)


def synthetic_slice_files(cfg: dict) -> list[tuple[Path, Path]]:
    """Require the complete 30-pair fold before training; never touches validation."""
    manifest = load_manifest(cfg)
    train, val, expected = fold_plan(cfg)
    if len(manifest["pairs"]) != 30 or len(expected) != 30:
        raise ValueError("registration fold must contain exactly 30 planned pairs")
    cache_key = Path(cfg["data"]["root"]).name
    missing = [e["synthetic_id"] for e in manifest["pairs"]
               if e["registration_status"] != "complete" or
               not Path(e["output_ct"]).is_file() or not Path(e["output_gt"]).is_file() or
               e.get("processed", {}).get(cache_key) != "complete"]
    if missing:
        raise ValueError(f"registration fold {manifest['fold']} incomplete ({len(missing)}/30): {missing[:5]}")
    if set(train) & set(val):
        raise ValueError("registration train/validation leakage")
    img_root, gt_root = processed_root(cfg) / "train" / "img", processed_root(cfg) / "train" / "gt"
    pairs = []
    for entry in manifest["pairs"]:
        sid = entry["synthetic_id"]
        images, labels = sorted(img_root.glob(f"{sid}_*.png")), sorted(gt_root.glob(f"{sid}_*.png"))
        if not images or [p.name for p in images] != [p.name for p in labels]:
            raise ValueError(f"{sid}: missing or mismatched synthetic slices in {processed_root(cfg)}")
        pairs.extend(zip(images, labels))
    return pairs


def visualize_one(cfg: dict, entry: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source_dir = REPO / cfg["data"]["preprocess"]["source_dir"] / "train"
    source_id, target_id = entry["source_patient"], entry["target_patient"]
    source = sitk.GetArrayFromImage(sitk.ReadImage(str(source_dir / source_id / f"{source_id}.nii.gz")))
    source_gt = sitk.GetArrayFromImage(sitk.ReadImage(str(source_dir / source_id / "GT.nii.gz")))
    target = sitk.GetArrayFromImage(sitk.ReadImage(str(source_dir / target_id / f"{target_id}.nii.gz")))
    warped = sitk.GetArrayFromImage(sitk.ReadImage(entry["output_ct"]))
    warped_gt = sitk.GetArrayFromImage(sitk.ReadImage(entry["output_gt"]))
    source_levels = np.quantile(np.flatnonzero((source_gt > 0).any(axis=(1, 2))), [0.25, 0.5, 0.75]).astype(int)
    target_levels = np.quantile(np.flatnonzero((warped_gt > 0).any(axis=(1, 2))), [0.25, 0.5, 0.75]).astype(int)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for row, (zs, zt) in enumerate(zip(source_levels, target_levels)):
        for ax, image, labels, z, title in (
            (axes[row, 0], source, source_gt, zs, f"{source_id} source"),
            (axes[row, 1], target, None, zt, f"{target_id} target"),
            (axes[row, 2], warped, warped_gt, zt, f"{source_id} warped to {target_id}"),
        ):
            ax.imshow(image[z], cmap="gray", vmin=-1000, vmax=400)
            if labels is not None:
                ax.imshow(np.ma.masked_equal(labels[z], 0), cmap="turbo", vmin=1, vmax=4, alpha=0.4)
            ax.set_title(f"{title}, slice {z}")
            ax.axis("off")
    fig.tight_layout()
    output = fold_root(cfg) / "qc" / f"{entry['synthetic_id']}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120)
    plt.close(fig)
    return output


def build(configs: list[dict], fold: int, limit: int | None = None,
          visualize: bool = False) -> dict:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    selected = [run_configs(cfg, fold=fold)[0] for cfg in configs]
    first = selected[0]
    split_keys = ("source_dir", "gt_version", "retains", "fold", "num_folds", "seed")
    for cfg in selected[1:]:
        if (cfg["data"]["registration"] != first["data"]["registration"] or
                any(cfg["data"]["preprocess"][key] != first["data"]["preprocess"][key]
                    for key in split_keys) or
                fold_plan(cfg) != fold_plan(first)):
            raise ValueError("registration arm configs must share source, split, pairing and parameters")
    ensure_storage(first)
    manifest = load_manifest(first, create=True)
    path = fold_root(first) / "manifest.json"
    selected_entries = manifest["pairs"][:limit]
    for entry in selected_entries:
        sid = entry["synthetic_id"]
        if (entry["registration_status"] != "complete" or
                not Path(entry["output_ct"]).is_file() or not Path(entry["output_gt"]).is_file()):
            print(f"registering {sid}: {entry['source_patient']} -> {entry['target_patient']}", flush=True)
            try:
                entry["registration_status"] = "running"
                write_manifest(path, manifest)
                entry["registration_info"] = register_one(first, entry, manifest["parameters"])
                entry["registration_status"] = "complete"
                write_manifest(path, manifest)
            except Exception as exc:
                entry["registration_status"] = "failed"
                entry["error"] = repr(exc)
                write_manifest(path, manifest)
                raise
        for cfg in selected:
            key = Path(cfg["data"]["root"]).name
            if entry["processed"].get(key) == "complete":
                continue
            print(f"preprocessing {sid} into {key}", flush=True)
            try:
                count = preprocess_one(cfg, entry)
                entry["processed"][key] = "complete"
                entry.setdefault("processed_slices", {})[key] = count
                write_manifest(path, manifest)
            except Exception as exc:
                entry["processed"][key] = f"failed: {exc!r}"
                write_manifest(path, manifest)
                raise
        if visualize:
            output = visualize_one(first, entry)
            entry["visualization"] = str(output)
            write_manifest(path, manifest)
            print(f"visualization: {output}", flush=True)
        print(f"{sid}: complete", flush=True)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, nargs="+", required=True,
                        help="registration arm configs sharing one synthetic NIfTI source")
    parser.add_argument("--fold", type=int, required=True, choices=range(4))
    parser.add_argument("--limit", type=int, help="process only the first N planned pairs")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args(argv)
    configs = [load_config(path) for path in args.configs]
    if any(cfg["data"]["registration"] is None for cfg in configs):
        parser.error("all configs must select data.registration")
    manifest = build(configs, args.fold, args.limit, args.visualize)
    print(f"fold {args.fold}: {sum(e['registration_status'] == 'complete' for e in manifest['pairs'])}/30 "
          f"registrations in {fold_root(run_configs(configs[0], fold=args.fold)[0])}", flush=True)


if __name__ == "__main__":
    main()
