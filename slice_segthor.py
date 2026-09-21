#!/usr/bin/env python3.7

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import pickle
import random
import argparse
import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import Callable

import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage.io import imsave
from skimage.transform import resize

from utils import map_, tqdm_


def norm_arr(img: np.ndarray) -> np.ndarray:
    casted = img.astype(np.float32)
    shifted = casted - casted.min()
    norm = shifted / shifted.max()
    res = 255 * norm

    assert 0 == res.min(), res.min()
    assert res.max() == 255, res.max()

    return res.astype(np.uint8)


def window_arr(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Fixed HU window [lo, hi] -> [0, 255]. Unlike norm_arr the mapping is the same for every patient."""
    lo, hi = np.float32(lo), np.float32(hi)
    res = (np.clip(img.astype(np.float32), lo, hi) - lo) / (hi - lo) * 255
    return np.rint(res).astype(np.uint8)


def ct_norm_stats(foreground_hu: np.ndarray) -> dict[str, float]:
    """CTNormalization stats of pooled foreground HU: clip window = 0.5th/99.5th percentile,
    mean/std of the foreground after clipping to that window."""
    lo, hi = np.percentile(foreground_hu, [0.5, 99.5])
    clipped = np.clip(foreground_hu, lo, hi)
    return {"lo": float(lo), "hi": float(hi), "mean": float(clipped.mean()), "std": float(clipped.std())}


def sanity_ct(ct, x, y, z, dx, dy, dz) -> bool:
    assert ct.dtype in [np.int16, np.int32], ct.dtype
    assert -1000 <= ct.min(), ct.min()
    assert ct.max() <= 31743, ct.max()

    assert 0.896 <= dx <= 1.37, dx  # Rounding error
    assert dx == dy
    assert 2 <= dz <= 3.7, dz

    assert (x, y) == (512, 512)
    assert x == y
    assert 135 <= z <= 284, z

    return True


def sanity_gt(gt, ct) -> bool:
    assert gt.shape == ct.shape
    assert gt.dtype in [np.uint8, np.int16], gt.dtype

    # moved check in the main function so we don't lose patient ID on error
    # Do the test on 3d: assume all organs are present..
    # assert set(np.unique(gt)) == set(range(5))

    return True


# Labels every patient must contain, per SEGTHOR version (see --gt_version)
EXPECTED_LABELS: dict[str, set[int]] = {"original": {0, 1, 2, 3}, "corrected": {0, 1, 2, 3, 4}}
# --normalize modes. Images are stored as uint8 PNGs, so z-scored floats can't be saved: both modes store the
# clipped [lo, hi] window rescaled to 0..255; `ct_window_zscore` also z-scores at load time (src/data.py).
NORMALIZE_MODES = ("ct_window", "ct_window_zscore")
NORM_STATS_FILE = "ct_norm_stats.json"
EXPECTED_WINDOW = (-991, 248)  # nnU-Net p0.5/p99.5 seen on this data during EDA, used only as a sanity check


resize_: Callable = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)


def median_target_spacing(spacings: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    """per-axis median: if max/min >= 3 the coarsest axis takes its 10th percentile."""
    s = np.asarray(spacings, dtype=float)
    target = np.median(s, axis=0)
    if target.max() / target.min() >= 3:
        coarsest = int(np.argmax(target))
        target[coarsest] = np.percentile(s[:, coarsest], 10)
    return tuple(float(t) for t in target)


def resample_image(ct: np.ndarray, spacing, target) -> np.ndarray:
    """Cubic (order=3). grid_mode=True aligns pixel edges, like skimage's resize used for slicing/stitching."""
    factors = np.asarray(spacing, dtype=float) / np.asarray(target, dtype=float)
    return ndimage.zoom(ct.astype(np.float32), factors, order=3, mode="nearest", grid_mode=True)


def resample_label(gt: np.ndarray, spacing, target) -> np.ndarray:
    """Nearest neighbour (order=0): never interpolate labels, that would invent in-between classes."""
    factors = np.asarray(spacing, dtype=float) / np.asarray(target, dtype=float)
    return ndimage.zoom(gt, factors, order=0, mode="nearest", grid_mode=True)


def resampled_affine(affine: np.ndarray, in_shape, out_shape) -> np.ndarray:
    """Affine of the resampled volume from the achieved (shape-rounded) voxel size, keeping the FOV in place."""
    ratio = np.asarray(in_shape, dtype=float) / np.asarray(out_shape, dtype=float)
    new = affine.copy()
    new[:3, :3] = affine[:3, :3] @ np.diag(ratio)
    new[:3, 3] = affine[:3, 3] + affine[:3, :3] @ ((ratio - 1) / 2)
    return new


ROI_BODY_HU = -500  # body vs air for the ROI window centre
ROI_MASK_STEP = 2  # the body mask is computed on every 2nd voxel: the centre only needs ~2 px accuracy
ROI_MARGIN, ROI_MULTIPLE = 15, 32  # voxels added on each side of the largest train requirement, then rounded up


def largest_components(mask: np.ndarray, min_fraction: float = 1.0) -> np.ndarray:
    """Union of the connected components at least `min_fraction` of the largest one (1.0 = largest only)."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask)
    sizes = np.bincount(lab.ravel())[1:]
    return np.isin(lab, np.flatnonzero(sizes >= min_fraction * sizes.max()) + 1)


def body_mask(ct: np.ndarray) -> np.ndarray:
    """Largest 3D component of HU > ROI_BODY_HU after an in-plane opening (drops thin couch rails and cables)."""
    return largest_components(ndimage.binary_opening(ct > ROI_BODY_HU, structure=np.ones((3, 3, 1))))


def inplane_extent(mask: np.ndarray, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """In-plane [lo, hi) of a 3D mask in voxels of the full grid (the mask may be subsampled by `step`)."""
    lo, hi = [], []
    for axis in (0, 1):
        idx = np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != axis)))
        lo.append(idx[0] * step)
        hi.append((idx[-1] + 1) * step)
    return np.array(lo, dtype=float), np.array(hi, dtype=float)


def roi_centre(ct: np.ndarray) -> np.ndarray:
    """In-plane (x, y) centre, in voxels, of the body's bounding box. Takes the IMAGE only: there are no labels at
    test time, so the window position must never depend on them (the labels only size the window, roi_required_size)."""
    small = ct[::ROI_MASK_STEP, ::ROI_MASK_STEP, ::ROI_MASK_STEP]
    mask = body_mask(small)
    if not mask.any():
        raise ValueError(f"no body voxels above {ROI_BODY_HU} HU (HU min/max {ct.min():.0f}/{ct.max():.0f})")
    lo, hi = inplane_extent(mask, ROI_MASK_STEP)
    return (lo + hi) / 2


def roi_start(centre: np.ndarray, size: int) -> tuple[int, int]:
    """Top-left corner of the size x size window centred on `centre` (negative when the window overruns the volume)."""
    return tuple(int(round(c - size / 2)) for c in centre)


def crop_pad_inplane(arr: np.ndarray, start: tuple[int, int], size: int, pad_value: float) -> np.ndarray:
    """size x size in-plane window at `start`, all z kept. Parts of the window outside `arr` are `pad_value`."""
    out = np.full((size, size) + arr.shape[2:], pad_value, dtype=arr.dtype)
    src = [slice(max(s, 0), max(min(s + size, n), max(s, 0))) for s, n in zip(start, arr.shape[:2])]
    out[src[0].start - start[0]:src[0].stop - start[0], src[1].start - start[1]:src[1].stop - start[1]] = arr[src[0], src[1]]
    return out


def paste_window_inplane(window: np.ndarray, start: tuple[int, int], shape: tuple[int, int]) -> np.ndarray:
    """Inverse of crop_pad_inplane for labels/predictions: the window in a zero (background) frame of in-plane `shape`."""
    out = np.zeros(tuple(shape) + window.shape[2:], dtype=window.dtype)
    size = window.shape[0]
    src = [slice(max(s, 0), max(min(s + size, n), max(s, 0))) for s, n in zip(start, shape)]
    out[src[0], src[1]] = window[src[0].start - start[0]:src[0].stop - start[0], src[1].start - start[1]:src[1].stop - start[1]]
    return out


def roi_window_size(required: float, margin: int = ROI_MARGIN, multiple: int = ROI_MULTIPLE) -> int:
    return int(np.ceil((required + 2 * margin) / multiple) * multiple)


def roi_required_size(id_: str, source_path: Path, target_spacing: tuple[float, float, float]) -> float:
    """Smallest square window (voxels of the resampled grid) around this patient's image-derived centre that contains
    every label voxel: 2 x the largest distance from the centre to a label edge, over both in-plane axes."""
    id_path: Path = source_path / "train" / id_
    ct_nib = nib.load(str(id_path / f"{id_}.nii.gz"))
    ct, gt = np.asarray(ct_nib.dataobj), np.asarray(nib.load(str(id_path / "GT.nii.gz")).dataobj)
    spacing = ct_nib.header.get_zooms()[:3]
    ct, gt = resample_image(ct, spacing, target_spacing), resample_label(gt, spacing, target_spacing)
    centre = roi_centre(ct)
    lo, hi = inplane_extent(gt > 0)
    return float(2 * np.maximum(centre - lo, hi - centre).max())


def foreground_hu(id_: str, source_path: Path,
                  target_spacing: tuple[float, float, float] | None = None) -> np.ndarray:
    """HU of the voxels with label > 0 of one training patient, on the same (resampled) grid that gets sliced."""
    id_path: Path = source_path / "train" / id_
    ct_nib = nib.load(str(id_path / f"{id_}.nii.gz"))
    ct, gt = np.asarray(ct_nib.dataobj), np.asarray(nib.load(str(id_path / "GT.nii.gz")).dataobj)
    if target_spacing is not None:
        spacing = ct_nib.header.get_zooms()[:3]
        ct, gt = resample_image(ct, spacing, target_spacing), resample_label(gt, spacing, target_spacing)
    return ct[gt > 0].astype(np.float32)


def slice_patient(id_: str, dest_path: Path, source_path: Path, shape: tuple[int, int],
                  test_mode: bool = False, gt_version: str | None = None,
                  target_spacing: tuple[float, float, float] | None = None,
                  norm_stats: dict[str, float] | None = None,
                  crop_size: int | None = None) -> tuple[float, float, float]:
    id_path: Path = source_path / ("train" if not test_mode else "test") / id_

    ct_path: Path = (id_path / f"{id_}.nii.gz") if not test_mode else (source_path / "test" / f"{id_}.nii.gz")
    nib_obj = nib.load(str(ct_path))
    ct: np.ndarray = np.asarray(nib_obj.dataobj)
    # dx, dy, dz = nib_obj.header.get_zooms()
    x, y, z = ct.shape
    dx, dy, dz = nib_obj.header.get_zooms()

    assert sanity_ct(ct, *ct.shape, *nib_obj.header.get_zooms())

    gt: np.ndarray
    if not test_mode:
        gt_path: Path = id_path / "GT.nii.gz"
        gt_nib = nib.load(str(gt_path))
        # print(nib_obj.affine, gt_nib.affine)
        gt = np.asarray(gt_nib.dataobj)
        assert sanity_gt(gt, ct)
        if gt_version:
            assert set(np.unique(gt)) == EXPECTED_LABELS[gt_version], (id_, gt_version, np.unique(gt))
    else:
        gt = np.zeros_like(ct, dtype=np.uint8)

    if target_spacing is not None:
        in_shape, spacing = ct.shape, (dx, dy, dz)
        hu_before = (ct.min(), ct.max())
        ct = resample_image(ct, spacing, target_spacing)
        gt = resample_label(gt, spacing, target_spacing)
        assert ct.shape == gt.shape, (id_, ct.shape, gt.shape)
        z = ct.shape[2]
        new_affine = resampled_affine(nib_obj.affine, in_shape, ct.shape)
        print(f"{id_}: spacing {np.round(spacing, 3).tolist()} -> "
              f"{np.round(np.abs(new_affine[:3, :3]).sum(0), 3).tolist()} mm, shape {in_shape} -> {ct.shape}, "
              f"HU min/max {hu_before[0]}/{hu_before[1]} -> {ct.min():.0f}/{ct.max():.0f}")
        if not test_mode:  # NIfTI copy of the source `train/<id>/` layout, for geometry checks
            out_dir = dest_path.parent / "resampled" / "train" / id_
            out_dir.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(ct, new_affine), str(out_dir / f"{id_}.nii.gz"))
            nib.save(nib.Nifti1Image(gt, new_affine), str(out_dir / "GT.nii.gz"))

    if crop_size:  # in-plane ROI window on the resampled grid; `shape` is (crop_size, crop_size), so the slice resize below is a no-op
        assert target_spacing is not None, "crop_size needs the resampled grid (--resample median)"
        assert tuple(shape) == (crop_size, crop_size), (shape, crop_size)
        centre = roi_centre(ct)  # image only: gt is never looked at to place the window
        start = roi_start(centre, crop_size)
        # Crop/pad BEFORE the intensity normalisation, padding the image with air: the padded voxels then get exactly
        # the grey value real air gets (the window's low end, 0 in the PNG), instead of a value made up after scaling.
        air = norm_stats["lo"] if norm_stats is not None else float(ct.min())
        resampled_shape, fg_before = ct.shape, int((gt > 0).sum())
        ct, gt = crop_pad_inplane(ct, start, crop_size, air), crop_pad_inplane(gt, start, crop_size, 0)
        assert ct.shape == gt.shape == (crop_size, crop_size, z), (id_, ct.shape, gt.shape)
        retained = int((gt > 0).sum()) / fg_before if fg_before else None
        print(f"{id_}: ROI window {crop_size}x{crop_size} at {start} of {resampled_shape[:2]}, "
              f"label voxels retained {'n/a' if retained is None else f'{100 * retained:.4f}%'}")
        crop_dir = dest_path.parent / "roi_crop"  # read back by src/evaluate.py to paste predictions into the full frame
        crop_dir.mkdir(parents=True, exist_ok=True)
        (crop_dir / f"{id_}.json").write_text(json.dumps(
            {"start": list(start), "size": crop_size, "resampled_shape": list(resampled_shape), "retained": retained}))

    # Intensity: per-volume min-max by default; with norm_stats the fixed train-set HU window (same for every patient)
    norm_ct: np.ndarray = norm_arr(ct) if norm_stats is None else window_arr(ct, norm_stats["lo"], norm_stats["hi"])

    to_slice_ct = norm_ct
    to_slice_gt = gt

    for idz in range(z):
        img_slice = resize_(to_slice_ct[:, :, idz], shape).astype(np.uint8)
        gt_slice = resize_(to_slice_gt[:, :, idz], shape, order=0).astype(np.uint8)
        assert img_slice.shape == gt_slice.shape
        gt_slice *= 63
        assert gt_slice.dtype == np.uint8, gt_slice.dtype
        # assert set(np.unique(gt_slice)) <= set(range(5))
        assert set(np.unique(gt_slice)) <= set([0, 63, 126, 189, 252]), np.unique(gt_slice)

        arrays: list[np.ndarray] = [img_slice, gt_slice]

        subfolders: list[str] = ["img", "gt"]
        assert len(arrays) == len(subfolders)
        for save_subfolder, data in zip(subfolders,
                                        arrays):
            filename = f"{id_}_{idz:04d}.png"

            save_path: Path = Path(dest_path, save_subfolder)
            save_path.mkdir(parents=True, exist_ok=True)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                imsave(str(save_path / filename), data)

    return dx, dy, dz


def get_splits(src_path: Path, retains: int, fold: int) -> tuple[list[str], list[str], list[str]]:
    ids: list[str] = sorted(map_(lambda p: p.name, (src_path / 'train').glob('Patient_*')))
    print(f"Founds {len(ids)} in the id list")
    print(ids[:10])
    assert len(ids) > retains

    random.shuffle(ids)  # Shuffle before to avoid any problem if the patients are sorted in any way
    validation_slice = slice(fold * retains, (fold + 1) * retains)
    validation_ids: list[str] = ids[validation_slice]
    assert len(validation_ids) == retains

    training_ids: list[str] = [e for e in ids if e not in validation_ids]
    assert (len(training_ids) + len(validation_ids)) == len(ids)

    test_ids: list[str] = sorted(map_(lambda p: Path(p.stem).stem, (src_path / 'test').glob('*')))
    print(f"Founds {len(test_ids)} test ids")
    print(test_ids[:10])

    return training_ids, validation_ids, test_ids


def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    # Assume the clean up is done before calling the script
    assert src_path.exists()
    assert not dest_path.exists()

    training_ids: list[str]
    validation_ids: list[str]
    test_ids: list[str]
    training_ids, validation_ids, test_ids = get_splits(src_path, args.retains, args.fold)

    target_spacing: tuple[float, float, float] | None = None
    if args.resample == "median":  # training patients only, then the same target for train and val
        train_spacings = [nib.load(str(src_path / "train" / i / f"{i}.nii.gz")).header.get_zooms()[:3]
                          for i in training_ids]
        target_spacing = median_target_spacing(train_spacings)
        print(f"Resampling every volume to target spacing {np.round(target_spacing, 4).tolist()} mm "
              f"(median of {len(training_ids)} training patients)")

    norm_stats: dict[str, float] | None = None
    if args.normalize:  # stats from the training patients only (after resampling), reused unchanged for val/test
        pooled = np.concatenate(Pool(args.process if args.process > 0 else None).starmap(
            foreground_hu, [(i, src_path, target_spacing) for i in training_ids]))
        norm_stats = ct_norm_stats(pooled)
        print(f"CT normalization ({args.normalize}) from {pooled.size} foreground voxels of {len(training_ids)} "
              f"training patients: " + ", ".join(f"{k}={v:.3f}" for k, v in norm_stats.items()))
        if not (abs(norm_stats["lo"] - EXPECTED_WINDOW[0]) < 100 and abs(norm_stats["hi"] - EXPECTED_WINDOW[1]) < 100):
            print(f"WARNING: lo/hi far from the expected ~{EXPECTED_WINDOW}, check the data version")
        dest_path.mkdir(parents=True, exist_ok=True)
        (dest_path / NORM_STATS_FILE).write_text(json.dumps(
            norm_stats | {"percentiles": [0.5, 99.5], "normalize": args.normalize, "n_foreground_voxels": int(pooled.size),
                          "train_patients": training_ids, "target_spacing": target_spacing}, indent=2))

    crop_size: int | None = None
    if args.crop:  # T from the training patients' labels only, the same T for train and val
        required = Pool(args.process if args.process > 0 else None).starmap(
            roi_required_size, [(i, src_path, target_spacing) for i in training_ids])
        crop_size = roi_window_size(max(required))
        print(f"ROI crop: train patients need a window of up to {max(required):.1f} px "
              f"({training_ids[int(np.argmax(required))]}); + {ROI_MARGIN} px margin per side, rounded up to a multiple "
              f"of {ROI_MULTIPLE} -> T = {crop_size} ({crop_size * target_spacing[0]:.0f} mm); slice shape is {crop_size}x{crop_size}")
        dest_path.mkdir(parents=True, exist_ok=True)
        (dest_path / "roi_crop.json").write_text(json.dumps(
            {"size": crop_size, "margin": ROI_MARGIN, "multiple": ROI_MULTIPLE, "centre": "body bounding box (image only)",
             "required_train": dict(zip(training_ids, required)), "target_spacing": target_spacing}, indent=2))

    resolution_dict: dict[str, tuple[float, float, float]] = {}

    split_ids: list[str]
    for mode, split_ids in zip(["train", "val"], [training_ids, validation_ids]):
        dest_mode: Path = dest_path / mode
        print(f"Slicing {len(split_ids)} pairs to {dest_mode}")

        pfun: Callable = partial(slice_patient,
                                 dest_path=dest_mode,
                                 source_path=src_path,
                                 shape=(crop_size, crop_size) if crop_size else tuple(args.shape),
                                 test_mode=mode == 'test',
                                 gt_version=args.gt_version,
                                 target_spacing=target_spacing,
                                 norm_stats=norm_stats,
                                 crop_size=crop_size)
        resolutions: list[tuple[float, float, float]]
        iterator = tqdm_(split_ids)
        match args.process:
            case 1:
                resolutions = list(map(pfun, iterator))
            case -1:
                resolutions = Pool().map(pfun, iterator)
            case _ as p:
                resolutions = Pool(p).map(pfun, iterator)

        for key, val in zip(split_ids, resolutions):
            resolution_dict[key] = val

    if crop_size:  # train patients were sized to fit (a miss is a bug); val is the out-of-sample check, only reported
        retained = {i: json.loads((dest_path / "roi_crop" / f"{i}.json").read_text())["retained"]
                    for i in training_ids + validation_ids}
        assert all(retained[i] == 1.0 for i in training_ids), {i: retained[i] for i in training_ids if retained[i] != 1.0}
        for i in validation_ids:
            if retained[i] != 1.0:
                print(f"WARNING: ROI window clips {i}: only {100 * retained[i]:.4f}% of its label voxels are retained")
        print(f"ROI crop retention: train {min(retained[i] for i in training_ids):.4%} (min), "
              f"val {min(retained[i] for i in validation_ids):.4%} (min)")

    with open(dest_path / "spacing.pkl", 'wb') as f:
        pickle.dump(resolution_dict, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved spacing dictionnary to {f}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slicing parameters')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)

    parser.add_argument('--shape', type=int, nargs="+", default=[256, 256])
    parser.add_argument('--retains', type=int, default=25, help="Number of retained patient for the validation data")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--resample', choices=['median'], default=None,
                        help="resample every volume to the median training spacing before slicing (default: off)")
    parser.add_argument('--normalize', choices=list(NORMALIZE_MODES), default=None,
                        help="clip to the training foreground 0.5-99.5 percentile HU window and rescale to 0..255 "
                             "instead of per-volume min-max (default: off); _zscore also z-scores at load time")
    parser.add_argument('--crop', choices=['roi'], default=None,
                        help="in-plane crop/pad of every resampled volume to one TxT window centred on the body "
                             "(image-derived), T from the training labels; replaces the resize to --shape (needs --resample)")
    parser.add_argument('--gt_version', choices=list(EXPECTED_LABELS), default=None,
                        help="SEGTHOR version of the GT; asserts every patient has exactly its labels (default: no check)")
    parser.add_argument('--process', '-p', type=int, default=1,
                        help="The number of cores to use for processing")
    args = parser.parse_args()
    if args.crop and not args.resample:
        parser.error("--crop needs --resample median: the window is a size in voxels of the common grid")
    random.seed(args.seed)

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
