"""Check (and optionally repair) processed 256x256 PNGs against their source NIfTI volumes.

Reslices every patient with slice_segthor.slice_patient into a staging folder, then compares the
staging PNGs with the processed dataset slice by slice. Without --fix nothing in --processed changes.
With --fix, mismatching patients' gt PNGs are backed up to --backup and then replaced.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
from skimage.io import imread

import slice_segthor
from slice_segthor import slice_patient


def sanity_gt_integer(gt, ct) -> bool:
    """slice_segthor requires uint8 GT; the corrected GT was saved as int16 but still holds labels 0-4."""
    assert gt.shape == ct.shape
    assert np.issubdtype(gt.dtype, np.integer), gt.dtype
    assert set(np.unique(gt)) <= {0, 1, 2, 3, 4}, np.unique(gt)
    return True


slice_segthor.sanity_gt = sanity_gt_integer


def compare(staging: Path, processed: Path, patient: str, kind: str) -> tuple[int, list[str]]:
    """Number of staged slices and the names of those that differ from (or are missing in) processed."""
    bad = []
    files = sorted((staging / kind).glob(f"{patient}_*.png"))
    for f in files:
        target = processed / kind / f.name
        if not target.exists() or not np.array_equal(imread(f), imread(target)):
            bad.append(f.name)
    return len(files), bad


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="folder holding train/Patient_XX/{CT,GT}.nii.gz")
    p.add_argument("--processed", type=Path, required=True, help="processed dataset with train/ and val/ PNG folders")
    p.add_argument("--staging", type=Path, required=True, help="scratch folder for the fresh slices (must not exist)")
    p.add_argument("--patients", nargs="*", help="default: every Patient_* under --source/train")
    p.add_argument("--fix", action="store_true")
    p.add_argument("--backup", type=Path, help="required with --fix; must not exist")
    args = p.parse_args()
    if args.fix and (args.backup is None or args.backup.exists()):
        raise SystemExit("--fix needs --backup pointing at a folder that does not exist yet")
    if args.staging.exists():
        raise SystemExit(f"Staging folder already exists: {args.staging}")

    patients = args.patients or sorted(d.name for d in (args.source / "train").glob("Patient_*") if d.is_dir())
    stale = {}
    for patient in patients:
        split = "val" if (args.processed / "val/gt" / f"{patient}_0000.png").exists() else "train"
        slice_patient(patient, args.staging, args.source, (256, 256))
        n_gt, bad_gt = compare(args.staging, args.processed / split, patient, "gt")
        n_img, bad_img = compare(args.staging, args.processed / split, patient, "img")
        print(f"{patient} ({split}): {n_gt} slices | gt mismatches {len(bad_gt)} | img mismatches {len(bad_img)}", flush=True)
        if bad_gt:
            stale[patient] = (split, bad_gt)

    print(f"Stale gt patients: {sorted(stale) or 'none'}")
    if args.fix:
        for patient, (split, names) in stale.items():
            backup = args.backup / split / "gt"
            backup.mkdir(parents=True, exist_ok=True)
            for name in names:
                target = args.processed / split / "gt" / name
                if target.exists():
                    shutil.copy2(target, backup / name)
                shutil.copy2(args.staging / "gt" / name, target)
            print(f"Replaced {len(names)} gt slices of {patient}; old ones saved in {backup}")


if __name__ == "__main__":
    main()
