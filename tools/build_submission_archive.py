"""Validation part of the submission archive: the post-processed TTA predictions of the final recipe, out of fold.

    python -m tools.build_submission_archive --experiment full_cv4_resenc_ds_unet_dice_ce_strong_geometric_50ep \
        --probs /scratch-shared/<user>/ai4mi_project/postprocess_probs_tta/<experiment>/all --out group-23 --workers 8

For every validation patient of the 4 folds, the TTA-averaged softmax of the 3 seeds (src.eval_tta --save-probs, variant
`all`) is averaged, argmaxed, stitched onto the source CT grid and reduced to the largest component of heart and trachea,
exactly as tools.prob_postprocess score does for the variant avg_lcc_heart_trachea. Writes
  <out>/val/pred/<patient>.nii.gz   that label volume
  <out>/val/gt/<patient>.nii.gz     the source GT (corrected labels)
  <out>/val/{dice,hd95,assd}.npz    patient -> (K,) array, background NaN, scored on exactly the saved volumes
With --reference (rows.csv of pp_tta_ctrl) the scores are compared with that file, and the archive is checked at the end:
same patients in pred, gt and every npz, labels in 0..K-1, shape and affine equal to the GT's.
"""

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import nibabel as nib
import numpy as np

from src.config import REPO, read_yaml
from src.evaluate import score_patient
from src.metrics_3d import METRICS
from tools.postprocess_eval import largest_component, run_dirs
from tools.postprocess_filters import take_organs
from tools.prob_postprocess import patients_of, read_rows, stitch_labels


def build_patient(task: tuple, probs_dir: Path, out: Path) -> tuple[str, dict]:
    """Writes the prediction and GT of one patient and returns its metric arrays (src.evaluate.score_patient)."""
    fold_runs, pattern, patient = task
    cfg = read_yaml((Path(fold_runs[0]) / "config.yaml").read_text())
    names = cfg["data"]["class_names"]
    gt_path = REPO / pattern.format(patient=patient)
    ref = nib.load(gt_path)
    rels = [Path(r).relative_to(REPO / "runs") for r in fold_runs]
    probs = np.mean([np.load(probs_dir / rel / f"{patient}.npy").astype(np.float32) for rel in rels], axis=0)
    vol = stitch_labels(probs.argmax(axis=1).astype(np.uint8), ref, patient, cfg)
    organs = [names.index("heart"), names.index("trachea")]
    vol = take_organs(vol, largest_component(vol, organs), organs)

    nib.save(nib.Nifti1Image(vol, ref.affine, ref.header), out / "val" / "pred" / f"{patient}.nii.gz")
    shutil.copyfile(gt_path, out / "val" / "gt" / f"{patient}.nii.gz")
    per_metric = {m: {} for m in METRICS}
    spacing = tuple(float(s) for s in ref.header.get_zooms()[:3])
    score_patient("val", patient, np.asarray(ref.dataobj), vol, spacing, cfg, [], per_metric)
    return patient, {m: per_metric[m][patient] for m in METRICS}


def compare_with_reference(per_patient: dict, reference: Path, variant: str) -> float:
    """Largest absolute difference of any patient/organ metric to the rows of `variant` in `reference`
    (infinite where only one of the two is NaN)."""
    worst = 0.0
    for row in read_rows(reference, (variant,)):
        for m in METRICS:
            ours, theirs = per_patient[row["patient"]][m][row["class_idx"]], row[m]
            if np.isnan(ours) != np.isnan(theirs):
                return float("inf")
            if not np.isnan(ours):
                worst = max(worst, abs(ours - theirs))
    return worst


def check_archive(out: Path, num_classes: int) -> None:
    """Fails unless pred, gt and every npz hold the same patients and the volumes are consistent."""
    pred = {p.name.removesuffix(".nii.gz") for p in (out / "val" / "pred").glob("*.nii.gz")}
    gt = {p.name.removesuffix(".nii.gz") for p in (out / "val" / "gt").glob("*.nii.gz")}
    assert pred == gt, f"pred and gt differ: {sorted(pred ^ gt)}"
    for m in METRICS:
        keys = set(np.load(out / "val" / f"{m}.npz").files)
        assert keys == pred, f"{m}.npz keys differ from val/pred: {sorted(keys ^ pred)}"
        assert all(np.load(out / "val" / f"{m}.npz")[k].shape == (num_classes,) for k in keys), f"{m}.npz: wrong shape"
    for patient in sorted(pred):
        a, b = nib.load(out / "val" / "pred" / f"{patient}.nii.gz"), nib.load(out / "val" / "gt" / f"{patient}.nii.gz")
        assert a.shape == b.shape and np.allclose(a.affine, b.affine), f"{patient}: shape or affine differs from the GT"
        labels = np.unique(np.asarray(a.dataobj))
        assert labels.min() >= 0 and labels.max() < num_classes, f"{patient}: labels {labels.tolist()}"
    print(f"archive check passed: {len(pred)} patients")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--probs", type=Path, required=True, help="TTA probabilities, <probs>/<experiment>_fold<k>/seed<s>/<patient>.npy")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reference", type=Path, default=REPO / "metrics" / "postprocessing" / "pp_tta_ctrl" / "rows.csv")
    args = parser.parse_args(argv)

    dirs = run_dirs(args.experiment, args.folds, args.seeds)
    cfg = read_yaml((dirs[0] / "config.yaml").read_text())
    pattern = cfg["data"]["source_pattern"]
    for sub in ("pred", "gt"):
        (args.out / "val" / sub).mkdir(parents=True, exist_ok=True)
    tasks = []
    for fold in range(args.folds):
        fold_runs = [str(d) for d in dirs if d.parent.name.endswith(f"_fold{fold}")]
        tasks += [(fold_runs, pattern, p) for p in patients_of(Path(fold_runs[0]), args.probs)]
    with ProcessPoolExecutor(args.workers) as pool:
        per_patient = dict(pool.map(partial(build_patient, probs_dir=args.probs, out=args.out), tasks))
    for m in METRICS:
        np.savez(args.out / "val" / f"{m}.npz", **{p: v[m] for p, v in sorted(per_patient.items())})

    if args.reference.exists():
        worst = compare_with_reference(per_patient, args.reference, "avg_lcc_heart_trachea")
        print(f"largest difference to {args.reference}: {worst:.2e}")
    fg = {m: np.nanmean([np.nanmean(v[m][1:]) for v in per_patient.values()]) for m in METRICS}
    print(f"{len(per_patient)} patients; mean over patients of the organ mean: " + ", ".join(f"{m} {fg[m]:.4f}" for m in METRICS))
    check_archive(args.out, cfg["data"]["num_classes"])


if __name__ == "__main__":
    main()
