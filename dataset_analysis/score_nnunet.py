"""Score nnU-Net validation predictions with the same 3D metrics as our own runs.

    python dataset_analysis/score_nnunet.py \
        --predictions <nnunet fold_0>/validation --output-dir dataset_analysis/results/nnunet/2d_tta

Reads one Patient_XX.nii.gz prediction per validation patient and scores it against the corrected
GT.nii.gz (src.metrics_3d: Dice, HD95, ASSD in mm on the original CT grid). Writes
<output-dir>/eval/metrics_3d.csv in the same schema as a pipeline run, so compare_before_after.py reads
it like any other run, plus <output-dir>/scoring.json (inputs, git commit, timestamps, per-organ means).
nnU-Net's own summary.json only has Dice, so HD95 and ASSD have to be computed here.
"""
import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO))  # appended, not inserted: the repo root has its own utils.py that must not shadow ours
from src.metrics_3d import METRICS, volume_metrics  # noqa: E402
from utils import CLASSES  # noqa: E402


def score_patient(prediction_path: Path, ground_truth_path: Path) -> list[dict]:
    """One row per organ for a patient; raises if the prediction is not on the ground-truth grid."""
    ref = nib.load(ground_truth_path)
    pred_nii = nib.load(prediction_path)
    if pred_nii.shape != ref.shape or not np.allclose(pred_nii.affine, ref.affine, atol=1e-3):
        raise ValueError(f"{prediction_path.name}: prediction grid (shape {pred_nii.shape}, "
                         f"{''.join(nib.aff2axcodes(pred_nii.affine))}) differs from ground truth "
                         f"(shape {ref.shape}, {''.join(nib.aff2axcodes(ref.affine))})")
    spacing = tuple(float(s) for s in ref.header.get_zooms()[:3])
    print(f"{prediction_path.stem.removesuffix('.nii')}: shape {ref.shape}, spacing "
          f"{np.round(spacing, 3).tolist()} mm, orientation {''.join(nib.aff2axcodes(ref.affine))}")
    gt, pred = np.asarray(ref.dataobj), np.asarray(pred_nii.dataobj)
    rows = []
    for k, name in CLASSES.items():
        rows.append({"split": "val", "patient": ground_truth_path.parent.name, "class_idx": k, "class_name": name,
                     "gt_voxels": int(np.count_nonzero(gt == k)), "pred_voxels": int(np.count_nonzero(pred == k)),
                     **volume_metrics(gt == k, pred == k, spacing)})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, required=True, help="folder with one Patient_XX.nii.gz per patient")
    p.add_argument("--ground-truth-dir", type=Path, default=REPO / "data/segthor_part1_corrected/train",
                   help="folder with Patient_XX/GT.nii.gz (corrected labels)")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    started = datetime.now(timezone.utc)
    prediction_paths = sorted(args.predictions.glob("Patient_*.nii.gz"))
    if not prediction_paths:
        raise FileNotFoundError(f"No Patient_*.nii.gz in {args.predictions}")
    rows = []
    for path in prediction_paths:
        patient = path.name.removesuffix(".nii.gz")
        ground_truth = args.ground_truth_dir / patient / "GT.nii.gz"
        if not ground_truth.is_file():
            raise FileNotFoundError(f"No ground truth for {patient}: {ground_truth}")
        rows += score_patient(path, ground_truth)

    (args.output_dir / "eval").mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "eval/metrics_3d.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    means = {name: {m: float(np.nanmean([r[m] for r in rows if r["class_idx"] == k])) for m in METRICS}
             for k, name in CLASSES.items()}
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (args.output_dir / "scoring.json").write_text(json.dumps({
        "predictions": str(args.predictions.resolve()), "ground_truth_dir": str(args.ground_truth_dir.resolve()),
        "git_commit": commit, "started": started.isoformat(), "finished": datetime.now(timezone.utc).isoformat(),
        "patients": [path.name.removesuffix(".nii.gz") for path in prediction_paths], "mean_per_organ": means}, indent=2))
    for name, values in means.items():
        print(f"{name:10s} " + "  ".join(f"{m} {v:.3f}" for m, v in values.items()))
    print(f"Scored {len(prediction_paths)} patients -> {args.output_dir}/eval/metrics_3d.csv")


if __name__ == "__main__":
    main()
