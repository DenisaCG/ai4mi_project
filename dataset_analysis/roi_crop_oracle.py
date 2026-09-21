"""Oracle check of the slice -> stitch round trip, per preprocessing: how well does a PERFECT model do?

For each config, the val ground-truth PNGs of its sliced dataset are stitched back onto the original CT grid exactly as
src/evaluate.py stitches predictions (crop paste-back included) and scored against the original GT volume. No model is
involved, so the 3D Dice is the ceiling that preprocessing allows:
  * for the ROI-crop dataset it must be about the ceiling of the same pipeline without the crop (else the paste-back is
    misplacing organs, or the window clips them);
  * the differences between datasets are what each preprocessing throws away (downsampling to 256 hits thin organs).

Run from the repo root, on a compute node:  python dataset_analysis/roi_crop_oracle.py
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, REPO.as_posix())  # repo-root utils.py, not dataset_analysis/utils.py
from src.config import load_config  # noqa: E402
from src.evaluate import group_by_patient, stitch  # noqa: E402
from src.run import read_json  # noqa: E402

DEFAULT_CONFIGS = [
    "configs/segthor_enet_ce_corrected_control.yaml",  # native spacing, min-max, squeezed to 256x256
    "configs/segthor_enet_ce_corrected_control_512.yaml",  # native spacing, min-max, 512x512 (no downsampling)
    "configs/segthor_enet_ce_corrected_ct_window_zscore_median_spacing.yaml",  # median + window + z-score, squeezed to 256x256
    "configs/segthor_enet_ce_corrected_ct_window_zscore_median_spacing_roi_crop.yaml",  # same + ROI crop to T x T
]


def dice(a: np.ndarray, b: np.ndarray) -> float:
    total = int(a.sum()) + int(b.sum())
    return float("nan") if total == 0 else 2 * int((a & b).sum()) / total


def oracle(config: str, log) -> tuple[str, list[dict]]:
    cfg = load_config(config)
    d = cfg["data"]
    root, p = REPO / d["root"], d.get("preprocess") or {}
    if not (root / "val" / "gt").is_dir():
        raise FileNotFoundError(f"{config}: dataset {root} is not built (run build_dataset.job first)")
    labels = {}
    for f in sorted((root / "val" / "gt").glob("*.png")):
        with Image.open(f) as im:
            labels[f.stem] = np.array(im) // d["label_scale"]  # what a perfect model would predict
    rows = []
    for patient, slices in group_by_patient(labels, d["patient_regex"]).items():
        ref = nib.load(REPO / d["source_pattern"].format(patient=patient))
        crop = read_json(root / "roi_crop" / f"{patient}.json") if p.get("crop") else None
        vol = stitch(slices, ref, patient, resampled=bool(p.get("resample")), crop=crop)
        gt = np.asarray(ref.dataobj)
        assert vol.shape == gt.shape, (patient, vol.shape, gt.shape)
        row = {"config": cfg["experiment"], "patient": patient, "slice_shape": next(iter(slices.values())).shape[0]}
        names = d["class_names"]
        row |= {"dice_" + names[k]: dice(gt == k, vol == k) for k in range(1, d["num_classes"])}
        rows.append(row)
        log(f"  {cfg['experiment'][-40:]:40s} {patient} " + " ".join(f"{names[k][:5]} {row['dice_' + names[k]]:.3f}"
                                                                    for k in range(1, d["num_classes"])))
    return cfg["experiment"], rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--out_dir", type=Path, default=REPO / "dataset_analysis/results/roi_crop_oracle")
    args = parser.parse_args()
    started, t0 = datetime.now().isoformat(timespec="seconds"), time.time()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()

    def log(msg: str) -> None:
        print(msg, flush=True)

    log(f"[{started}] roi_crop_oracle, commit {commit}, corrected SEGTHOR, val patients of each dataset, configs {args.configs}")
    rows: list[dict] = []
    for config in args.configs:
        rows += oracle(config, log)[1]

    class_cols = [c for c in rows[0] if c.startswith("dice_")]
    summary = {}
    log("\noracle 3D Dice on the original grid, mean over the val patients (a perfect model, so this is the ceiling):")
    log(f"{'dataset (slice size)':64s} " + " ".join(f"{c[5:]:>10s}" for c in class_cols) + "       mean")
    for exp in dict.fromkeys(r["config"] for r in rows):
        mine = [r for r in rows if r["config"] == exp]
        means = {c: float(np.nanmean([r[c] for r in mine])) for c in class_cols}
        summary[exp] = means | {"mean": float(np.mean(list(means.values()))), "slice_shape": mine[0]["slice_shape"]}
        log(f"{exp[24:] + ' (' + str(mine[0]['slice_shape']) + ')':64s} " + " ".join(f"{means[c]:10.4f}" for c in class_cols)
            + f" {summary[exp]['mean']:10.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "oracle_per_patient.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "run.json").write_text(json.dumps({
        "git_commit": commit, "started": started, "finished": datetime.now().isoformat(timespec="seconds"),
        "seconds": round(time.time() - t0, 1), "segthor_version": "corrected", "configs": args.configs, "summary": summary},
        indent=2))
    log(f"[{datetime.now().isoformat(timespec='seconds')}] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
