"""Held-out score of the full prediction pipeline (seed average, test-time augmentation, largest component) on one CV fold.

    python -m src.eval_ensemble --experiment full_cv4_resenc_ds_unet_dice_ce_strong_geometric_50ep --fold 0
    python -m src.eval_ensemble --experiment ... --aggregate      # after all folds: pooled tables

For fold k it loads the seeds' best checkpoints (the same ones the cross-validation was scored with), predicts the fold's
validation patients and scores, in 3D on the original CT grid against the original ground truth:

  single              one run on its own (each seed separately), nothing added: the usual cross-validation number
  single_tta          one run + geometric test-time augmentation (src.eval_tta, variant "all")
  single_lcc          one run + largest component of heart and trachea
  seeds               softmax average of the seeds
  seeds_lcc           seed average + largest component of heart and trachea
  seeds_tta           seed average of the TTA predictions
  seeds_tta_lcc       the full pipeline: seed average of the TTA predictions + largest component (what predict_test --tta writes)

Writes <out>/fold<k>.csv (variant, seed, patient, class, dice, hd95, assd). The `single` rows must reproduce each run's own
eval/metrics_3d.csv; the script stops if they do not.
"""

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from src.config import REPO
from src.eval_tta import VARIANTS, average_probs, fill_value
from src.evaluate import stitch
from src.metrics_3d import volume_metrics
from src.predict_test import largest_component, load_net, slices_by_patient
from src.run import read_json, resolve_device

VARIANT_ORDER = ["single", "single_tta", "single_lcc", "seeds", "seeds_lcc", "seeds_tta", "seeds_tta_lcc"]


@torch.no_grad()
def patient_probs(net, images: torch.Tensor, fill: float, device, tta: bool, batch: int = 16) -> np.ndarray:
    names = VARIANTS["all"] if tta else []
    return torch.cat([average_probs(net, x.to(device), names, fill) for x in images.split(batch)]).cpu().numpy()


def score(vol: np.ndarray, gt: np.ndarray, spacing, variant: str, seed, patient: str, names: list[str]) -> list[dict]:
    rows = []
    for k in range(1, len(names)):
        rows.append({"variant": variant, "seed": seed, "patient": patient, "class_idx": k, "class_name": names[k],
                     **volume_metrics(gt == k, vol == k, spacing)})
    return rows


def run_fold(experiment: str, fold: int, seeds: list[int], runs_dir: Path, out: Path, device: torch.device) -> None:
    run_dirs = [runs_dir / f"{experiment}_fold{fold}" / f"seed{s}" for s in seeds]
    loaded = [load_net(r, device) for r in run_dirs]
    cfg = loaded[0][0]
    cfg = cfg | {"train": cfg["train"] | {"debug_samples": 0}}
    d, pre = cfg["data"], cfg["data"].get("preprocess") or {}
    root, fill, names = REPO / d["root"], fill_value(cfg), d["class_names"]
    resampled = bool(pre.get("resample"))
    data = slices_by_patient(cfg, 16, split="val")
    rows = []
    for patient, (stems, images) in data.items():
        ref = nib.load(REPO / d["source_pattern"].format(patient=patient))
        spacing, gt = tuple(float(s) for s in ref.header.get_zooms()[:3]), np.asarray(ref.dataobj)
        crop = read_json(root / "roi_crop" / f"{patient}.json") if pre.get("crop") else None

        def volume(probs: np.ndarray) -> np.ndarray:
            return stitch(dict(enumerate(probs.argmax(axis=1).astype(np.uint8))), ref, patient, resampled, crop)

        plain, tta = [], []
        for seed, (_, net) in zip(seeds, loaded):
            plain.append(patient_probs(net, images, fill, device, tta=False))
            tta.append(patient_probs(net, images, fill, device, tta=True))
            for variant, probs, lcc in (("single", plain[-1], False), ("single_tta", tta[-1], False),
                                        ("single_lcc", plain[-1], True)):
                vol = volume(probs)
                rows += score(largest_component(vol) if lcc else vol, gt, spacing, variant, seed, patient, names)
        for variant, group, lcc in (("seeds", plain, False), ("seeds_lcc", plain, True),
                                    ("seeds_tta", tta, False), ("seeds_tta_lcc", tta, True)):
            vol = volume(sum(group) / len(group))
            rows += score(largest_component(vol) if lcc else vol, gt, spacing, variant, -1, patient, names)
        print(f"fold {fold} {patient}: {len(stems)} slices, {len(seeds)} seeds scored", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    with (out / f"fold{fold}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # the plain single-run rows must reproduce the runs' own evaluation
    mine = {(r["seed"], r["patient"], r["class_idx"]): r["dice"] for r in rows if r["variant"] == "single"}
    worst = 0.0
    for seed, run in zip(seeds, run_dirs):
        for r in csv.DictReader((run / "eval" / "metrics_3d.csv").open()):
            if r["split"] == "val":
                worst = max(worst, abs(mine[(seed, r["patient"], int(r["class_idx"]))] - float(r["dice"])))
    print(f"single-run rows vs each run's eval/metrics_3d.csv: max |dice difference| = {worst:.2e}")
    if worst > 1e-3:
        raise SystemExit("the single-run variant does not reproduce the runs' own evaluation")


def aggregate(out: Path) -> None:
    files = sorted(out.glob("fold*.csv"))
    d = pd.concat([pd.read_csv(f).assign(fold=int(f.stem.removeprefix("fold"))) for f in files])
    # one value per patient: the three seeds of a single-run variant are averaged first
    per_patient = (d.groupby(["variant", "fold", "patient", "class_name"])[["dice", "hd95", "assd"]].mean()
                   .groupby(["variant", "fold", "patient"]).mean())
    n_patients = per_patient.loc["single"].shape[0]
    overall = per_patient.groupby("variant").agg(["mean", "std"]).loc[VARIANT_ORDER]
    per_fold = per_patient.groupby(["variant", "fold"]).mean()["dice"].unstack("fold").loc[VARIANT_ORDER]
    organ = d.groupby(["variant", "class_name"])[["dice", "hd95", "assd"]].mean().unstack("class_name").loc[VARIANT_ORDER]
    print(f"{len(files)} folds, {n_patients} patients (std over patients)\n")
    print("OVERALL (mean over the 4 organs, then over patients)")
    print(overall.round(4).to_string(), "\n")
    print("PER ORGAN")
    print(organ.round(3).to_string(), "\n")
    print("DICE PER FOLD")
    print(per_fold.round(4).to_string(), "\n")
    base = per_patient.loc["single"]
    print("PAIRED vs single run (same patients): mean difference, patients that improved")
    for v in VARIANT_ORDER[1:]:
        diff = per_patient.loc[v] - base
        print(f"  {v:14s} dice {diff.dice.mean():+.4f} (better for {(diff.dice > 0).sum()}/{len(diff)})"
              f"  hd95 {diff.hd95.mean():+.2f}  assd {diff.assd.mean():+.3f}")
    out.mkdir(parents=True, exist_ok=True)
    overall.round(5).to_csv(out / "summary_overall.csv")
    organ.round(5).to_csv(out / "summary_per_organ.csv")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True)
    p.add_argument("--fold", type=int)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    p.add_argument("--out", type=Path)
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = p.parse_args(argv)
    out = args.out or args.runs_dir.resolve().parent / "ensemble_eval" / args.experiment
    if args.aggregate:
        return aggregate(out)
    if args.fold is None:
        p.error("--fold is required unless --aggregate")
    run_fold(args.experiment, args.fold, args.seeds, args.runs_dir, out, resolve_device(args.device))


if __name__ == "__main__":
    main()
