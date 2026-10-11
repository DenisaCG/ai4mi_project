"""Pool the metrics_3d_tta.csv files of one experiment: per variant overall and per organ, and paired against `none`.

    python -m src.aggregate_tta --experiment full_cv4_resenc_ds_unet_dice_ce_strong_geometric_50ep
"""

import argparse
from pathlib import Path

import pandas as pd

from src.config import REPO


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True)
    p.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    args = p.parse_args(argv)
    frames = []
    for f in sorted(args.runs_dir.glob(f"{args.experiment}_fold*/seed*/eval/metrics_3d_tta.csv")):
        d = pd.read_csv(f)
        d["run"] = f.parts[-4].removeprefix(args.experiment + "_") + "/" + f.parts[-3]
        frames.append(d)
    if not frames:
        raise SystemExit(f"no metrics_3d_tta.csv for {args.experiment}")
    d = pd.concat(frames)
    runs = d.run.nunique()
    per_run = d.groupby(["variant", "run", "class_name"])[["dice", "hd95", "assd"]].mean().groupby(["variant", "run"]).mean()
    order = ["none", "rot", "scale", "shift", "all", "flip"]
    overall = per_run.groupby("variant").agg(["mean", "std"]).loc[order]
    print(f"{args.experiment}: {runs} runs, {d.patient.nunique()} patients\n")
    print("OVERALL (mean over 4 organs, then over runs; std over runs)")
    print(overall.round(4).to_string(), "\n")
    organ = d.groupby(["variant", "class_name"])[["dice", "hd95", "assd"]].mean().unstack("class_name")
    print("PER ORGAN")
    print(organ.loc[order].round(3).to_string(), "\n")
    base = per_run.loc["none"]
    print("PAIRED vs none (same runs): mean difference, runs that improved")
    for v in order[1:]:
        diff = per_run.loc[v] - base
        print(f"  {v:6s} dice {diff.dice.mean():+.4f} (sd {diff.dice.std():.4f}, better in {(diff.dice > 0).sum()}/{runs})"
              f"  hd95 {diff.hd95.mean():+.2f}  assd {diff.assd.mean():+.3f}")


if __name__ == "__main__":
    main()
