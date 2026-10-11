"""Audit the augmentation figure: run counts, key uniqueness, bounds and agreement with the other sources.

    python dataset_analysis/validate_augmentation.py [--metrics-dir metrics]

Checks that every experiment in per_run.csv has one row per (fold, seed) with metrics in physical bounds (12 runs, or 4 runs of
seed 0 for the single-transform screening); that per_arm.csv holds the same means; that the paired changes equal those of
paired_deltas.csv; that the two no-augmentation controls equal the "all runs" row of metrics/cv_summary.md to its three
decimals and the runs read from metrics/ to numerical precision; that a change published against an intermediate arm (elastic, 50 epochs)
plus that arm's own change equals the plotted one; and that the plotted csv holds the computed changes.
"""

import argparse
import math
from pathlib import Path

from augmentation_results import (
    ENET,
    POINTS,
    RESENC,
    RUNS,
    load_runs,
    paired_change,
    points,
)
from utils import N_FOLDS, N_SEEDS, REPO, load_arch_runs, read_csv, summarise
from validate_architecture_overview import require, summary_dice

DOCS = REPO / "docs/augmentation_results"


def mean(runs: dict, key: str) -> float:
    return sum(r[key] for r in runs.values()) / len(runs)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metrics-dir", type=Path, default=REPO / "metrics")
    args = parser.parse_args()
    runs = load_runs(RUNS)

    for name, arm in runs.items():
        single = len(arm) == N_FOLDS
        expected = (
            {(f, 0) for f in range(N_FOLDS)}
            if single
            else {(f, s) for f in range(N_FOLDS) for s in range(N_SEEDS)}
        )
        require(
            set(arm) == expected or len(arm) == 4 and {s for _, s in arm} == {0},
            f"{name}: runs are not the expected folds and seeds",
        )
        for r in arm.values():
            require(
                0 < r["dice"] < 1 and r["hd95"] > 0 and r["assd"] > 0,
                f"{name}: metric out of bounds",
            )

    for r in read_csv(DOCS / "per_arm.csv"):
        require(
            int(r["n_runs"]) == len(runs[r["experiment"]]),
            f"{r['experiment']}: run count differs from per_run.csv",
        )
        for key, column in (
            ("dice", "dice_fg_mean"),
            ("hd95", "hd95_fg_mean"),
            ("assd", "assd_fg_mean"),
        ):
            require(
                math.isclose(
                    mean(runs[r["experiment"]], key), float(r[column]), abs_tol=5e-6
                ),
                f"{r['experiment']}: {key} differs from per_arm.csv",
            )

    published = {
        (r["arm"], r["reference"]): r for r in read_csv(DOCS / "paired_deltas.csv")
    }
    for _, experiment, reference, _, _ in POINTS:
        n, dice, hd95 = paired_change(runs[experiment], runs[reference])
        if (experiment, reference) in published:
            rows = [(published[experiment, reference], 0.0, 0.0)]
        else:  # published against an intermediate arm: the changes add up over the same runs
            rows = [
                (
                    r,
                    paired_change(runs[mid], runs[reference])[1],
                    paired_change(runs[mid], runs[reference])[2],
                )
                for (arm, mid), r in published.items()
                if arm == experiment
            ]
        require(rows, f"{experiment}: no published comparison")
        row, extra_dice, extra_hd95 = rows[0]
        require(n == int(row["n_pairs"]), f"{experiment}: pairs differ")
        require(
            abs(dice - (float(row["dice_fg_delta_mean"]) + extra_dice)) < 5e-6,
            f"{experiment}: Dice change differs from paired_deltas.csv",
        )
        require(
            abs(-hd95 - (float(row["hd95_fg_delta_mean"]) - extra_hd95)) < 5e-5,
            f"{experiment}: HD95 change differs from paired_deltas.csv",
        )

    for name in (ENET, RESENC):
        local, _ = load_arch_runs(args.metrics_dir, name)
        summary = summarise(local)
        require(summary["n_runs"] == N_FOLDS * N_SEEDS, f"{name}: not 12 local runs")
        require(
            math.isclose(summary["dice"], mean(runs[name], "dice"), abs_tol=5e-6),
            f"{name}: per_run.csv differs from the local runs",
        )
        require(
            round(summary["dice"], 3)
            == summary_dice(args.metrics_dir / "cv_summary.md", name)[1],
            f"{name}: Dice differs from cv_summary.md",
        )

    plotted = {
        r["point"]: r for r in read_csv(REPO / "figures/augmentation_results.csv")
    }
    require(len(plotted) == len(POINTS), "plotted point count")
    for p in points(runs):
        row = plotted[p["label"].strip()]
        require(
            abs(float(row["dice_change"]) - p["dice_change"]) < 5e-5,
            f"{p['label']}: plotted Dice change differs",
        )
        require(
            abs(float(row["hd95_reduction"]) - p["hd95_reduction"]) < 5e-4,
            f"{p['label']}: plotted HD95 change differs",
        )
    print("augmentation figures: all checks passed")


if __name__ == "__main__":
    main()
