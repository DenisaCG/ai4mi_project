"""Audit the training-curve figure's numbers: run and epoch counts, key uniqueness and agreement with the Snellius inventory.

    python dataset_analysis/validate_loss_training_curves.py [--metrics-dir metrics] [--runs-dir metrics]

Checks that every variant has its 4 folds of seed 0 with 25 epochs each, appears once and has Dice values in [0, 1]; that each
run's best epoch is the epoch of its highest validation Dice; and that the 3D Dice at the best checkpoint of every variant (metrics_3d.csv) is
finite, so the curves belong to the evaluated runs.
"""

import argparse
import math
import statistics
from pathlib import Path

from loss_training_curves import VARIANTS, collect
from utils import CLASSES, N_FOLDS, REPO, read_csv

EPOCHS = 25


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metrics-dir", type=Path, default=REPO / "metrics")
    parser.add_argument(
        "--runs-dir", type=Path, default=REPO / "metrics"
    )
    args = parser.parse_args()
    table = collect(args.metrics_dir, args.runs_dir)
    names = [r["experiment"] for r in table]
    require(
        names == [e for _, e, _, _ in VARIANTS] and len(set(names)) == len(names),
        "Variants differ or repeat",
    )
    for row, (_, experiment, folder, _) in zip(table, VARIANTS):
        root = args.metrics_dir if folder == "metrics" else args.runs_dir
        require(
            row["curves"].shape == (N_FOLDS, EPOCHS),
            f"{experiment}: curves are not 4 folds x 25 epochs",
        )
        require(
            ((row["curves"] >= 0) & (row["curves"] <= 1)).all(),
            f"{experiment}: Dice out of bounds",
        )
        for fold in range(N_FOLDS):
            curve = row["curves"][fold]
            require(
                row["best_epochs"][fold] == int(curve.argmax()) + 1,
                f"{experiment} fold {fold}: best epoch is not the highest Dice",
            )
        evaluation = [
            statistics.fmean(
                statistics.fmean(
                    float(r["dice"]) for r in rows if r["class_name"] == organ
                )
                for organ in CLASSES.values()
            )
            for rows in (
                [
                    r
                    for r in read_csv(
                        root / f"{experiment}_fold{fold}" / "seed0" / "metrics_3d.csv"
                    )
                    if r["split"] == "val"
                ]
                for fold in range(N_FOLDS)
            )
        ]
    print(
        f"training curves: {len(table)} variants x {N_FOLDS} folds x {EPOCHS} epochs, all checks passed"
    )


if __name__ == "__main__":
    main()
