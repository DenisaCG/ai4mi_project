"""Audit the training-curve figure's numbers: run and epoch counts, key uniqueness and agreement with the Snellius inventories.

    python dataset_analysis/validate_loss_training_curves.py [--metrics-dir metrics] [--runs-dir deck/snellius_inventory/loss_runs]

Checks that every variant has seed 0 of all 4 folds with 25 epochs each, appears once and has Dice values in [0, 1]; that each
run's best epoch is the epoch of its highest validation Dice; that the best Dice of every run equals loss_curves_summary.csv
where the inventory lists it; and that the 3D Dice and HD95 of every run (metrics_3d.csv, mean over the patients and the four
organs) equal the per-run tables (loss_per_run.csv) of the two Snellius accounts (one is rounded to 4 decimals).
"""

import argparse
import math
import statistics
from pathlib import Path

from loss_training_curves import PANELS, VARIANTS, collect
from utils import CLASSES, N_FOLDS, REPO, read_csv

EPOCHS = 25
INVENTORY = REPO / "deck/snellius_inventory"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inventory_runs() -> dict[tuple[str, int, int], list[dict[str, float]]]:
    """Per-run 3D means of both accounts' loss_per_run.csv, keyed by (experiment, fold, seed).

    One account lists a mean over the organs per run, the other one row per organ; both give the mean over the four organs.
    A key can hold two values: Tversky fold 0 and 1 of seed 0 were trained in both accounts and differ.
    """
    runs, organs = {}, {}
    for path in INVENTORY.glob("*_loss/loss_per_run.csv"):
        for r in read_csv(path):
            if r["fold"] == "":
                continue
            key = (r["experiment"], int(float(r["fold"])), int(r["seed"]))
            if "dice_mean4" in r:
                runs.setdefault(key, []).append(
                    {"dice": float(r["dice_mean4"]), "hd95": float(r["hd95_mean4"])}
                )
            else:
                organs.setdefault(key, []).append(r)
    for key, rows in organs.items():
        require(
            len(rows) == len(CLASSES), f"{key}: not 4 organ rows in loss_per_run.csv"
        )
        runs.setdefault(key, []).append(
            {m: statistics.fmean(float(r[m]) for r in rows) for m in ("dice", "hd95")}
        )
    return runs


def logged_best() -> dict[tuple[str, int, int], float]:
    """Best validation Dice of every run listed in loss_curves_summary.csv of the inventories."""
    best = {}
    for path in INVENTORY.glob("*_loss/loss_curves_summary.csv"):
        for r in read_csv(path):
            if r["fold"] != "":
                best.setdefault(
                    (r["experiment"], int(float(r["fold"])), int(r["seed"])),
                    float(r["best_val_dice_fg"]),
                )
    return best


def run_means(path: Path) -> dict[str, float]:
    """3D Dice and HD95 of one run: patient mean per organ, then the mean over the organs."""
    rows = [r for r in read_csv(path) if r["split"] == "val"]
    require(len(rows) == 10 * len(CLASSES), f"{path}: not 10 patients x 4 organs")
    return {
        key: statistics.fmean(
            statistics.fmean(float(r[key]) for r in rows if r["class_name"] == organ)
            for organ in CLASSES.values()
        )
        for key in ("dice", "hd95")
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metrics-dir", type=Path, default=REPO / "metrics")
    parser.add_argument("--runs-dir", type=Path, default=INVENTORY / "loss_runs")
    args = parser.parse_args()
    table = collect(args.metrics_dir, args.runs_dir)
    names = [r["experiment"] for r in table]
    require(
        names == [e for _, e, _, _ in VARIANTS] and len(set(names)) == len(names),
        "Variants differ or repeat",
    )
    runs, best = inventory_runs(), logged_best()
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
            if (experiment, fold, 0) in best:
                require(
                    math.isclose(curve.max(), best[experiment, fold, 0], rel_tol=1e-9),
                    f"{experiment} fold {fold}: best Dice differs from loss_curves_summary.csv",
                )
            means = run_means(
                root / f"{experiment}_fold{fold}" / "seed0" / "metrics_3d.csv"
            )
            require(
                (experiment, fold, 0) in runs,
                f"{experiment} fold {fold}: not in loss_per_run.csv",
            )
            for key, value in means.items():
                require(
                    any(
                        math.isclose(value, c[key], abs_tol=1e-4)
                        for c in runs[experiment, fold, 0]
                    ),
                    f"{experiment} fold {fold}: {key} differs from loss_per_run.csv",
                )
    print(
        f"training curves: {len(table)} variants in {len(PANELS)} panels x {N_FOLDS} folds x {EPOCHS} epochs, all checks passed"
    )


if __name__ == "__main__":
    main()
