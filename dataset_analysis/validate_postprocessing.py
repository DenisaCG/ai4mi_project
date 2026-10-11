"""Audit the post-processing figure's numbers: run counts, key uniqueness, bounds and agreement with the other sources.

    python dataset_analysis/validate_postprocessing.py [--metrics-dir metrics]

Checks that the base model has 12 runs x 40 validation patients x 4 organs and that its plotted Dice equals the "all runs"
row of metrics/cv_summary.md to its three decimals and the base column of the inventory to numerical precision; that
every method appears once with 12 runs (4 fold-level ensembles for the votes), one row per fold, seed, patient and organ,
and 40 patients; that the plotted means equal the mean over the patients behind the dots; that a method repeated in
several result folders of the inventory has identical numbers; that only the gating oracle uses the ground truth; and that
the metrics are in physical bounds.
"""

import argparse
import math
import statistics
from pathlib import Path

from postprocessing_figure import BASE_EXPERIMENT, PP_DIR, collect
from utils import CLASSES, N_FOLDS, N_SEEDS, REPO, read_csv
from validate_architecture_overview import require, summary_dice

N_PATIENTS = 40
ORACLE = "gating_oracle:gate"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metrics-dir", type=Path, default=REPO / "metrics")
    parser.add_argument("--pp-dir", type=Path, default=PP_DIR)
    args = parser.parse_args()
    data = collect(args.metrics_dir, args.pp_dir)
    base = data["base"]

    require(base["n_runs"] == N_FOLDS * N_SEEDS, "base: not 12 runs")
    require(len(base["patient_dice"]) == N_PATIENTS, "base: not 40 patients")
    _, summary = summary_dice(args.metrics_dir / "cv_summary.md", BASE_EXPERIMENT)
    require(
        round(base["dice"], 3) == summary,
        f"base: Dice {base['dice']:.4f} differs from cv_summary.md {summary}",
    )
    keys = [m["key"] for m in data["methods"]]
    require(len(set(keys)) == len(keys), "Duplicate method")
    require(
        [m["key"] for m in data["methods"] if m["oracle"]] == [ORACLE],
        "Only the gating oracle may be marked as using the ground truth",
    )
    for m in data["methods"]:
        name = m["key"]
        ensemble = "seed_ensemble" in name
        require(
            m["n_runs"] == (N_FOLDS if ensemble else N_FOLDS * N_SEEDS),
            f"{name}: {m['n_runs']} runs",
        )
        cells = [(r["fold"], r["seed"], r["patient"], r["organ"]) for r in m["rows"]]
        seeds = 1 if ensemble else N_SEEDS
        require(len(set(cells)) == len(cells), f"{name}: duplicate rows")
        require(len(cells) == seeds * N_PATIENTS * len(CLASSES), f"{name}: row count")
        require(len(m["patient_dice"]) == N_PATIENTS, f"{name}: not 40 patients")
        require(
            math.isclose(m["base_dice"], base["dice"], abs_tol=1e-4)
            and math.isclose(m["base_hd95"], base["hd95"], abs_tol=0.05),
            f"{name}: base in the inventory differs from the metrics",
        )
        require(0 < m["dice"] < 1 and m["hd95"] > 0, f"{name}: out of bounds")
        for metric, tol in (("dice", 1e-3), ("hd95", 0.2)):
            patient_mean = statistics.fmean(m[f"patient_{metric}"].values())
            require(
                abs(patient_mean - m[metric]) < tol,
                f"{name}: {metric} {m[metric]:.4f} differs from the patient mean {patient_mean:.4f}",
            )
    inventory = {}
    for r in read_csv(args.pp_dir / "pp_summary_from_rows.csv"):
        if r["organ"] == "mean" and r["n_runs"] != "1":
            inventory.setdefault(r["method"], set()).add((r["dice"], r["hd95"]))
    for m in data["methods"]:
        variant = m["key"].split(":")[1]
        require(
            len(inventory[variant]) == 1,
            f"{variant}: result folders disagree on the same runs",
        )
    print(f"post-processing: {len(keys)} methods, all checks passed")


if __name__ == "__main__":
    main()
