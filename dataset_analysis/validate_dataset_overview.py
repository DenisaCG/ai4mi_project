"""Audit the dataset-overview and error-cases figures: run counts, key uniqueness, bounds and agreement with the other sources.

    python dataset_analysis/validate_dataset_overview.py [--metrics-dir metrics]

Checks that the Dice and HD95 of each shown case and its slice equal cases.csv; that the upgraded baseline has 12 runs x 10 validation patients = 120 values per organ and one row per organ;
that the plotted median, 5th and 95th percentile equal organ_metrics.csv of the Snellius analysis; that the mean of the
same values equals the "all runs" row of metrics/cv_summary.md (Dice and HD95); that the 40 scans, their slice-thickness counts and the pooled label
shares equal the Snellius tables (shares sum to 100 %); and that the metrics are in physical bounds.
"""

import argparse
import statistics
from pathlib import Path

from error_cases_figure import CASE_DIR, collect_cases
from dataset_overview_figure import (
    DATA_DIR,
    EXPERIMENT,
    ORGANS,
    geometry,
    label_shares,
    summarise,
)
from utils import N_FOLDS, N_SEEDS, REPO, load_cv_runs, read_csv
from validate_architecture_overview import require

PATIENTS_PER_FOLD = 10
N_PATIENTS = 40


def summary_row(path: Path, experiment: str, metric: str) -> dict[str, float]:
    """Per-organ mean (first number of each cell) of the 'all runs' row of one table of one experiment in cv_summary.md."""
    section = path.read_text().split(f"## {experiment}\n", 1)[1].split("\n## ", 1)[0]
    table = section.split(f"### {metric}", 1)[1]
    row = next(line for line in table.splitlines() if line.startswith("| all runs"))
    cells = [float(c.split("±")[0]) for c in row.strip("|").split("|")[1:]]
    return dict(zip(ORGANS, cells))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metrics-dir", type=Path, default=REPO / "metrics")
    args = parser.parse_args()
    runs = load_cv_runs(args.metrics_dir, EXPERIMENT, keys=("dice", "hd95"))
    rows = summarise(runs)

    require([r["organ"] for r in rows] == ORGANS, "organ rows")
    reference = {
        r["organ"]: r
        for r in read_csv(DATA_DIR / "organ_metrics.csv")
        if r["experiment"] == EXPERIMENT
    }
    for row in rows:
        organ = row["organ"]
        require(
            row["n"] == N_FOLDS * N_SEEDS * PATIENTS_PER_FOLD,
            f"{organ}: n = {row['n']}",
        )
        for metric in ("dice", "hd95"):
            for stat in ("median", "p5", "p95"):
                require(
                    abs(
                        row[f"{metric}_{stat}"]
                        - float(reference[organ][f"{metric}_{stat}"])
                    )
                    < 1e-9,
                    f"{organ} {metric} {stat} differs from organ_metrics.csv",
                )
        require(
            0 <= row["dice_p5"] <= row["dice_median"] <= row["dice_p95"] <= 1
            and 0 <= row["hd95_p5"] <= row["hd95_median"] <= row["hd95_p95"],
            f"{organ}: out of bounds",
        )
    for metric, column, decimals in (("Dice", "dice", 3), ("HD95 mm", "hd95", 1)):
        table = summary_row(args.metrics_dir / "cv_summary.md", EXPERIMENT, metric)
        for organ in ORGANS:
            mean = statistics.fmean(r[column] for r in runs if r["organ"] == organ)
            require(
                round(mean, decimals) == table[organ],
                f"{organ} {metric}: mean {mean:.4f} differs from cv_summary.md {table[organ]}",
            )
    scans = geometry()
    require(len(scans) == N_PATIENTS, f"{len(scans)} scans")
    counts = {
        float(r["slice_thickness_mm"]): int(r["n_patients"])
        for r in read_csv(DATA_DIR / "slice_thickness_counts.csv")
    }
    for thickness, n in counts.items():
        require(
            sum(s["thickness"] == thickness for s in scans) == n,
            f"{thickness} mm count differs",
        )
    require(
        sum(counts.values()) == N_PATIENTS, "thickness counts do not cover 40 scans"
    )
    shares = label_shares()
    require(abs(sum(shares.values()) - 100) < 1e-6, "label shares do not sum to 100 %")
    pooled = {r["name"]: r for r in read_csv(DATA_DIR / "label_shares_pooled.csv")}
    foreground = sum(shares[o] for o in ORGANS)
    for organ in ORGANS:
        require(
            abs(
                100 * shares[organ] / foreground
                - float(pooled[organ]["share_of_foreground_pct"])
            )
            < 1e-6,
            f"{organ}: foreground share differs",
        )
    reference_cases = {
        (int(r["seed"]), r["patient"], r["organ"]): r
        for r in read_csv(CASE_DIR / "cases.csv")
        if r["experiment"] == EXPERIMENT
    }
    for case in collect_cases(runs):
        ref = reference_cases[case["seed"], case["patient"], case["organ"]]
        label = f"{case['patient']} {case['organ']}"
        require(case["png"].is_file(), f"missing render {case['png']}")
        require(
            int(ref["fold"]) == case["fold"], f"{label}: fold differs from cases.csv"
        )
        require(
            case["slice"] == int(ref["slice_index_superior_inferior"]),
            f"{label}: slice",
        )
        require(
            abs(case["dice"] - float(ref["dice"])) < 1e-9,
            f"{label}: Dice differs from cases.csv",
        )
        require(
            abs(case["hd95"] - float(ref["hd95"])) < 1e-9,
            f"{label}: HD95 differs from cases.csv",
        )
    print("dataset overview and error cases: all checks passed")


if __name__ == "__main__":
    main()
