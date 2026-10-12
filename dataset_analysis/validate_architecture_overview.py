"""Audit the architecture overview figure's numbers: run counts, key uniqueness, bounds and agreement with the other sources.

    python dataset_analysis/validate_architecture_overview.py [--metrics-dir metrics]

Checks that every architecture has all 12 runs, appears once, has a parameter count (except the frozen-encoder model) and
metrics in physical bounds; that the plotted Dice values equal the "all runs" row of metrics/cv_summary.md to its three
decimals; and that the baseline's
seed and fold spread are the values quoted on the cross-validation slide.
"""

import argparse
import math
from pathlib import Path

from architecture_overview import ARCHITECTURES, DINO, collect
from utils import N_FOLDS, N_SEEDS, REPO


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summary_dice(path: Path, experiment: str) -> tuple[float, float] | None:
    """Esophagus and mean-over-organs Dice (None if the experiment is not listed) from the 'all runs' row of the Dice table of one experiment in cv_summary.md."""
    sections = path.read_text().split(f"## {experiment}\n", 1)
    if len(sections) == 1:
        return None
    section = sections[1].split("\n## ", 1)[0]
    row = next(
        line
        for line in section.split("### Dice", 1)[1].splitlines()
        if line.startswith("| all runs")
    )
    cells = [c.split("±")[0].strip() for c in row.strip("|").split("|")[1:]]
    return float(cells[0]), float(cells[-1])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metrics-dir", type=Path, default=REPO / "metrics")
    parser.add_argument(
        "--parameters", type=Path, default=REPO / "figures/model_parameters.csv"
    )
    args = parser.parse_args()
    table = collect(args.metrics_dir, args.parameters)
    experiments = [r["experiment"] for r in table]
    require(
        experiments == [e for _, e in ARCHITECTURES],
        "Rows differ from the architecture list",
    )
    require(len(set(experiments)) == len(experiments), "Duplicate experiment")
    for row in table:
        name = row["experiment"]
        require(
            row["n_runs"] == N_FOLDS * N_SEEDS,
            f"{name}: {row['n_runs']} runs, expected 12",
        )
        require(
            (row["trainable_parameters"] is None) == (name == DINO),
            f"{name}: parameter count missing or unexpected",
        )
        require(
            0 < row["esophagus_dice"] < row["dice"] < 1, f"{name}: Dice out of bounds"
        )
        require(row["hd95"] > 0 and row["assd"] > 0, f"{name}: distance out of bounds")
        require(
            row["dice_seed_std"] > 0 and row["dice_fold_std"] > 0,
            f"{name}: missing spread",
        )
        summary = summary_dice(args.metrics_dir / "cv_summary.md", name)
        if summary is not None:
            esophagus, mean = summary
            require(
                round(row["dice"], 3) == mean,
                f"{name}: mean Dice {row['dice']:.4f} differs from cv_summary.md {mean}",
            )
            require(
                round(row["esophagus_dice"], 3) == esophagus,
                f"{name}: esophagus Dice differs from cv_summary.md",
            )
    base = table[0]
    require(
        round(base["dice_seed_std"], 3) == 0.002
        and round(base["dice_fold_std"], 3) == 0.024,
        "Baseline spreads differ from the cross-validation slide",
    )
    print(f"architecture overview: {len(table)} architectures, all checks passed")


if __name__ == "__main__":
    main()
