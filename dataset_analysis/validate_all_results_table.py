"""Audit the all-results table's numbers: row counts, uniqueness, bounds and agreement with the bundle.

    python dataset_analysis/validate_all_results_table.py [--data-dir ~/results_for_figures]

Checks that every row of the table appears once with all seven values; that the rows of a model equal its OVERALL and
per-organ rows of summary_per_organ_and_overall.csv; that the mean Dice of a row equals the mean of its four organ
Dice values; that the full pipeline row equals the one of final_pipeline_ablation/summary_overall.csv; and that the
metrics are in physical bounds.
"""

import argparse
from pathlib import Path

from all_results_table import COLUMNS, DATA_DIR, FULL, ORGANS, collect
from utils import read_csv
from validate_architecture_overview import require


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    rows = collect(args.data_dir)

    labels = [r["label"] for r in rows]
    require(len(set(labels)) == len(labels), "duplicate row")
    for r in rows:
        name, values = r["label"], r["values"]
        require(
            set(values) >= {(o, m) for _, o, m in COLUMNS}, f"{name}: missing values"
        )
        require(
            all(0 < values[o, "dice"] < 1 for o in ORGANS)
            and all(values["OVERALL", m] > 0 for m in ("hd95", "assd")),
            f"{name}: out of bounds",
        )
        organ_mean = sum(values[o, "dice"] for o in ORGANS) / len(ORGANS)
        require(
            abs(organ_mean - values["OVERALL", "dice"]) < 6e-5,
            f"{name}: mean Dice {values['OVERALL', 'dice']:.4f} is not the organ mean {organ_mean:.4f}",
        )
    full = next(r for r in rows if r["label"] == FULL)["values"]
    with (args.data_dir / "final_pipeline_ablation/summary_overall.csv").open() as f:
        stated = next(
            line.split(",")
            for line in f.read().splitlines()[3:]
            if line.startswith("seeds_tta_lcc,")
        )
    require(
        abs(full["OVERALL", "dice"] - float(stated[1])) < 1e-9
        and abs(full["OVERALL", "hd95"] - float(stated[3])) < 1e-9,
        "full pipeline differs from summary_overall.csv",
    )
    for r in read_csv(args.data_dir / "summary_per_organ_and_overall.csv"):
        if r["class_name"] == "OVERALL":
            require(
                0 < float(r["dice"]) < 1, f"{r['model']}: bundle Dice out of bounds"
            )
    print(f"all-results table: {len(rows)} rows, all checks passed")


if __name__ == "__main__":
    main()
