"""Audit the final-model comparison figure's numbers: patient and organ counts, bounds and agreement with the bundle.

    python dataset_analysis/validate_final_model_comparison.py [--data-dir deck/results_for_figures]

Checks that every model of the figure has one score per patient and organ for the 40 validation patients, the same patients
for every model; that the plotted mean Dice, HD95 and ASSD equal the OVERALL rows of summary_per_organ_and_overall.csv
(the final model: the full-pipeline row of final_pipeline_ablation/summary_overall.csv); and that the metrics are in
physical bounds.
"""

import argparse
from pathlib import Path

from final_model_comparison import (
    ABLATION,
    DATA_DIR,
    FULL_PIPELINE,
    ORGANS,
    MODEL,
    REFERENCES,
    collect,
    overall,
)
from utils import read_csv
from validate_architecture_overview import require

N_PATIENTS = 40
TOLERANCE = {"dice": 1e-4, "hd95": 1e-2, "assd": 1e-2}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    data = collect(args.data_dir)

    summary = {
        r["model"]: r
        for r in read_csv(args.data_dir / "summary_per_organ_and_overall.csv")
        if r["class_name"] == "OVERALL"
    }
    reference = {label: summary[key] for label, key in REFERENCES}
    with (args.data_dir / ABLATION / "summary_overall.csv").open() as f:
        lines = f.read().splitlines()[3:]
    pipeline = next(
        row.split(",") for row in lines if row.startswith(FULL_PIPELINE + ",")
    )
    reference[MODEL] = {"dice": pipeline[1], "hd95": pipeline[3], "assd": pipeline[5]}

    patients = {p for p, _ in data[MODEL]}
    require(len(patients) == N_PATIENTS, f"{len(patients)} patients")
    for label, scores in data.items():
        require(
            set(scores) == {(p, o) for p in patients for o in ORGANS},
            f"{label}: not one score per patient and organ",
        )
        require(
            all(
                0 <= v["dice"] <= 1 and v["hd95"] >= 0 and v["assd"] >= 0
                for v in scores.values()
            ),
            f"{label}: out of bounds",
        )
        for metric, tol in TOLERANCE.items():
            plotted, stated = overall(scores, metric), float(reference[label][metric])
            require(
                abs(plotted - stated) <= tol,
                f"{label}: {metric} {plotted:.4f} differs from the bundle's {stated:.4f}",
            )
    print(f"final-model comparison: {len(data)} models, all checks passed")


if __name__ == "__main__":
    main()
