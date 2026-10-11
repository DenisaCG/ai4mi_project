"""Audit the post-processing figures' numbers: run counts, key uniqueness, bounds and agreement with the other sources.

    python dataset_analysis/validate_postprocessing.py [--metrics-dir metrics]

Checks that the base model has 12 runs x 40 validation patients x 4 organs and that its plotted Dice equals the "all runs"
row of metrics/cv_summary.md to its three decimals; that every method appears once and has one row per fold, seed label,
patient and organ (12 runs for the single-model rows, 4 for the vote, softmax-average, seed-0 and slice-presence rows) and 40
patients; that the plotted means equal the mean over the patients behind the dots; that the base of the runs a method was
scored against equals the base model's (except for the seed-0 and slice-presence rows, which have their own); that a method
repeated in several result folders of the inventory has identical numbers; that the slide rows are a subset of the methods;
that only the two oracle rows use the ground truth; and that the metrics are in physical bounds.
"""

import argparse
import math
import statistics
from pathlib import Path

from postprocessing_figure import BASE_EXPERIMENT, PP_DIR, SLIDE, collect
from utils import CLASSES, N_FOLDS, N_SEEDS, REPO, read_csv
from validate_architecture_overview import require, summary_dice

N_PATIENTS = 40
ORACLES = {"gating_oracle:gate", "gating_oracle:gate_lcc"}
OWN_BASE = (
    "presence:",
    "seed0_crf",
)  # scored on seed 0 only, against a base of their own


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
        {m["key"] for m in data["methods"] if m["oracle"]} == ORACLES,
        "Only the two oracle rows may be marked as using the ground truth",
    )
    require(set(SLIDE) <= set(keys), "A slide row is not a method")
    for m in data["methods"]:
        name = m["key"]
        seeds = len({r["seed"] for r in m["rows"]})
        cells = [(r["fold"], r["seed"], r["patient"], r["organ"]) for r in m["rows"]]
        require(
            m["n_runs"] == N_FOLDS * seeds, f"{name}: {m['n_runs']} runs, {seeds} seeds"
        )
        require(len(set(cells)) == len(cells), f"{name}: duplicate rows")
        require(len(cells) == seeds * N_PATIENTS * len(CLASSES), f"{name}: row count")
        require(len(m["patient_dice"]) == N_PATIENTS, f"{name}: not 40 patients")
        if not any(tag in name for tag in OWN_BASE):
            require(
                math.isclose(m["own_base_dice"], base["dice"], abs_tol=1e-4)
                and math.isclose(m["own_base_hd95"], base["hd95"], abs_tol=0.05),
                f"{name}: base in the inventory differs from the metrics",
            )
        require(0 < m["dice"] < 1 and m["hd95"] > 0, f"{name}: out of bounds")
        for metric, tol in (("dice", 1e-3), ("hd95", 0.3)):
            patient_mean = statistics.fmean(m[f"patient_{metric}"].values())
            require(
                abs(patient_mean - m[metric]) < tol,
                f"{name}: {metric} {m[metric]:.4f} differs from the patient mean {patient_mean:.4f}",
            )
    repeated = {}
    for r in read_csv(args.pp_dir / "pp_summary_from_rows.csv"):
        if (
            r["organ"] == "mean"
            and "smoke" not in r["root"]
            and "combo" not in r["method"]
        ):
            repeated.setdefault(r["method"], set()).add((r["dice"], r["hd95"]))
    for variant, values in repeated.items():
        require(
            len(values) == 1, f"{variant}: result folders disagree on the same runs"
        )
    print(f"post-processing: {len(keys)} methods, all checks passed")


if __name__ == "__main__":
    main()
