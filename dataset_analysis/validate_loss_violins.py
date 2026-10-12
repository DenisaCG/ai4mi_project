"""Audit the violin figure's numbers: patient counts, key uniqueness and agreement with the per-run tables of the Snellius accounts.

    python dataset_analysis/validate_loss_violins.py [--metrics-dir metrics] [--runs-dir ~/snellius_inventory/loss_runs]

Checks that every variant and its reference cover the same 40 validation patients x 4 organs, that every violin holds 40
values and appears once, that the Dice and HD95 of every run (mean over patients and organs) equal loss_per_run.csv where the
Snellius inventories list the run and the run's own summary.json evaluation where it was pulled (at least one must exist), that
the model, optimiser, epochs and pre-processing of every run equal the reference's wherever config.yaml was pulled, and that
the mean change plotted over all patients and organs equals the difference of the per-run means of the variant and its reference.
Needs PyYAML.
"""

import argparse
import math
import json
import statistics
from pathlib import Path

import yaml

from loss_violins import METRICS, MODELS, ORGANS, collect, load_patient_metrics
from utils import INPUTS, N_FOLDS, REPO
from validate_loss_training_curves import inventory_runs, require, run_means


def setup(path: Path) -> dict:
    """The settings of a run that must match the reference for the losses to be compared."""
    config = yaml.safe_load(path.read_text())
    return {
        "model": config["model"],
        "optim": config["optim"],
        "scheduler": config["scheduler"],
        "epochs": config["train"]["epochs"],
        "augment": config["data"]["augment"],
        "preprocess": config["data"]["preprocess"],
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metrics-dir", type=Path, default=REPO / "metrics")
    parser.add_argument(
        "--runs-dir", type=Path, default=INPUTS / "snellius_inventory/loss_runs"
    )
    args = parser.parse_args()
    table = collect(args.metrics_dir, args.runs_dir)
    keys = [(r["variant"], r["model"], r["metric"], r["organ"]) for r in table]
    require(len(set(keys)) == len(keys), "A violin appears twice")
    require(
        len(table) == sum(len(v) for _, _, v in MODELS) * len(METRICS) * len(ORGANS),
        "Unexpected number of violins",
    )
    require(
        all(len(r["gain"]) == 40 for r in table), "A violin does not hold 40 patients"
    )
    runs = inventory_runs()
    for model, (reference, _), variants in MODELS:
        ref = load_patient_metrics(args.metrics_dir, reference)
        require(
            len(ref) == 40 * len(ORGANS), f"{reference}: not 40 patients x 4 organs"
        )
        ref_runs = [
            run_means(
                args.metrics_dir / f"{reference}_fold{f}" / "seed0" / "metrics_3d.csv"
            )
            for f in range(N_FOLDS)
        ]
        for label, experiment in variants:
            var = load_patient_metrics(args.runs_dir, experiment)
            require(
                var.keys() == ref.keys(),
                f"{experiment}: patients differ from {reference}",
            )
            var_runs = [
                run_means(
                    args.runs_dir / f"{experiment}_fold{f}" / "seed0" / "metrics_3d.csv"
                )
                for f in range(N_FOLDS)
            ]
            for fold, means in enumerate(var_runs):
                run_dir = args.runs_dir / f"{experiment}_fold{fold}" / "seed0"
                listed = runs.get((experiment, fold, 0), [])
                summary = run_dir / "summary.json"
                require(
                    listed or summary.exists(),
                    f"{experiment} fold {fold}: no independent source for the run means",
                )
                for metric, value in means.items():
                    if listed:
                        require(
                            any(
                                math.isclose(value, c[metric], abs_tol=1e-4)
                                for c in listed
                            ),
                            f"{experiment} fold {fold}: {metric} differs from loss_per_run.csv",
                        )
                    if summary.exists():
                        logged = json.loads(summary.read_text())["eval"][
                            f"val_{metric}_fg"
                        ]
                        require(
                            math.isclose(value, logged, abs_tol=1e-6),
                            f"{experiment} fold {fold}: {metric} differs from summary.json",
                        )
                if (run_dir / "config.yaml").exists():
                    require(
                        setup(run_dir / "config.yaml")
                        == setup(
                            args.metrics_dir
                            / f"{reference}_fold{fold}"
                            / "seed0"
                            / "config.yaml"
                        ),
                        f"{experiment} fold {fold}: model, optimiser, epochs or pre-processing differ from the reference",
                    )
            for metric, (sign, _, _) in METRICS.items():
                plotted = statistics.fmean(
                    float(r["gain"].mean())
                    for r in table
                    if r["variant"] == label
                    and r["model"] == model
                    and r["metric"] == metric
                )
                expected = sign * (
                    statistics.fmean(m[metric] for m in var_runs)
                    - statistics.fmean(m[metric] for m in ref_runs)
                )
                require(
                    math.isclose(plotted, expected, abs_tol=1e-9),
                    f"{experiment}: mean {metric} change differs from the run means",
                )
    print(f"violins: {len(table)} violins of 40 patients, all checks passed")


if __name__ == "__main__":
    main()
