"""Cross-validation summary of src.aggregate, on fake metrics/ trees with numbers known by hand.

Run (fold f, seed s, patient offset d = +0.01 for the first patient of a fold, -0.01 for the second) has
  dice = 0.5 + 0.1 * organ + 0.02 * f + 0.01 * s + d,   hd95 = 10 + organ + 2 * f + s,   assd = 2 + 0.5 * f
with organ 0 = esophagus and organ 1 = heart, so a fold's mean over the three seeds is
0.51 + 0.1 * organ + 0.02 * f with std 0.01.
"""

import contextlib
import csv
import io
import json
import math
import shutil
import statistics
import tempfile
import unittest
from pathlib import Path

from src.aggregate import main

ORGANS = ("esophagus", "heart")
FOLDS, SEEDS = range(4), range(3)


def finite_mean(values: list[float]) -> float:
    values = [v for v in values if not math.isnan(v)]
    return statistics.mean(values) if values else float("nan")


def patient_rows(fold: int, seed: int, nan_hd95: tuple[str, str] | None) -> list[dict]:
    rows = []
    for i, offset in enumerate((0.01, -0.01)):
        for k, organ in enumerate(ORGANS):
            patient = f"Patient_{2 * fold + i + 1:02d}"
            hd95 = (
                float("nan")
                if nan_hd95 == (patient, organ)
                else 10 + k + 2 * fold + seed
            )
            rows.append(
                {
                    "split": "val",
                    "patient": patient,
                    "class_idx": k + 1,
                    "class_name": organ,
                    "gt_voxels": 100,
                    "pred_voxels": 90,
                    "dice": 0.5 + 0.1 * k + 0.02 * fold + 0.01 * seed + offset,
                    "hd95": hd95,
                    "assd": 2 + 0.5 * fold,
                }
            )
    return rows


def write_run(metrics_dir: Path, experiment: str, seed: int, rows: list[dict]) -> None:
    run_dir = metrics_dir / experiment / f"seed{seed}"
    run_dir.mkdir(parents=True)
    with (run_dir / "metrics_3d.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    block = {}
    for metric in ("dice", "hd95", "assd"):
        per_organ = {
            o: finite_mean([r[metric] for r in rows if r["class_name"] == o])
            for o in ORGANS
        }
        block |= {f"val_{metric}_{o}": v for o, v in per_organ.items()}
        block[f"val_{metric}_fg"] = finite_mean(list(per_organ.values()))
    summary = {
        "experiment": experiment,
        "seed": seed,
        "run": f"seed{seed}",
        "model": "enet",
        "loss": "ce",
        "best": {"val_dice_fg": 0.5},
        "eval": block,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))


def write_cv_experiment(
    metrics_dir: Path,
    experiment: str,
    skip: tuple[int, int] | None = None,
    nan_hd95: tuple[int, int, str, str] | None = None,
) -> None:
    for fold in FOLDS:
        for seed in SEEDS:
            if (fold, seed) == skip:
                continue
            nan = nan_hd95[2:] if nan_hd95 and nan_hd95[:2] == (fold, seed) else None
            write_run(
                metrics_dir,
                f"{experiment}_fold{fold}",
                seed,
                patient_rows(fold, seed, nan),
            )


def run_aggregate(metrics_dir: Path) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        main(["--metrics-dir", str(metrics_dir)])


def table_rows(markdown: str, section: str, metric: str) -> dict[str, list[str]]:
    """The rows of the `metric` table in the `section` experiment, as {row label: cells}."""
    block = markdown.split(f"## {section}\n")[1].split("\n## ")[0]
    table = block.split(f"### {metric}\n")[1].split("\n###")[0]
    rows = [
        [c.strip() for c in line.strip("|").split("|")]
        for line in table.splitlines()
        if line.startswith("| ")
    ]
    return {row[0]: row[1:] for row in rows[1:]}


class CrossValidationSummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.metrics = Path(self.tmp.name)

    def test_complete_experiment_per_fold_overall_and_fold_std(self):
        write_cv_experiment(self.metrics, "exp")
        run_aggregate(self.metrics)
        report = (self.metrics / "cv_summary.md").read_text()
        self.assertIn("12/12 runs found: complete", report)

        dice = table_rows(report, "exp", "Dice")
        # fold 2: 0.51 + 0.04 = 0.55 for esophagus, +0.1 for heart, fg is their mean; std over the seeds is 0.01
        self.assertEqual(
            dice["fold 2 (3 runs)"], ["0.550 ± 0.010", "0.650 ± 0.010", "0.600 ± 0.010"]
        )
        # 12 runs: mean 0.54, sample std sqrt(12 / 11 * (0.0004 * 1.25 + 0.0001 * 2 / 3)) = 0.0249
        self.assertEqual(
            dice["all runs (12 runs)"],
            ["0.540 ± 0.025", "0.640 ± 0.025", "0.590 ± 0.025"],
        )
        # fold means 0.01 + 0.02 f for f in 0..3: sample std 0.02 * sqrt(5 / 3) = 0.0258
        self.assertEqual(dice["std of fold means"], ["0.026", "0.026", "0.026"])

        hd95 = table_rows(report, "exp", "HD95 mm")
        # fold 1: 10 + 2 + seed in 0..2 -> 13 ± 1; heart 14 ± 1; all runs: 14, sample std sqrt(12 / 11 * (4 * 1.25 + 2 / 3)) = 2.49
        self.assertEqual(
            hd95["fold 1 (3 runs)"], ["13.0 ± 1.0", "14.0 ± 1.0", "13.5 ± 1.0"]
        )
        self.assertEqual(
            hd95["all runs (12 runs)"], ["14.0 ± 2.5", "15.0 ± 2.5", "14.5 ± 2.5"]
        )
        self.assertEqual(
            table_rows(report, "exp", "ASSD mm")["fold 3 (3 runs)"], ["3.50 ± 0.00"] * 3
        )

    def test_pooled_csv_has_each_patient_and_organ_averaged_over_seeds(self):
        write_cv_experiment(self.metrics, "exp")
        run_aggregate(self.metrics)
        with (self.metrics / "cv_pooled_exp.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 8 * len(ORGANS))
        self.assertEqual(
            list(rows[0]),
            ["patient", "fold", "class_name", "n_seeds", "dice", "hd95", "assd"],
        )
        by_key = {(r["patient"], r["class_name"]): r for r in rows}
        # Patient_05 is the first patient of fold 2: dice 0.5 + 0.04 + 0.01 (seed mean) + 0.01 (offset)
        row = by_key["Patient_05", "esophagus"]
        self.assertEqual((row["fold"], row["n_seeds"]), ("2", "3"))
        self.assertAlmostEqual(float(row["dice"]), 0.56)
        self.assertAlmostEqual(float(row["hd95"]), 15.0)  # 10 + 4 + seed mean 1
        self.assertAlmostEqual(
            float(by_key["Patient_06", "heart"]["dice"]), 0.64
        )  # second patient, offset -0.01
        self.assertAlmostEqual(float(by_key["Patient_06", "heart"]["assd"]), 3.0)

    def test_incomplete_experiment_is_marked_and_lists_missing_runs(self):
        write_cv_experiment(self.metrics, "exp", skip=(2, 1))
        run_aggregate(self.metrics)
        report = (self.metrics / "cv_summary.md").read_text()
        self.assertIn("11/12 runs found: INCOMPLETE, missing fold2/seed1", report)
        dice = table_rows(report, "exp", "Dice")
        self.assertEqual(
            dice["fold 2 (2 runs)"][0], "0.550 ± 0.014"
        )  # seeds 0 and 2 only
        self.assertEqual(
            dice["all runs (11 runs)"][0][:5], "0.539"
        )  # (12 * 0.54 - 0.55) / 11
        with (self.metrics / "cv_pooled_exp.csv").open() as f:
            n_seeds = {r["fold"]: r["n_seeds"] for r in csv.DictReader(f)}
        self.assertEqual(n_seeds, {"0": "3", "1": "3", "2": "2", "3": "3"})

    def test_nan_hd95_is_counted_per_organ_and_left_out_of_means(self):
        # Patient_03 (fold 1, first patient), heart, has no HD95 in seed 0 only
        write_cv_experiment(self.metrics, "exp", nan_hd95=(1, 0, "Patient_03", "heart"))
        run_aggregate(self.metrics)
        report = (self.metrics / "cv_summary.md").read_text()
        hd95 = table_rows(report, "exp", "HD95 mm")
        self.assertEqual(
            hd95["fold 1 (3 runs)"][0], "13.0 ± 1.0"
        )  # esophagus untouched, no NaN note
        self.assertTrue(hd95["fold 1 (3 runs)"][1].endswith("(1 NaN)"))
        self.assertTrue(hd95["all runs (12 runs)"][1].endswith("(1 NaN)"))
        self.assertNotIn("NaN", hd95["fold 0 (3 runs)"][1])
        self.assertNotIn("NaN", report.split("### HD95 mm")[0])
        with (self.metrics / "cv_pooled_exp.csv").open() as f:
            row = next(
                r
                for r in csv.DictReader(f)
                if (r["patient"], r["class_name"]) == ("Patient_03", "heart")
            )
        self.assertEqual(row["n_seeds"], "3")
        self.assertAlmostEqual(
            float(row["hd95"]), 14.5
        )  # seeds 1 and 2 only: 11 + 2 + (1 + 2) / 2

    def test_fg_counts_organ_values_missing_from_the_fg_mean(self):
        # every HD95 of the heart is NaN in fold 1 seed 0, so that run's heart value is NaN and fg is the esophagus only
        write_cv_experiment(self.metrics, "exp")
        run_dir = self.metrics / "exp_fold1" / "seed0"
        rows = patient_rows(1, 0, None)
        for row in rows:
            if row["class_name"] == "heart":
                row["hd95"] = float("nan")
        shutil.rmtree(run_dir)
        write_run(self.metrics, "exp_fold1", 0, rows)
        run_aggregate(self.metrics)
        hd95 = table_rows(
            (self.metrics / "cv_summary.md").read_text(), "exp", "HD95 mm"
        )
        self.assertEqual(
            hd95["fold 1 (3 runs)"][1], "14.5 ± 0.7 (2 NaN)"
        )  # the two heart patients
        self.assertEqual(
            hd95["fold 1 (3 runs)"][2], "13.3 ± 1.3 (1 NaN)"
        )  # fg = esophagus alone in that run
        self.assertEqual(hd95["fold 0 (3 runs)"][2], "11.5 ± 1.0")
        self.assertEqual(hd95["all runs (12 runs)"][2], "14.5 ± 2.5 (1 NaN)")

    def test_filtered_run_replaces_cv_summary_and_unfiltered_run_has_all(self):
        write_cv_experiment(self.metrics, "exp_a")
        write_cv_experiment(self.metrics, "exp_b")
        for name_filter in ("exp_a", "exp_b"):
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--metrics-dir", str(self.metrics), "--filter", name_filter])
        report = (self.metrics / "cv_summary.md").read_text()
        self.assertIn("## exp_b\n", report)
        self.assertNotIn("## exp_a\n", report)
        run_aggregate(self.metrics)
        report = (self.metrics / "cv_summary.md").read_text()
        self.assertIn("## exp_a\n", report)
        self.assertIn("## exp_b\n", report)

    def test_non_cv_experiment_output_is_unchanged(self):
        for seed in SEEDS:
            write_run(self.metrics, "plain", seed, patient_rows(0, seed, None))
        run_aggregate(self.metrics)
        # comparison.md as written before the cross-validation summary existed
        expected = (
            "| experiment | model | loss | seeds | val Dice 2D | 3D Dice fg | 3D Dice esophagus | 3D Dice heart"
            " | HD95 mm | ASSD mm |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n"
            "| plain | enet | ce | 3 | 0.500 ± 0.000 | 0.560 ± 0.010 | 0.510 ± 0.010 | 0.610 ± 0.010"
            " | 11.5 ± 1.0 | 2.00 ± 0.00 |\n\n"
            "mean ± std over seeds. 2D Dice: patient-level at 256x256 (training selection metric). "
            "3D: best checkpoint on the original CT grid, foreground = eval.classes. See docs/metrics.md.\n"
        )
        self.assertEqual((self.metrics / "comparison.md").read_text(), expected)
        self.assertEqual(
            sorted(p.name for p in self.metrics.glob("*.*")),
            ["comparison.md", "comparison_runs.csv"],
        )

    def test_cv_experiment_next_to_a_plain_one_keeps_the_plain_row(self):
        for seed in SEEDS:
            write_run(self.metrics, "plain", seed, patient_rows(0, seed, None))
        write_cv_experiment(self.metrics, "exp")
        run_aggregate(self.metrics)
        row = next(
            line
            for line in (self.metrics / "comparison.md").read_text().splitlines()
            if line.startswith("| plain |")
        )
        self.assertEqual(row.split("|")[4].strip(), "3")


if __name__ == "__main__":
    unittest.main()
