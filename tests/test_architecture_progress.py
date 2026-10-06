"""dataset_analysis/architecture_progress.py on a fake metrics/ tree: 3 complete 4 x 3 experiments, one with a NaN HD95.

The script runs as a subprocess (it imports dataset_analysis/utils.py, which the repo-root utils.py would shadow
in this process) and the plotted means are compared with the numbers src.aggregate writes to cv_summary.md.
"""

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.aggregate import CV_METRICS
from tests import test_aggregate_cv as fake

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "dataset_analysis/architecture_progress.py"
ORGANS = ("esophagus", "heart", "trachea", "aorta")
EXPERIMENTS = {
    "exp_a": 0,
    "exp_b": 1,
    "exp_c": 2,
}  # name -> improvement step in the fake values
STEPS = [
    "Zeta=exp_c",
    "Alpha\\nfirst=exp_a",
    "Mid=exp_b",
]  # neither alphabetical nor creation order
NAN_HD95 = (1, 0, "Patient_11", "heart")  # fold, seed, patient, organ: in exp_b only


def run_rows(
    step: int, fold: int, seed: int, nan: tuple[str, str] | None
) -> list[dict]:
    rng = np.random.default_rng([step, fold, seed])
    rows = []
    for i in range(10):
        patient = f"Patient_{10 * fold + i + 1:02d}"
        for k, organ in enumerate(ORGANS):
            rows.append(
                {
                    "split": "val",
                    "patient": patient,
                    "class_idx": k + 1,
                    "class_name": organ,
                    "gt_voxels": 100,
                    "pred_voxels": 90,
                    "dice": 0.6 + 0.05 * k + 0.03 * step + rng.normal(0, 0.05),
                    "hd95": float("nan")
                    if nan == (patient, organ)
                    else 20 - 3 * step + 2 * k + abs(rng.normal(0, 3)),
                    "assd": 4 - 0.5 * step + 0.3 * k + abs(rng.normal(0, 0.5)),
                }
            )
    return rows


class ArchitectureProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.metrics = Path(cls.tmp.name) / "metrics"
        cls.out = Path(cls.tmp.name) / "figures"
        patcher = mock.patch.object(
            fake, "ORGANS", ORGANS
        )  # write_run reads the organs from there
        patcher.start()
        cls.addClassCleanup(patcher.stop)
        for experiment, step in EXPERIMENTS.items():
            for fold in fake.FOLDS:
                for seed in fake.SEEDS:
                    nan = (
                        NAN_HD95[2:]
                        if experiment == "exp_b" and (fold, seed) == NAN_HD95[:2]
                        else None
                    )
                    fake.write_run(
                        cls.metrics,
                        f"{experiment}_fold{fold}",
                        seed,
                        run_rows(step, fold, seed, nan),
                    )
        fake.run_aggregate(cls.metrics)
        cls.result = cls.run_script(
            "--experiments", *STEPS, "--compare", "Zeta", "Mid", "--out", str(cls.out)
        )
        cls.report = (cls.metrics / "cv_summary.md").read_text()

    @classmethod
    def run_script(cls, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--metrics-dir", str(cls.metrics), *args],
            capture_output=True,
            check=False,
            text=True,
        )

    def table(self) -> list[dict]:
        with (self.out / "architecture_progress.csv").open() as f:
            return list(csv.DictReader(f))

    def test_writes_one_figure_per_metric_and_the_per_patient_figure(self):
        self.assertEqual(self.result.returncode, 0, self.result.stderr)
        for _, key, _ in CV_METRICS:
            for suffix in ("png", "pdf"):
                self.assertTrue(
                    (self.out / f"architecture_progress_{key}.{suffix}").is_file()
                )
        for suffix in ("png", "pdf"):
            self.assertTrue(
                (self.out / f"architecture_per_patient_zeta_vs_mid.{suffix}").is_file()
            )

    def test_steps_follow_the_order_of_experiments(self):
        steps = list(dict.fromkeys(row["step"] for row in self.table()))
        self.assertEqual(steps, ["Zeta", "Alpha first", "Mid"])

    def test_marker_means_and_nan_counts_equal_the_aggregate(self):
        by_key = {(r["metric"], r["experiment"], r["organ"]): r for r in self.table()}
        for label, key, digits in CV_METRICS:
            for experiment in EXPERIMENTS:
                cells = fake.table_rows(self.report, experiment, label)[
                    "all runs (12 runs)"
                ]
                for organ, cell in zip(ORGANS, cells):
                    row = by_key[key, experiment, organ]
                    mean, _, nan = (
                        cell.partition(" ± ")[0],
                        None,
                        cell.partition("(")[2],
                    )
                    self.assertEqual(f"{float(row['mean']):.{digits}f}", mean)
                    self.assertEqual(
                        int(row["n_nan"]), int(nan.split()[0]) if nan else 0
                    )
        self.assertEqual(by_key["hd95", "exp_b", "heart"]["n_nan"], "1")

    def test_incomplete_experiment_is_rejected(self):
        result = self.run_script(
            "--experiments", "Zeta=exp_c", "Gone=exp_missing", "--out", str(self.out)
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exp_missing: 0/12 runs found", result.stderr)


if __name__ == "__main__":
    unittest.main()
