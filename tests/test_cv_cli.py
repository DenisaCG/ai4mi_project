"""`--fold` on src.train and src.run, and --smoke on a cross-validation config."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.config import REPO
from src.train import main as train_main

CONFIG = "experiment: cli_test\ndata:\n  preprocess: {{source_dir: {source}, gt_version: corrected, retains: 10, fold: {fold}, seed: 0}}\n"


class CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        for i in range(1, 41):
            (root / "data" / "train" / f"Patient_{i:02d}").mkdir(parents=True)
        cls.cv = root / "cv.yaml"
        cls.plain = root / "plain.yaml"
        cls.cv.write_text(CONFIG.format(source=root / "data", fold=4))
        cls.plain.write_text(CONFIG.format(source=root / "data", fold=0))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def trained(self, *args: str) -> list[tuple[str, bool, bool]]:
        """(experiment, force, smoke) of every run that train.main starts, without training anything."""
        with mock.patch("src.train.train_one") as train_one:
            train_main(list(args))
        return [
            (c.args[0]["experiment"], c.kwargs["force"], c.kwargs["smoke"])
            for c in train_one.call_args_list
        ]

    def run_dir(self, config: Path, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "src.run", "--config", str(config), *args],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_train_without_fold_runs_every_fold_in_order(self):
        runs = self.trained("--config", str(self.cv))
        self.assertEqual(runs, [(f"cli_test_fold{k}", False, False) for k in range(4)])

    def test_train_fold_runs_exactly_that_fold(self):
        self.assertEqual(
            self.trained("--config", str(self.cv), "--fold", "2"),
            [("cli_test_fold2", False, False)],
        )
        forced = self.trained("--config", str(self.cv), "--fold", "1", "--force")
        self.assertEqual(
            forced, [("cli_test_fold1", True, False)]
        )  # --force never touches the other folds

    def test_a_failing_fold_stops_the_loop_but_not_a_separate_fold_run(self):
        started = []

        def flaky(cfg, **kwargs):
            started.append(cfg["experiment"])
            if cfg["experiment"].endswith("fold1"):
                raise RuntimeError("fold 1 failed")

        with mock.patch("src.train.train_one", side_effect=flaky):
            with self.assertRaises(RuntimeError):
                train_main(["--config", str(self.cv)])
            self.assertEqual(started, ["cli_test_fold0", "cli_test_fold1"])
            train_main(["--config", str(self.cv), "--fold", "2"])
        self.assertEqual(started[-1], "cli_test_fold2")

    def test_train_smoke_runs_fold_zero_only(self):
        self.assertEqual(
            self.trained("--config", str(self.cv), "--smoke"),
            [("cli_test_fold0", False, True)],
        )

    def test_train_plain_config_is_unchanged(self):
        self.assertEqual(
            self.trained("--config", str(self.plain)), [("cli_test", False, False)]
        )
        self.assertEqual(
            self.trained("--config", str(self.plain), "--smoke"),
            [("cli_test", False, True)],
        )
        with self.assertRaises(ValueError):
            self.trained("--config", str(self.plain), "--fold", "0")

    def test_train_rejects_a_fold_out_of_range(self):
        with self.assertRaises(ValueError):
            self.trained("--config", str(self.cv), "--fold", "4")

    def test_run_prints_the_fold_run_directory(self):
        done = self.run_dir(self.cv, "--fold", "2", "--set", "seed=5")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(
            done.stdout.strip(), str(REPO / "runs" / "cli_test_fold2" / "seed5")
        )

    def test_run_smoke_defaults_to_fold_zero(self):
        done = self.run_dir(self.cv, "--smoke")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(
            done.stdout.strip(),
            str(REPO / "runs" / "_smoke" / "cli_test_fold0" / "seed0"),
        )

    def test_run_needs_a_fold_for_a_cv_config(self):
        done = self.run_dir(self.cv)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("--fold", done.stderr)

    def test_run_plain_config_is_unchanged(self):
        done = self.run_dir(self.plain)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip(), str(REPO / "runs" / "cli_test" / "seed0"))
        self.assertNotEqual(self.run_dir(self.plain, "--fold", "0").returncode, 0)

    def test_every_fold_has_a_distinct_run_directory(self):
        dirs = {self.run_dir(self.cv, "--fold", str(k)).stdout for k in range(4)}
        self.assertEqual(len(dirs), 4)


if __name__ == "__main__":
    unittest.main()
