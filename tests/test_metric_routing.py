"""Augmentation metric-copy routing and discovery in the categorized layout."""

import json
import tempfile
import unittest
from pathlib import Path

from src.aggregate import load_runs, run_metrics_dir
from src.run import metrics_dir_for


class MetricRoutingTests(unittest.TestCase):
    def test_copy_destinations_use_category_and_preserve_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = {"paths": {"metrics_dir": str(root)},
                   "data": {"augment": [], "registration": None}}
            self.assertEqual(metrics_dir_for(cfg), root)
            for category in ("screening", "joint", "registration"):
                cfg["paths"]["metrics_category"] = category
                self.assertEqual(metrics_dir_for(cfg), root / "augmentation_metrics" / category)
            cfg["paths"].pop("metrics_category")
            cfg["data"]["augment"] = [{"name": "random_rotation"}]
            self.assertEqual(metrics_dir_for(cfg), root / "augmentation_metrics/screening")
            cfg["data"]["augment"].append({"name": "random_scaling"})
            self.assertEqual(metrics_dir_for(cfg), root / "augmentation_metrics/joint")
            cfg["data"]["registration"] = {"root": "synthetic"}
            self.assertEqual(metrics_dir_for(cfg), root / "augmentation_metrics/registration")

    def test_aggregate_discovers_nested_runs_without_double_counting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for subdir, experiment in (("", "control_fold0"),
                                       ("augmentation_metrics/joint", "augmented_fold0")):
                run_dir = root / subdir / experiment / "seed0"
                run_dir.mkdir(parents=True)
                (run_dir / "summary.json").write_text(json.dumps({
                    "experiment": experiment, "run": "seed0", "seed": 0,
                    "eval": {"val_dice_fg": 0.8},
                }))
            runs = load_runs(root, "")
            self.assertEqual(len(runs), 2)
            augmented = load_runs(root, "augmented")
            self.assertEqual(len(augmented), 1)
            self.assertEqual(run_metrics_dir(root, augmented[0]),
                             root / "augmentation_metrics/joint/augmented_fold0/seed0")
            duplicate = root / "augmented_fold0/seed0"
            duplicate.mkdir(parents=True)
            (duplicate / "summary.json").write_text((
                root / "augmentation_metrics/joint/augmented_fold0/seed0/summary.json"
            ).read_text())
            with self.assertRaisesRegex(ValueError, "duplicate metrics"):
                load_runs(root, "")


if __name__ == "__main__":
    unittest.main()
