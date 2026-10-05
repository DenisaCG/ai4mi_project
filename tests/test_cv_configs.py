"""The cross-validation configs: same split and recipe, differing only in preprocessing, loss and name."""

import unittest

from src.config import REPO, load_config
from src.folds import is_cv, run_configs

CONFIGS = REPO / "configs"
STARTER, CURRENT = "full_cv4_enet_ce", "full_cv4_enet_dice_ce"
HAS_DATA = (REPO / "data" / "segthor_train_full" / "train").is_dir()


def load(name: str) -> dict:
    return load_config(CONFIGS / f"{name}.yaml")


class CvConfigTests(unittest.TestCase):
    def test_experiment_is_the_file_name(self):
        for name in (STARTER, CURRENT):
            self.assertEqual(load(name)["experiment"], name)

    def test_both_use_the_same_split_on_the_full_dataset(self):
        for name in (STARTER, CURRENT):
            cfg = load(name)
            p = cfg["data"]["preprocess"]
            self.assertEqual(p["source_dir"], "data/segthor_train_full")
            self.assertEqual(p["gt_version"], "corrected")
            self.assertEqual((p["retains"], p["fold"], p["seed"]), (10, 4, 0))
            self.assertEqual(
                cfg["data"]["source_pattern"],
                "data/segthor_train_full/train/{patient}/GT.nii.gz",
            )
            self.assertEqual(cfg["seed"], 0)
            self.assertEqual(cfg["eval"]["classes"], [1, 2, 3, 4])
            self.assertTrue(is_cv(cfg))

    def test_starter_has_no_preprocessing(self):
        p = load(STARTER)["data"]["preprocess"]
        self.assertFalse({"resample", "normalize", "crop"} & set(p))
        self.assertEqual(p["shape"], [256, 256])  # the course default

    def test_current_has_the_full_preprocessing_and_dice_ce(self):
        cfg = load(CURRENT)
        p = cfg["data"]["preprocess"]
        self.assertEqual(
            (p["resample"], p["normalize"], p["crop"]),
            ("median", "ct_window_zscore", "roi"),
        )
        self.assertEqual(cfg["loss"]["name"], "dice_ce")
        self.assertEqual(cfg["loss"]["kwargs"]["ce_idk"], [0, 1, 2, 3, 4])
        self.assertEqual(cfg["loss"]["kwargs"]["dice_idk"], [1, 2, 3, 4])

    def test_starter_loss_is_cross_entropy_over_all_classes(self):
        loss = load(STARTER)["loss"]
        self.assertEqual(
            (loss["name"], loss["kwargs"]), ("cross_entropy", {"idk": [0, 1, 2, 3, 4]})
        )

    def test_only_preprocessing_loss_and_names_differ(self):
        starter, current = load(STARTER), load(CURRENT)
        different = {k for k in starter if starter[k] != current[k]}
        self.assertEqual(different, {"experiment", "notes", "data", "loss"})
        self.assertEqual(
            {k for k in starter["data"] if starter["data"][k] != current["data"][k]},
            {"preprocess", "root"},
        )

    @unittest.skipUnless(HAS_DATA, "data/segthor_train_full is not on this machine")
    def test_folds_expand_to_distinct_datasets_and_runs(self):
        configs = run_configs(load(STARTER)) + run_configs(load(CURRENT))
        self.assertEqual(len(configs), 8)
        self.assertEqual(len({c["data"]["root"] for c in configs}), 8)
        self.assertEqual(len({c["experiment"] for c in configs}), 8)

    @unittest.skipUnless(HAS_DATA, "data/segthor_train_full is not on this machine")
    def test_every_patient_has_its_ct_and_ground_truth(self):
        pattern = load(STARTER)["data"]["source_pattern"]
        patients = sorted(
            p.name
            for p in (REPO / "data" / "segthor_train_full" / "train").glob("Patient_*")
        )
        self.assertEqual(len(patients), 40)
        for patient in patients:
            gt = REPO / pattern.format(patient=patient)
            self.assertTrue(gt.is_file(), gt)
            self.assertTrue((gt.parent / f"{patient}.nii.gz").is_file())


class ExistingConfigTests(unittest.TestCase):
    def test_non_cv_configs_are_untouched_by_the_fold_logic(self):
        for path in sorted(CONFIGS.glob("*.yaml")):
            if path.stem in (STARTER, CURRENT) or path.name == "base.yaml":
                continue
            cfg = load_config(path)
            self.assertFalse(is_cv(cfg), path.name)
            self.assertEqual(run_configs(cfg), [cfg], path.name)
            self.assertIs(run_configs(cfg, smoke=True)[0], cfg, path.name)


if __name__ == "__main__":
    unittest.main()
