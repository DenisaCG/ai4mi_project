"""Cross-validation folds: disjoint validation sets covering every patient, one config and dataset per fold."""

import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest import mock

from PIL import Image

from slice_segthor import get_splits
from src.config import load_config
from src.data import ensure_sliced
from src.folds import check_dataset, is_cv, run_configs, slice_command

PATIENTS, RETAINS, NUM_FOLDS = 40, 10, 4
FULL_PREPROCESSING = "resample: median, normalize: ct_window_zscore, crop: roi"


class FoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.source = Path(cls.tmp.name)
        for i in range(1, PATIENTS + 1):
            (cls.source / "train" / f"Patient_{i:02d}").mkdir(parents=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def config(
        self,
        *overrides: str,
        retains: int = RETAINS,
        fold: int = NUM_FOLDS,
        extra: str = "",
    ) -> dict:
        """Config over the 40 empty patient directories; `fold` > 1 is a CV config."""
        preprocess = (
            f"{{source_dir: {self.source}, gt_version: corrected, retains: {retains}, "
            f"fold: {fold}, seed: 0{extra}}}"
        )
        return load_config(
            None, ["experiment=cv_test", f"data.preprocess={preprocess}", *overrides]
        )

    def split(self, cfg: dict) -> tuple[list[str], list[str]]:
        """(train, val) patients exactly as slice_segthor.py draws them for one fold config."""
        p = cfg["data"]["preprocess"]
        random.seed(p["seed"])
        with contextlib.redirect_stdout(io.StringIO()):
            train, val, _ = get_splits(
                Path(p["source_dir"]), p["retains"], p["fold"], p["num_folds"]
            )
        return train, val

    def test_validation_sets_partition_the_patients(self):
        splits = [self.split(c) for c in run_configs(self.config())]
        every = {f"Patient_{i:02d}" for i in range(1, PATIENTS + 1)}
        validation = [set(val) for _, val in splits]
        self.assertEqual([len(v) for v in validation], [RETAINS] * NUM_FOLDS)
        # every patient is validated in some fold, and in exactly one
        self.assertEqual(set.union(*validation), every)
        self.assertEqual(sum(len(v) for v in validation), PATIENTS)
        for train, val in splits:
            self.assertEqual(set(train) | set(val), every)
            self.assertFalse(set(train) & set(val))
            self.assertEqual(len(train), PATIENTS - RETAINS)

    def test_split_is_stable_across_reruns_and_run_seeds(self):
        first = [self.split(c) for c in run_configs(self.config())]
        self.assertEqual(first, [self.split(c) for c in run_configs(self.config())])
        other_run_seed = run_configs(self.config("seed=7"))
        self.assertEqual(first, [self.split(c) for c in other_run_seed])

    def test_each_fold_has_its_own_name_dataset_and_split_settings(self):
        base = self.config()
        configs = run_configs(base)
        names = [f"cv_test_fold{k}" for k in range(NUM_FOLDS)]
        self.assertEqual([c["experiment"] for c in configs], names)
        self.assertEqual(len({c["data"]["root"] for c in configs}), NUM_FOLDS)

        def rest(cfg):
            return {k: v for k, v in cfg.items() if k not in ("experiment", "data")}

        for k, c in enumerate(configs):
            p = c["data"]["preprocess"]
            self.assertEqual(
                (p["fold"], p["num_folds"], p["retains"], p["seed"]),
                (k, NUM_FOLDS, RETAINS, 0),
            )
            self.assertEqual(rest(c), rest(base))
        # the input config is not modified
        self.assertIsNone(base["data"]["preprocess"].get("num_folds"))

    def test_run_seed_does_not_change_dataset_roots(self):
        roots = [c["data"]["root"] for c in run_configs(self.config())]
        seeded = [c["data"]["root"] for c in run_configs(self.config("seed=3"))]
        self.assertEqual(roots, seeded)

    def test_fold_selects_exactly_one(self):
        (only,) = run_configs(self.config(), fold=2)
        self.assertEqual(only["experiment"], "cv_test_fold2")
        self.assertEqual(only["data"]["preprocess"]["fold"], 2)
        self.assertEqual(
            only["data"]["root"], run_configs(self.config())[2]["data"]["root"]
        )

    def test_fold_out_of_range_is_rejected(self):
        for fold in (-1, NUM_FOLDS, 10):
            with self.assertRaises(ValueError, msg=fold):
                run_configs(self.config(), fold=fold)

    def test_smoke_runs_fold_zero_only(self):
        (only,) = run_configs(self.config(), smoke=True)
        self.assertEqual(only["experiment"], "cv_test_fold0")
        # an explicit fold wins over the smoke default
        (chosen,) = run_configs(self.config(), fold=3, smoke=True)
        self.assertEqual(chosen["experiment"], "cv_test_fold3")

    def test_invalid_split_sizes_fail_before_any_run(self):
        with self.assertRaises(ValueError):
            run_configs(self.config(retains=8))

    def test_plain_config_is_returned_unchanged(self):
        plain = self.config(fold=0)
        self.assertFalse(is_cv(plain))
        self.assertEqual(run_configs(plain), [plain])
        self.assertIs(run_configs(plain)[0], plain)
        self.assertEqual(run_configs(plain, smoke=True), [plain])
        with self.assertRaises(ValueError):
            run_configs(plain, fold=0)
        no_preprocess = load_config(None, ["experiment=x"])
        self.assertEqual(run_configs(no_preprocess), [no_preprocess])

    def test_saved_fold_config_is_a_single_split(self):
        (saved,) = run_configs(self.config(), fold=1)
        self.assertFalse(is_cv(saved))
        self.assertEqual(run_configs(saved), [saved])

    def test_slice_command_matches_ensure_sliced(self):
        """The CPU build job and a training run must build the same dataset.

        Their slice_segthor.py arguments are identical, up to the destination and the workers.
        """
        for cfg in (self.config(), self.config(extra=", " + FULL_PREPROCESSING)):
            for fold_cfg in run_configs(cfg):
                patched = mock.patch(
                    "src.data.subprocess.run", side_effect=KeyError("stop")
                )
                with patched as run, self.assertRaises(KeyError):
                    ensure_sliced(fold_cfg)
                used = run.call_args.args[0]
                command = slice_command(fold_cfg, "DEST", processes=7)
                self.assertEqual(command[-2:], ["--process", "7"])
                dest_at = used.index("--dest_dir") + 1
                used[dest_at] = "DEST"
                self.assertEqual(used, command[:-2])


class CheckDatasetTests(unittest.TestCase):
    """check_dataset on a miniature sliced dataset: 6 patients, 2 of them validation, 8x8 slices."""

    TRAIN, VAL = (
        ["Patient_01", "Patient_02", "Patient_03", "Patient_04"],
        ["Patient_05", "Patient_06"],
    )
    P: ClassVar[dict] = {"retains": 2, "shape": [8, 8]}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.everyone = set(self.TRAIN + self.VAL)
        for split, patients in (("train", self.TRAIN), ("val", self.VAL)):
            for kind in ("img", "gt"):
                (self.root / split / kind).mkdir(parents=True)
                for patient in patients:
                    for z in range(3):
                        Image.new("L", (8, 8)).save(
                            self.root / split / kind / f"{patient}_{z:04d}.png"
                        )

    def write(self, name: str, content: dict):
        (self.root / name).write_text(json.dumps(content))

    def test_consistent_dataset_returns_the_validation_patients(self):
        self.write("ct_norm_stats.json", {"train_patients": self.TRAIN})
        self.write(
            "roi_crop.json",
            {"size": 8, "required_train": dict.fromkeys(self.TRAIN, 1.0)},
        )
        p = self.P | {"crop": "roi"}
        self.assertEqual(check_dataset(self.root, p, self.everyone), set(self.VAL))
        self.assertEqual(check_dataset(self.root, self.P, self.everyone), set(self.VAL))

    def test_statistics_from_a_validation_patient_are_rejected(self):
        self.write(
            "ct_norm_stats.json", {"train_patients": [*self.TRAIN, "Patient_05"]}
        )
        with self.assertRaisesRegex(ValueError, "ct_norm_stats.json.*Patient_05"):
            check_dataset(self.root, self.P, self.everyone)

    def test_roi_window_from_a_validation_patient_is_rejected(self):
        self.write(
            "roi_crop.json",
            {
                "size": 8,
                "required_train": dict.fromkeys([*self.TRAIN, "Patient_06"], 1.0),
            },
        )
        with self.assertRaisesRegex(ValueError, "roi_crop.json.*Patient_06"):
            check_dataset(self.root, self.P | {"crop": "roi"}, self.everyone)

    def test_patient_in_both_splits_is_rejected(self):
        for kind in ("img", "gt"):
            Image.new("L", (8, 8)).save(
                self.root / "val" / kind / "Patient_01_0000.png"
            )
        with self.assertRaisesRegex(ValueError, "both train and val"):
            check_dataset(self.root, self.P | {"retains": 3}, self.everyone)

    def test_wrong_validation_size_or_missing_patient_is_rejected(self):
        with self.assertRaises(ValueError):
            check_dataset(self.root, self.P | {"retains": 3}, self.everyone)
        with self.assertRaises(ValueError):
            check_dataset(self.root, self.P, self.everyone | {"Patient_07"})

    def test_slice_size_and_label_mismatch_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected 16x16"):
            check_dataset(self.root, self.P | {"shape": [16, 16]}, self.everyone)
        (self.root / "val" / "gt" / "Patient_06_0002.png").unlink()
        with self.assertRaisesRegex(ValueError, "differ"):
            check_dataset(self.root, self.P, self.everyone)


if __name__ == "__main__":
    unittest.main()
