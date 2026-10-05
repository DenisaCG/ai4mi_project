"""Cross-validation folds: disjoint validation sets covering every patient, one config and dataset per fold."""

import contextlib
import io
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from slice_segthor import get_splits
from src.config import load_config
from src.data import ensure_sliced
from src.folds import is_cv, run_configs, slice_command

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


if __name__ == "__main__":
    unittest.main()
