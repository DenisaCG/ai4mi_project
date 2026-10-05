"""Cross-validation: a config with `data.preprocess.fold > 1` asks for that many folds and expands to one config per fold."""

import copy
import sys

from src.config import REPO, resolve_preprocess
from src.splits import validate_split


def is_cv(cfg: dict) -> bool:
    """Whether `cfg` asks for several folds. A per-fold config already has `num_folds` and is a single split."""
    p = cfg["data"]["preprocess"]
    return bool(p) and p["fold"] > 1 and p.get("num_folds") is None


def run_configs(cfg: dict, fold: int | None = None, smoke: bool = False) -> list[dict]:
    """Returns the configs to run for `cfg`.

    A plain config yields itself. A CV config yields one config per selected fold, named
    `<experiment>_fold<k>`, each with its own sliced dataset, run directory and metrics directory.

    Args:
        cfg: Resolved config.
        fold: Zero-based fold to select. None selects every fold, or only fold 0 when `smoke` is set.
        smoke: A smoke test needs one fold, not all of them.

    Raises:
        ValueError: `fold` given for a plain config or outside 0..num_folds-1, or the split sizes are invalid.
    """
    if not is_cv(cfg):
        if fold is not None:
            raise ValueError(
                "--fold needs a cross-validation config (data.preprocess.fold > 1)"
            )
        return [cfg]
    p = cfg["data"]["preprocess"]
    num_folds = p["fold"]
    patients = list((REPO / p["source_dir"] / "train").glob("Patient_*"))
    validate_split(len(patients), p["retains"], 0, num_folds)
    if fold is None:
        folds = [0] if smoke else range(num_folds)
    elif 0 <= fold < num_folds:
        folds = [fold]
    else:
        raise ValueError(f"--fold {fold} is outside 0..{num_folds - 1}")
    configs = []
    for k in folds:
        fold_cfg = copy.deepcopy(cfg)
        fold_cfg["experiment"] = f"{cfg['experiment']}_fold{k}"
        fold_cfg["data"]["preprocess"].update(fold=k, num_folds=num_folds)
        resolve_preprocess(fold_cfg)
        configs.append(fold_cfg)
    return configs


def slice_command(cfg: dict, dest: str, processes: int = 1) -> list[str]:
    """Returns the slice_segthor.py command that builds the dataset of a resolved `cfg` at `dest`.

    Same arguments as `src.data.ensure_sliced`, plus the number of worker processes, so a CPU job can
    build a dataset in parallel before any GPU time is spent.
    """
    p = cfg["data"]["preprocess"]
    return (
        [
            sys.executable,
            "slice_segthor.py",
            "--source_dir",
            p["source_dir"],
            "--dest_dir",
            dest,
            "--shape",
            *map(str, p["shape"]),
            "--retains",
            str(p["retains"]),
            "--fold",
            str(p["fold"]),
            "--seed",
            str(p["seed"]),
            "--gt_version",
            p["gt_version"],
        ]
        + (["--resample", p["resample"]] if p.get("resample") else [])
        + (
            ["--num_folds", str(p["num_folds"])]
            if p.get("num_folds") is not None
            else []
        )
        + (["--normalize", p["normalize"]] if p.get("normalize") else [])
        + (["--crop", p["crop"]] if p.get("crop") else [])
        + ["--process", str(processes)]
    )
