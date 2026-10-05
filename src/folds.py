"""Cross-validation: a config with `data.preprocess.fold > 1` asks for that many folds and expands to one config per fold."""

import copy
import json
import sys
from pathlib import Path

from PIL import Image

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


def check_dataset(root: Path, p: dict, patients: set[str]) -> set[str]:
    """Checks a sliced fold dataset and returns its validation patients.

    Args:
        root: Sliced dataset directory (`data/sliced_<hash>`).
        p: The `data.preprocess` block it was built from.
        patients: Every patient of the source dataset.

    Raises:
        ValueError: Image and label slices differ, a slice has the wrong size, train and validation patients
            overlap, the split has the wrong size or does not cover `patients`, or the normalisation or ROI
            statistics were computed from patients other than the training ones.
    """
    side = (
        json.loads((root / "roi_crop.json").read_text())["size"]
        if p.get("crop")
        else p["shape"][0]
    )
    split_patients = {}
    for split in ("train", "val"):
        img = sorted(f.name for f in (root / split / "img").glob("*.png"))
        gt = sorted(f.name for f in (root / split / "gt").glob("*.png"))
        if not img or img != gt:
            raise ValueError(
                f"{split}: image and label slices differ or are missing ({len(img)} vs {len(gt)})"
            )
        for name in (img[0], img[len(img) // 2], img[-1]):
            for kind in ("img", "gt"):
                with Image.open(root / split / kind / name) as png:
                    if png.size != (side, side):
                        raise ValueError(
                            f"{split}/{kind}/{name}: size {png.size}, expected {side}x{side}"
                        )
        split_patients[split] = {name.rsplit("_", 1)[0] for name in img}
    train, val = split_patients["train"], split_patients["val"]
    if train & val:
        raise ValueError(f"patients in both train and val: {sorted(train & val)}")
    if len(val) != p["retains"] or train | val != patients:
        raise ValueError(
            f"split has {len(train)} train / {len(val)} val patients, expected {p['retains']} val of {len(patients)}"
        )
    for stats, key in (
        ("ct_norm_stats.json", "train_patients"),
        ("roi_crop.json", "required_train"),
    ):
        if (root / stats).exists():
            used = set(json.loads((root / stats).read_text())[key])
            if used != train:
                raise ValueError(
                    f"{stats} was computed from {sorted(used ^ train)} that are not the training patients"
                )
    return val
