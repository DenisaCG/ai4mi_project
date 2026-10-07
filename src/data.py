"""Data loaders built from the `data` config section, reusing the original SliceDataset.

Extension point — online augmentation: register a factory under kind "augment" returning a callable
(image (C,H,W) float, gt (K,H,W) one-hot) -> (image, gt), then list it in `data.augment`:
    augment: [{name: random_flip, kwargs: {p: 0.5}}]
C is 1, or 2*data.context+1 with 2.5D input: an augmentation must apply identical geometry to every channel.
Augmentations run on the train split only. Offline preprocessing (HU windowing, resampling, ...)
belongs in the slicing step: write a new sliced dataset and point `data.root` at it.
"""
import json
import random
import re
import shutil
import subprocess
import sys
from functools import partial

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dataset import SliceDataset
from src.distance_maps import WithDistanceMaps
import src.augment  # noqa: F401  registers online augmentations
from src.config import REPO
from src.registry import build
from utils import class2one_hot


def ensure_sliced(cfg: dict) -> None:
    """Build data.root with slice_segthor.py from data.preprocess unless it already exists.
    Slices into a temp dir and renames, so a crash never leaves a half-built dataset behind."""
    p = cfg["data"]["preprocess"]
    root = REPO / cfg["data"]["root"]
    if p is None or root.exists():
        return
    tmp = root.with_name(root.name + "_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    subprocess.run([sys.executable, "slice_segthor.py", "--source_dir", p["source_dir"], "--dest_dir", str(tmp),
                    "--shape", *map(str, p["shape"]), "--retains", str(p["retains"]),
                    "--fold", str(p["fold"]), "--seed", str(p["seed"]),
                    "--gt_version", p["gt_version"]] + (["--resample", p["resample"]] if p.get("resample") else [])
                   + (["--num_folds", str(p["num_folds"])] if p.get("num_folds") is not None else [])
                   + (["--normalize", p["normalize"]] if p.get("normalize") else [])
                   + (["--crop", p["crop"]] if p.get("crop") else []),
                   cwd=REPO, check=True)
    tmp.rename(root)


def img_transform(img: Image.Image, zscore: dict | None = None) -> torch.Tensor:
    arr = np.array(img.convert("L"))[np.newaxis, ...] / 255  # (1, H, W) in [0, 1]
    if zscore is not None:
        # Branch taken for data.preprocess.normalize == ct_window_zscore: the PNG is the clipped HU window rescaled
        # to 0..255 (slice_segthor.py), so de-quantize to HU and apply the train-set foreground mean/std. Applied
        # here only, and only on this path: the PNGs never hold z-scores, so nothing is normalized twice.
        arr = (zscore["lo"] + arr * (zscore["hi"] - zscore["lo"]) - zscore["mean"]) / zscore["std"]
    return torch.tensor(arr, dtype=torch.float32)


def gt_transform(num_classes: int, label_scale: int, img: Image.Image) -> torch.Tensor:
    labels = torch.tensor(np.array(img) // label_scale, dtype=torch.int64)[None, ...]
    return class2one_hot(labels, K=num_classes)[0]  # (K, H, W)


class SliceStack(Dataset):
    """2.5D input: slices z-c..z+c of the same patient stacked as channels, target and stem of slice z.

    `files` is the split's full (img, gt) list, so neighbours exist even when `base` was subsampled. Slices past either
    end of a volume repeat the nearest edge slice. Each neighbour goes through the base img_transform (z-score per channel).
    """

    def __init__(self, base: SliceDataset, context: int, files: list, patient_regex: str):
        self.base, self.context = base, context
        self.paths: dict[tuple[str, int], object] = {}
        for img, _ in files:
            self.paths[patient_of(img.stem, patient_regex), slice_index(img.stem)] = img
        self.z_range: dict[str, tuple[int, int]] = {}
        for patient, z in self.paths:
            lo, hi = self.z_range.get(patient, (z, z))
            self.z_range[patient] = (min(lo, z), max(hi, z))
        self.patient_regex = patient_regex

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        patient, z = patient_of(item["stems"], self.patient_regex), slice_index(item["stems"])
        lo, hi = self.z_range[patient]
        channels = []
        for dz in range(-self.context, self.context + 1):
            if dz == 0:
                channels.append(item["images"])
                continue
            neighbour = min(max(z + dz, lo), hi)
            if (patient, neighbour) not in self.paths:
                raise FileNotFoundError(f"{patient} has no slice {neighbour} (needed as context for slice {z})")
            channels.append(self.base.img_transform(Image.open(self.paths[patient, neighbour])))
        item["images"] = torch.cat(channels, dim=0)  # (2c+1, H, W)
        return item


class Augmented(Dataset):
    def __init__(self, base: SliceDataset, augments: list):
        self.base, self.augments = base, augments

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        for aug in self.augments:
            item["images"], item["gts"] = aug(item["images"], item["gts"])
        return item


def seed_worker(worker_id: int) -> None:
    # torch seeds each worker deterministically; propagate that to numpy/random for augmentations
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_dataset(cfg: dict, split: str) -> Dataset:
    d = cfg["data"]
    zscore = None
    if (d["preprocess"] or {}).get("normalize") == "ct_window_zscore":
        zscore = json.loads((REPO / d["root"] / "ct_norm_stats.json").read_text())  # same numbers for every split
    dataset = SliceDataset(split, REPO / d["root"], img_transform=partial(img_transform, zscore=zscore),
                           gt_transform=partial(gt_transform, d["num_classes"], d["label_scale"]))
    all_files = list(dataset.files)  # neighbour lookup for 2.5D input must not shrink with the subsets below
    if split == "train" and cfg["train"].get("train_patients"):
        dataset.files = keep_patients(dataset.files, d["patient_regex"], cfg["train"]["train_patients"])
    if cfg["train"]["debug_samples"]:
        dataset.files = random.Random(cfg["seed"]).sample(dataset.files, cfg["train"]["debug_samples"])
    context = d.get("context", 0)  # saved config.yaml files of runs from before 2.5D (evaluate.py) have no such key
    if context > 0:
        dataset = SliceStack(dataset, context, all_files, d["patient_regex"])  # before Augmented: it sees the whole stack
    if split == "train" and d["augment"]:
        augments = []
        for a in d["augment"]:
            kwargs = dict(a.get("kwargs", {}))
            if kwargs.get("fill") == "ct_window_low":
                if zscore is None:
                    raise ValueError("ct_window_low fill requires ct_window_zscore normalization")
                kwargs["fill"] = (zscore["lo"] - zscore["mean"]) / zscore["std"]
            augments.append(build("augment", a["name"], **kwargs))
        dataset = Augmented(dataset, augments)
    if split != "test" and cfg.get("loss", {}).get("name") in ("boundary", "ce_dice_boundary", "weighted_ce_dice_boundary"):
        classes = cfg["loss"]["kwargs"].get("boundary_idk")
        dataset = WithDistanceMaps(dataset, list(range(1, d["num_classes"])) if classes is None else classes)
    return dataset


def build_loader(cfg: dict, split: str, device: torch.device) -> DataLoader:
    d = cfg["data"]
    return DataLoader(build_dataset(cfg, split), batch_size=d["batch_size"],
                      num_workers=d["num_workers"], shuffle=split == "train",
                      worker_init_fn=seed_worker, pin_memory=device.type == "cuda")


def patient_of(stem: str, regex: str) -> str:
    match = re.fullmatch(regex, stem)
    if match is None:
        raise ValueError(f"slice '{stem}' does not match data.patient_regex '{regex}'")
    return match.group(1)


def slice_index(stem: str) -> int:
    """z index of a slice stem such as Patient_01_0042."""
    return int(stem.rsplit("_", 1)[1])


def keep_patients(files: list, regex: str, n: int) -> list:
    """Keeps the slices of the first `n` patients of a fixed shuffle, so smaller sets nest in larger ones.

    The shuffle is seeded with 0, not with the run seed: every run sees the same patients.
    """
    patients = sorted({patient_of(img.stem, regex) for img, _ in files})
    if n > len(patients):
        raise ValueError(f"train.train_patients={n} but the split has only {len(patients)} patients")
    random.Random(0).shuffle(patients)
    keep = set(patients[:n])
    return [f for f in files if patient_of(f[0].stem, regex) in keep]
