"""Convert the corrected SegTHOR release to nnU-Net raw format and write its split.

Usage, with nnUNet_raw and nnUNet_preprocessed exported:

    python nnunet/prepare_dataset.py convert
    nnUNetv2_plan_and_preprocess -d 102 --verify_dataset_integrity -c 3d_fullres
    python nnunet/prepare_dataset.py splits
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

REPO = Path(__file__).resolve().parents[1]
DATASET_NAME = "Dataset102_SegTHOR_corrected"
LABELS = {"background": 0, "esophagus": 1, "heart": 2, "trachea": 3, "aorta": 4}
# Same five patients as data/SEGTHOR_corrected/val (slice_segthor.py, --retain 5, seed 0).
VAL_IDS = ["Patient_01", "Patient_11", "Patient_15", "Patient_17", "Patient_19"]
SUFFIX = ".nii.gz"


def link(src: Path, dst: Path) -> None:
    """Hardlink src to dst, copying instead when they are on different filesystems."""
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def convert(data_dir: Path, dataset_dir: Path) -> list[str]:
    """Put every patient's CT and ground truth into nnU-Net's raw layout.

    Only Patient_XX.nii.gz and GT.nii.gz are used; the other files in a patient
    folder (original labels, segmentation project, label table) are left out.

    Args:
        data_dir: Folder holding one Patient_XX folder per patient.
        dataset_dir: nnU-Net raw dataset folder to fill (imagesTr, labelsTr, dataset.json).

    Returns:
        The patient IDs that were converted, sorted.

    Raises:
        FileNotFoundError: If no patient folder exists or a patient lacks its CT or GT.
    """
    patients = sorted(p for p in data_dir.glob("Patient_*") if p.is_dir())
    if not patients:
        raise FileNotFoundError(f"no Patient_* folders in {data_dir}")
    missing = [
        p / name
        for p in patients
        for name in (p.name + SUFFIX, "GT" + SUFFIX)
        if not (p / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing files: {', '.join(map(str, missing))}")

    images, labels = dataset_dir / "imagesTr", dataset_dir / "labelsTr"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for p in patients:
        link(p / (p.name + SUFFIX), images / f"{p.name}_0000{SUFFIX}")
        link(p / ("GT" + SUFFIX), labels / (p.name + SUFFIX))

    generate_dataset_json(
        str(dataset_dir),
        channel_names={0: "CT"},
        labels=LABELS,
        num_training_cases=len(patients),
        file_ending=SUFFIX,
        dataset_name=DATASET_NAME,
        description="SegTHOR part 1 with corrected labels",
        converted_by="nnunet/prepare_dataset.py",
    )
    return [p.name for p in patients]


def write_splits(dataset_dir: Path, preprocessed_dir: Path) -> None:
    """Write nnU-Net's splits_final.json with a single fold.

    The fold validates on VAL_IDS and trains on every other patient.

    Args:
        dataset_dir: nnU-Net raw dataset folder, used to list the patients.
        preprocessed_dir: Preprocessed dataset folder; must already exist, so
            planning has to run first.

    Raises:
        ValueError: If a validation patient is not in the dataset.
    """
    ids = sorted(
        p.name.removesuffix(SUFFIX)
        for p in (dataset_dir / "labelsTr").glob("*" + SUFFIX)
    )
    unknown = sorted(set(VAL_IDS) - set(ids))
    if unknown:
        raise ValueError(f"validation patients not in the dataset: {unknown}")
    split = [{"train": [i for i in ids if i not in VAL_IDS], "val": VAL_IDS}]
    (preprocessed_dir / "splits_final.json").write_text(
        json.dumps(split, indent=2) + "\n"
    )


def env_dir(var: str) -> Path:
    """Return the dataset folder inside the directory named by an nnU-Net environment variable."""
    root = os.environ.get(var)
    if not root:
        sys.exit(f"{var} is not set")
    return Path(root) / DATASET_NAME


def main() -> None:
    """Run the convert or splits subcommand."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser(
        "convert", help="write imagesTr, labelsTr and dataset.json into nnUNet_raw"
    )
    c.add_argument(
        "--data-dir", type=Path, default=REPO / "data/segthor_part1_corrected/train"
    )
    sub.add_parser("splits", help="write splits_final.json into nnUNet_preprocessed")
    args = p.parse_args()

    if args.command == "convert":
        ids = convert(args.data_dir, env_dir("nnUNet_raw"))
        print(f"converted {len(ids)} patients into {env_dir('nnUNet_raw')}")
    else:
        write_splits(env_dir("nnUNet_raw"), env_dir("nnUNet_preprocessed"))
        print(f"wrote {env_dir('nnUNet_preprocessed') / 'splits_final.json'}")


if __name__ == "__main__":
    main()
