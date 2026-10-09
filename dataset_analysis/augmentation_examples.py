"""One axial CT slice under each transform of the augmentation recipe, with the ground-truth outline, sized for a slide.

    python dataset_analysis/augmentation_examples.py [--data-dir data/segthor_part1_corrected/train] [--out figures]

The slice is the one, over all patients of the corrected release, where the smallest of the four organs is largest (every organ
present). It is shown unchanged, under rotation, scaling and shift alone, under the three in sequence (joint recipe) and under the strong recipe, each drawn with
the transform of src/augment.py (bilinear CT, nearest-neighbour labels, background outside the source) at probability 1 and a
fixed seed. Ranges: rotation +-10 degrees, scaling 0.9-1.1 and shift up to 10% of the crop for the single transforms and the joint recipe; +-15, 0.8-1.2 and 10% for the strong recipe. The
local src/augment.py has no shift, so it is written here with the same interpolation and fill. The CT is shown in the
-160 to 240 HU window, cropped to 288 x 288 pixels around the organs of the slice; the fill is the lower end of that window.
Writes augmentation_examples.{png,pdf} and augmentation_examples.csv (the draws behind every panel).
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib import patheffects
from matplotlib.lines import Line2D
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import affine

from utils import CLASSES, REPO, pyplot, write_csv

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "dataset_analysis/profile_figures"))
from label_correction_example import WINDOW, load_patient  # noqa: E402
from plot_style import EARTH_LABEL_COLORS  # noqa: E402
from augmentation_results import COLORS, GEOMETRIC, JOINT, STRONG  # noqa: E402
from src.augment import build_random_rotation, build_random_scaling  # noqa: E402

CROP = 288  # side of the slice crop in pixels, as the ROI crop of the pipeline
P = 1.0  # every transform fires, so the panel shows the draw
JOINT_RANGES = (
    [-10.0, 10.0],
    [0.9, 1.1],
    0.1,
)  # rotation degrees, scaling, shift fraction of the joint recipe
STRONG_RANGES = ([-15.0, 15.0], [0.8, 1.2], 0.1)  # the strong geometric recipe
SEED = 4  # fixed seed of every panel
PANEL, GAP, KEY = 1.5, 0.1, 0.3  # panel side, gap and height of the organ key in inches
INK = "#333333"


def best_slice(gt: np.ndarray) -> tuple[int, int]:
    """Axial slice where the smallest of the four organs is largest.

    Args:
        gt: Label volume (x, y, z).

    Returns:
        Slice index and the area in pixels of the smallest organ on it (0 if no slice shows all four).
    """
    areas = np.stack([(gt == k).sum(axis=(0, 1)) for k in CLASSES])
    smallest = areas.min(axis=0)
    return int(smallest.argmax()), int(smallest.max())


def random_shift(p: float, max_fraction: float, fill: float):
    """Translation of the CT and labels together by up to a fraction of the image side along each axis.

    Args:
        p: Probability that the shift is applied.
        max_fraction: Largest shift as a fraction of the image side.
        fill: CT value outside the source.

    Returns:
        Function mapping (image, one-hot labels) to the shifted pair.
    """

    def shift(image, gt):
        if random.random() >= p:
            return image, gt
        h, w = image.shape[-2:]
        move = [
            round(random.uniform(-max_fraction, max_fraction) * w),
            round(random.uniform(-max_fraction, max_fraction) * h),
        ]
        params = {"angle": 0.0, "translate": move, "scale": 1.0, "shear": [0.0, 0.0]}
        image = affine(
            image, **params, interpolation=InterpolationMode.BILINEAR, fill=[fill]
        )
        gt = affine(
            gt,
            **params,
            interpolation=InterpolationMode.NEAREST,
            fill=[1.0] + [0.0] * (gt.shape[0] - 1),
        )
        return image, gt

    return shift


def load_slice(
    data_dir: Path, patient: str, z: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Windowed CT and one-hot labels of one slice, cropped around the organs.

    Args:
        data_dir: Folder with one Patient_* folder per patient.
        patient: Patient folder name.
        z: Axial slice index.

    Returns:
        CT (1, CROP, CROP) in [0, 1] and one-hot labels (5, CROP, CROP); rows run anterior to posterior.
    """
    gt, ct, _ = load_patient(data_dir, patient)
    labels, image = gt[:, :, z].T, ct[:, :, z].T
    rows, cols = np.nonzero(labels)
    top = int(np.clip(rows.mean() - CROP / 2, 0, labels.shape[0] - CROP))
    left = int(np.clip(cols.mean() - CROP / 2, 0, labels.shape[1] - CROP))
    box = (slice(top, top + CROP), slice(left, left + CROP))
    image = np.clip((image[box] - WINDOW[0]) / (WINDOW[1] - WINDOW[0]), 0, 1)
    onehot = np.stack([labels[box] == k for k in (0, *CLASSES)]).astype(np.float32)
    return torch.from_numpy(image[None].astype(np.float32)), torch.from_numpy(onehot)


def apply(
    steps: list, image: torch.Tensor, onehot: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Apply transforms in order under the fixed seed.

    Args:
        steps: Transforms taking and returning (image, labels).
        image: CT (1, H, W).
        onehot: One-hot labels (C, H, W).

    Returns:
        The transformed CT and labels, and every value the transforms drew with random.uniform, in order.
    """
    draws, uniform = [], random.uniform

    def record(a, b):
        draws.append(uniform(a, b))
        return draws[-1]

    random.seed(SEED)
    random.uniform = record
    try:
        for step in steps:
            image, onehot = step(image, onehot)
    finally:
        random.uniform = uniform
    return image, onehot, draws


def caption(name: str, draws: list[float]) -> str:
    """Panel caption with the values the transform drew; the sequence of all three has none."""
    if name == "Rotation":
        return f"{name} {draws[0]:+.1f}\N{DEGREE SIGN}"
    if name == "Scaling":
        return f"{name} \N{MULTIPLICATION SIGN}{draws[0]:.2f}"
    if name == "Shift":
        return f"{name} {draws[0] * CROP:+.0f}, {draws[1] * CROP:+.0f} px"
    if name == "Joint":
        return "Rot. + scal. + shift"
    return "Strong recipe"


def draw_slice(
    ax, image: torch.Tensor, onehot: torch.Tensor, text: str, box: str = "black"
) -> None:
    """CT slice with the outline of every organ and a caption box of the given colour at the bottom left."""
    ax.imshow(image[0], cmap="gray", vmin=0, vmax=1)
    for k in CLASSES:
        ax.contour(
            onehot[k], levels=[0.5], colors=[EARTH_LABEL_COLORS[k]], linewidths=1.6
        )
    ax.text(
        0.03, 0.03, text, transform=ax.transAxes, color="white", fontsize=10, ha="left", va="bottom", linespacing=1.25,
        bbox={"boxstyle": "round,pad=0.2", "fc": box, "ec": "none", "alpha": 0.8 if box != "black" else 0.55},
    )  # fmt: skip
    ax.set_axis_off()


def draw_key(fig, first) -> None:
    """Organ colours as a row under the panels, and the anatomical directions on the first panel."""
    handles = [
        Line2D([], [], color=EARTH_LABEL_COLORS[k], lw=3, label=name)
        for k, name in CLASSES.items()
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=4, fontsize=11, frameon=False, handlelength=1.2, handletextpad=0.4,
        columnspacing=1.0, bbox_to_anchor=(0.5, 0.0),
    )  # fmt: skip
    outline = [patheffects.withStroke(linewidth=2, foreground="black")]
    for text, xy, ha, va, rotation in [
        ("anterior", (0.5, 0.98), "center", "top", 0),
        ("right", (0.02, 0.55), "left", "center", 90),
        ("left", (0.98, 0.55), "right", "center", 270),
    ]:
        first.text(
            *xy,
            text,
            transform=first.transAxes,
            color="white",
            fontsize=10,
            ha=ha,
            va=va,
            rotation=rotation,
            path_effects=outline,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir", type=Path, default=REPO / "data/segthor_part1_corrected/train"
    )
    parser.add_argument("--out", type=Path, default=REPO / "figures")
    args = parser.parse_args(argv)

    best = {
        f.name: best_slice(load_patient(args.data_dir, f.name)[0])
        for f in sorted(args.data_dir.glob("Patient_*"))
    }
    patient = max(best, key=lambda name: best[name][1])
    z, smallest = best[patient]
    print(f"{patient} slice {z}: smallest organ {smallest} pixels")

    image, onehot = load_slice(args.data_dir, patient, z)
    chains = {}
    for name, (degrees, scales, shift) in (
        ("Joint", JOINT_RANGES),
        ("Strong", STRONG_RANGES),
    ):
        chains[name] = [
            build_random_rotation(P, degrees, 0.0),
            build_random_scaling(P, scales, 0.0),
            random_shift(P, shift, 0.0),
        ]
    rotation, scaling, shift = chains["Joint"]
    plt = pyplot()
    fig = plt.figure(figsize=(3 * PANEL + 2 * GAP, 2 * PANEL + GAP + KEY))
    width, height = fig.get_size_inches()
    spots = [
        (c * (PANEL + GAP), (1 - r) * (PANEL + GAP) + KEY)
        for r in range(2)
        for c in range(3)
    ]
    axes = [
        fig.add_axes([x / width, y / height, PANEL / width, PANEL / height])
        for x, y in spots
    ]

    draw_slice(axes[0], image, onehot, f"Original\nP{patient.split('_')[1]}, slice {z}")
    rows = [{"panel": "Original", "patient": patient, "slice": z, "draws": ""}]
    panels = [
        ("Rotation", [rotation]),
        ("Scaling", [scaling]),
        ("Shift", [shift]),
        ("Joint", chains["Joint"]),
        ("Strong", chains["Strong"]),
    ]
    for ax, (name, steps) in zip(axes[1:], panels):
        moved, labels, draws = apply(steps, image, onehot)
        draw_slice(
            ax,
            moved,
            labels,
            caption(name, draws),
            COLORS[{"Joint": JOINT, "Strong": STRONG}.get(name, GEOMETRIC)],
        )
        rows.append(
            {
                "panel": name,
                "patient": patient,
                "slice": z,
                "draws": " ".join(f"{d:.4f}" for d in draws),
            }
        )
    draw_key(fig, axes[0])

    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"augmentation_examples.{ext}", dpi=300)
    write_csv(args.out / "augmentation_examples.csv", rows)


if __name__ == "__main__":
    main()
