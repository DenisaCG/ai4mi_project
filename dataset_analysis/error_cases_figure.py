"""Three real prediction errors of the upgraded baseline, sized for a slide.

    python dataset_analysis/error_cases_figure.py [--out figures]

The existing renders of the Snellius analysis (deck/snellius_inventory/deficiencies_scur0049/cases), cropped and
re-labelled, not re-rendered: white outline = ground truth, coloured outline = prediction of the upgraded baseline
(ENet, Dice + CE, full pre-processing). Under each case the Dice and HD95 of that organ, patient and seed (3D, original
CT grid, best checkpoint). Writes error_cases.{png,pdf}.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from utils import CLASSES, REPO, load_cv_runs, pyplot, read_csv

sys.path.insert(0, str(REPO / "tools"))
from plot_style import EARTH_LABEL_COLORS  # noqa: E402

EXPERIMENT = "full_cv4_enet_dice_ce"
CASE_DIR = REPO / "deck/snellius_inventory/deficiencies_scur0049"
ORGAN_COLORS = {name: EARTH_LABEL_COLORS[idx] for idx, name in CLASSES.items()}
# (fold, seed, patient, organ): chosen after viewing the renders of every ENet case with HD95 above 50 mm and of the
# lowest-Dice esophagus.
CASES = [
    (1, 1, "Patient_24", "esophagus"),
    (0, 0, "Patient_02", "trachea"),
    (3, 1, "Patient_03", "aorta"),
]
LABEL_PX = 44  # height of the label box baked into the top of every render
FIG_W, FIG_H = 11.9, 3.0  # inches, the slide slot
CELL_W, CELL_GAP = 3.9, 0.1
IMAGE_H = 2.45


def crop_render(img: np.ndarray, label_px: int = LABEL_PX) -> np.ndarray:
    """Cut a case render down to its image area, without axis labels or the baked-in label box.

    Args:
        img: RGB render, white background, the image area framed by a dark box.
        label_px: Rows to drop from the top of the image area (the label box).

    Returns:
        The image area without its one-pixel frame and without the top `label_px` rows.
    """
    ink = (img < 250).any(axis=2)
    rows, cols = ink.sum(axis=1), ink.sum(axis=0)
    r = np.flatnonzero(rows > 0.5 * rows.max())
    c = np.flatnonzero(cols > 0.5 * cols.max())
    return img[r[0] + 1 + label_px : r[-1], c[0] + 1 : c[-1]]


def collect_cases(rows: list[dict], case_dir: Path = CASE_DIR) -> list[dict]:
    """The rendered cases with their slice, Dice and HD95.

    Args:
        rows: Validation rows (Dice, HD95) from load_cv_runs.
        case_dir: Folder with cases.csv and the case renders.

    Returns:
        One dict per case: patient, seed, organ, slice, dice, hd95 and the render path.
    """
    slices = {
        (int(r["seed"]), r["patient"], r["organ"]): int(
            r["slice_index_superior_inferior"]
        )
        for r in read_csv(case_dir / "cases.csv")
        if r["experiment"] == EXPERIMENT
    }
    cases = []
    for fold, seed, patient, organ in CASES:
        row = next(
            r for r in rows
            if (r["fold"], r["seed"], r["patient"], r["organ"]) == (fold, seed, patient, organ)
        )  # fmt: skip
        cases.append(
            {
                "fold": fold, "seed": seed, "patient": patient, "organ": organ,
                "slice": slices[seed, patient, organ], "dice": row["dice"], "hd95": row["hd95"],
                "png": case_dir / "cases" / f"{EXPERIMENT}_fold{fold}_seed{seed}_{patient}_{organ}.png",
            }
        )  # fmt: skip
    return cases


def draw_case(fig, left, case) -> None:
    """Cropped render with its label inside and its Dice and HD95 below, in a column starting at `left` inches."""
    ax = fig.add_axes(
        [left / FIG_W, 1 - IMAGE_H / FIG_H, CELL_W / FIG_W, IMAGE_H / FIG_H]
    )
    ax.imshow(
        crop_render(np.asarray(Image.open(case["png"]).convert("RGB"))),
        interpolation="lanczos",
    )
    ax.axis("off")
    ax.set_anchor("NW")
    ax.text(
        0.02, 0.98, f"{case['patient'].replace('_', ' ')}, seed {case['seed']}, slice {case['slice']}",
        transform=ax.transAxes, va="top", ha="left", fontsize=11, color="white",
        bbox={"facecolor": "black", "alpha": 0.6, "pad": 2, "lw": 0},
    )  # fmt: skip
    fig.text(
        (left + 0.02) / FIG_W, 1 - (IMAGE_H + 0.1) / FIG_H,
        f"{case['organ'].capitalize()}: Dice {case['dice']:.2f}, HD95 {case['hd95']:.0f} mm",
        fontsize=12, va="top", color=ORGAN_COLORS[case["organ"]], fontweight="bold",
    )  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metrics-dir", type=Path, default=REPO / "metrics")
    parser.add_argument("--out", type=Path, default=REPO / "figures")
    args = parser.parse_args(argv)

    plt = pyplot()
    rows = load_cv_runs(args.metrics_dir, EXPERIMENT, keys=("dice", "hd95"))
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    for i, case in enumerate(collect_cases(rows)):
        draw_case(fig, i * (CELL_W + CELL_GAP), case)
    fig.text(
        0.0, 0.0, "White outline: ground truth. Coloured outline: prediction.",
        fontsize=11, va="bottom", color="#333333",
    )  # fmt: skip
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"error_cases.{ext}")


if __name__ == "__main__":
    main()
