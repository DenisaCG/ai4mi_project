"""Real failure cases behind the post-processing candidates, and the schematic strip of what each method does.

    python dataset_analysis/postprocessing_cases.py [--out figures]

postprocessing_cases.png: one tile per validation case from deck/snellius_inventory/deficiencies_scur0049,
cropped to its CT picture, the prediction shaded red and the ground truth outlined in blue. The tiles carry no
text: the slide says in each card which case the picture is. Each tile sits in the column of the method it
motivates.
postprocessing_strip.png: the shapes of figures/postprocessing_schematic.png without its titles and captions.
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage

from postprocessing_schematic import BASELINE, HEIGHT_IN as SCHEMATIC_HEIGHT_IN, SIDE_IN
from utils import REPO, pyplot

sys.path.insert(0, str(REPO / "tools"))
from plot_style import apply_style  # noqa: E402

INVENTORY = REPO / "deck/snellius_inventory/deficiencies_scur0049"
ENET, RESENC = "full_cv4_enet_dice_ce", "full_cv4_resenc_ds_unet_dice_ce"
# (schematic column, experiment, fold, seed, patient, organ, window)
# window = (centre x, centre y, half width, half height) in pixels of the saved 630 x 630 case image: a wide, short
# rectangle around the issue
CASES = [
    (0, RESENC, 0, 0, "Patient_30", "heart", (280, 425, 175, 125)),
    (1, ENET, 0, 2, "Patient_30", "heart", (330, 310, 210, 150)),
]
WIDTH_IN, TILE_W, TILE_H = 11.9, 2.2, 1.5
PREDICTION_RGB = (255, 45, 45)
CASE_SCALE = 0.85  # size of the axial pictures relative to the tile of the coronal ones
TRUTH_RGB = (0, 190, 255)
MIN_OUTLINE_PIXELS = (
    150  # a ground-truth outline is one long line; bone leaves small specks
)


YELLOW_RGB = (255, 210, 0)
SNAPSHOTS = REPO / "deck/snellius_inventory"
# (schematic column, folder of the seed volumes, CT, coronal plane, centre column, {seed: colour}): the seed that goes
# wrong is shaded red, the one that does not is outlined yellow
SEED_CASES = [
    (
        2,
        SNAPSHOTS / "p08_seeds",
        REPO / "data/segthor_part1_corrected/train/Patient_08/Patient_08.nii.gz",
        236,
        262,
        {1: PREDICTION_RGB, 0: YELLOW_RGB},
    ),
    (
        3,
        SNAPSHOTS / "p30_seeds",
        SNAPSHOTS / "p30_seeds/ct.nii.gz",
        265,
        275,
        {0: PREDICTION_RGB, 2: YELLOW_RGB},
    ),
]


def crop_picture(path: Path, window: tuple[int, int, int, int]) -> np.ndarray:
    """Crops a saved case image to a wide rectangle around the issue, without the axis labels around the CT picture.

    Args:
        path: PNG written by the deficiencies analysis (picture inside a white margin with axis labels).
        window: Centre x, centre y, half width and half height of the rectangle, in pixels of the saved image.

    Returns:
        The rectangle as an RGB array.
    """
    img = np.asarray(Image.open(path).convert("RGB"))
    inked = img.min(axis=2) < 250
    rows = np.flatnonzero(inked.mean(axis=1) > 0.6)
    cols = np.flatnonzero(inked.mean(axis=0) > 0.6)
    cx, cy, half_w, half_h = window
    assert (
        rows[0] <= cy - half_h
        and cy + half_h <= rows[-1]
        and cols[0] <= cx - half_w
        and cx + half_w <= cols[-1]
    ), path
    return emphasise(img[cy - half_h : cy + half_h, cx - half_w : cx + half_w])


def emphasise(pic: np.ndarray) -> np.ndarray:
    """Thickens and shades the outlines of a case picture so they read when shown small.

    The CT is grey, so every coloured pixel is the prediction outline; the ground-truth outline is the thin white
    line (bone is white too but in blobs, which an opening keeps). Both outlines are thickened and drawn solid,
    and what a closed outline encloses is tinted (prediction red, ground truth blue).

    Args:
        pic: RGB case picture, grey CT with a white ground-truth outline and a coloured prediction outline.

    Returns:
        The picture with the prediction shaded red and the ground truth outlined in blue.
    """
    white = pic.min(axis=2) > 235
    thin_white = white & ~ndimage.binary_opening(white, np.ones((5, 5)))
    pieces, _ = ndimage.label(thin_white, np.ones((3, 3)))
    thin_white = np.isin(
        pieces, np.flatnonzero(np.bincount(pieces.ravel()) >= MIN_OUTLINE_PIXELS)[1:]
    )  # drop bone specks
    coloured = pic.max(axis=2).astype(int) - pic.min(axis=2) > 40
    outlines = (
        (ndimage.binary_dilation(thin_white, iterations=3), TRUTH_RGB),
        (ndimage.binary_dilation(coloured, iterations=2), PREDICTION_RGB),
    )
    out = pic.astype(float)
    for line, rgb in outlines:
        inside = ndimage.binary_fill_holes(line)
        out[inside] = 0.6 * out[inside] + 0.4 * np.array(rgb)
        out[line] = rgb
    return out.astype(np.uint8)


def seed_comparison(
    ax, seeds: Path, ct_path: Path, plane: int, centre: int, colors: dict[int, tuple]
) -> None:
    """Draws the heart labels of two seeds of one patient on the same coronal CT slice.

    Args:
        ax: Axes to draw into.
        seeds: Folder holding seed<k>.nii.gz, the heart labels (value 2) of each seed on the CT grid.
        ct_path: CT of the patient.
        plane: Index of the coronal plane to draw.
        centre: Voxel column the picture is centred on.
        colors: Colour of each seed; the one given red is shaded, the other outlined.
    """
    ct = nib.load(ct_path)
    dx, _, dz = (float(z) for z in ct.header.get_zooms())
    columns = slice(
        centre - 180, centre + 180
    )  # as wide as the rows below are tall, in mm
    rows = slice(34, 171)  # from the heart to the top of the scan
    slab = np.asarray(ct.dataobj)[columns, plane].T[rows].astype(float)
    ax.imshow(
        np.clip((slab + 160) / 400, 0, 1),
        cmap="gray",
        origin="lower",
        aspect=dz / dx,
        vmin=0,
        vmax=1,
    )
    for seed, rgb in colors.items():
        mask = np.asarray(nib.load(seeds / f"seed{seed}.nii.gz").dataobj) == 2
        pieces, _ = ndimage.label(mask)
        main = pieces == 1 + np.argmax(np.bincount(pieces.ravel())[1:])
        # a piece away from the heart can be a single slice thick: one slice each side keeps it visible when small
        heart = (main | ndimage.binary_dilation(mask & ~main, np.ones((1, 1, 3))))[
            columns, plane
        ].T[rows]
        edge = heart & ~ndimage.binary_erosion(heart, iterations=3)
        overlay = np.zeros((*heart.shape, 4))
        overlay[edge] = (*(c / 255 for c in rgb), 1)
        if rgb == PREDICTION_RGB:
            overlay[heart & ~edge] = (*(c / 255 for c in rgb), 0.35)
        ax.imshow(overlay, origin="lower", aspect=dz / dx)
    ax.axis("off")


def main() -> None:
    """Draws the two figures and writes them."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=REPO / "figures")
    args = parser.parse_args()

    apply_style()
    plt = pyplot()
    fig = plt.figure(figsize=(WIDTH_IN, TILE_H))
    for column, exp, fold, seed, patient, organ, window in CASES:
        ax = fig.add_axes(
            [
                ((column + 0.5) * WIDTH_IN / 4 - TILE_W * CASE_SCALE / 2) / WIDTH_IN,
                1 - CASE_SCALE,
                TILE_W * CASE_SCALE / WIDTH_IN,
                CASE_SCALE,
            ]
        )
        ax.imshow(
            crop_picture(
                INVENTORY
                / "cases"
                / f"{exp}_fold{fold}_seed{seed}_{patient}_{organ}.png",
                window,
            )
        )
        ax.axis("off")
    for column, *case in SEED_CASES:
        ax = fig.add_axes(
            [
                ((column + 0.5) * WIDTH_IN / 4 - TILE_W / 2) / WIDTH_IN,
                0,
                TILE_W / WIDTH_IN,
                1,
            ]
        )
        seed_comparison(ax, *case)
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "postprocessing_cases.png", dpi=200, facecolor="white")
    schematic = Image.open(args.out / "postprocessing_schematic.png").convert("RGB")
    top = round(
        schematic.height * (1 - BASELINE - SIDE_IN / SCHEMATIC_HEIGHT_IN - 0.02)
    )  # panels, a margin above
    bottom = round(schematic.height * (1 - BASELINE + 0.04))
    schematic.crop((0, top, schematic.width, bottom)).save(
        args.out / "postprocessing_strip.png"
    )
    print(f"wrote {args.out / 'postprocessing_cases.png'} and postprocessing_strip.png")


if __name__ == "__main__":
    main()
