"""Schematic of what the post-processing methods do, on made-up masks (no patient data), sized for a slide.

    python dataset_analysis/postprocessing_schematic.py [--out figures]

Four panels, each a prediction before and after one method: the largest connected component removes a stray piece,
the convex hull fills a notch of the heart, the majority vote of three seeds keeps what two of them agree on, and the
softmax average of three probability maps is labelled once. Writes postprocessing_schematic.{png,pdf}.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
from scipy.spatial import ConvexHull

from utils import REPO, pyplot

sys.path.insert(0, str(REPO / "tools"))
from plot_style import GRID_COLOR, tint  # noqa: E402
from postprocessing_figure import BOX_COLORS  # noqa: E402

SIZE = 40


def disc(cy: float, cx: float, ry: float, rx: float) -> np.ndarray:
    """Boolean ellipse on the SIZE x SIZE grid."""
    y, x = np.mgrid[:SIZE, :SIZE]
    return ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 4-connected piece of a boolean mask."""
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    return labels == 1 + int(np.argmax(sizes))


def convex_hull(mask: np.ndarray) -> np.ndarray:
    """Fill the convex hull of a boolean mask."""
    points = np.argwhere(mask)
    hull = points[ConvexHull(points).vertices]
    y, x = np.mgrid[:SIZE, :SIZE]
    inside = MplPath(hull).contains_points(np.c_[y.ravel(), x.ravel()], radius=1e-9)
    return inside.reshape(SIZE, SIZE) | mask


def majority(masks: list[np.ndarray]) -> np.ndarray:
    """Voxels labelled by more than half of the masks."""
    return np.sum(masks, axis=0) * 2 > len(masks)


def scenes() -> dict[str, tuple]:
    """The made-up inputs and the output of each method, as (before, after) pairs of boolean or float images."""
    organ = disc(20, 20, 11, 9)
    stray = disc(6, 33, 3, 3)
    heart = disc(20, 20, 13, 13) & ~disc(11, 29, 6, 6)
    seeds = [
        organ | disc(8, 31, 2.5, 2.5),
        disc(21, 21, 11, 9),
        disc(19, 19, 11, 10) | disc(33, 8, 2.5, 2.5),
    ]
    rng = np.random.default_rng(0)
    probabilities = [
        np.clip(
            ndimage.gaussian_filter(m.astype(float), 1.5)
            + rng.normal(0, 0.12, m.shape),
            0,
            1,
        )
        for m in seeds
    ]
    both = heart | disc(6, 33, 3, 3)
    return {
        "Largest component": (organ | stray, largest_component(organ | stray)),
        "Convex hull": (heart, convex_hull(heart)),
        "Filters combined": (both, convex_hull(largest_component(both))),
        "Majority vote": (seeds, majority(seeds)),
        "Softmax average": (probabilities, np.mean(probabilities, axis=0) > 0.5),
    }


def draw(ax, image: np.ndarray, color: str) -> None:
    """One mask, agreement map or probability map on a bare square axes."""
    ax.imshow(
        np.ma.masked_where(image <= 0, image.astype(float)),
        cmap=LinearSegmentedColormap.from_list("t", [tint(color, 0.25), color]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax.set_zorder(2)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(GRID_COLOR)


def figure(plt):
    """The five-panel schematic, one tinted panel per group of the post-processing figure."""
    fig = plt.figure(figsize=(11.9, 3.1))
    side = 0.82  # inches
    w, h = side / 11.9, side / 3.1
    groups = ["filter", "shape", "combo", "vote", "softmax"]
    captions = [
        ("prediction", "after"),
        ("prediction", "after"),
        ("prediction", "after"),
        ("3 seeds", "after"),
        ("mean of 3", "after"),
    ]
    step = 2.41 / 11.9  # the pitch of the cards below the picture on the slide
    panel_w = 2.26 / 11.9
    for k, ((name, (before, after)), group) in enumerate(zip(scenes().items(), groups)):
        fill, color = BOX_COLORS[group]
        px = k * step
        x0 = px + (panel_w - (2 * w + 0.026)) / 2
        if isinstance(before, list):
            before = np.mean(before, axis=0)
        fig.text(
            px + panel_w / 2, 0.78, name, ha="center", va="center", fontsize=13,
            fontweight="bold", color=color,
        )  # fmt: skip
        draw(fig.add_axes([x0, 0.3, w, h]), before, color)
        draw(fig.add_axes([x0 + w + 0.026, 0.3, w, h]), after, color)
        fig.text(
            px + panel_w / 2,
            0.3 + h / 2,
            "\u2192",
            ha="center",
            va="center",
            fontsize=20,
        )
        fig.text(x0 + w / 2, 0.2, captions[k][0], ha="center", va="center", fontsize=11)
        fig.text(
            x0 + w + 0.026 + w / 2,
            0.2,
            captions[k][1],
            ha="center",
            va="center",
            fontsize=11,
        )
    fig.text(
        0.985, 0.97, "What each method does (schematic, not patient data)", ha="right",
        va="top", fontsize=14, fontweight="bold",
    )  # fmt: skip
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=REPO / "figures")
    args = parser.parse_args(argv)
    fig = figure(pyplot())
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"postprocessing_schematic.{ext}")


if __name__ == "__main__":
    main()
