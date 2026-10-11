"""Effect of data augmentation on the 3D Dice and HD95 as a change against the control, sized for a slide.

    python dataset_analysis/augmentation_results.py [--runs docs/augmentation_results/per_run.csv] [--out figures]

One point per arm in the plane of the Dice change (right = higher) and the HD95 reduction (up = lower), so the upper right is
better on both. Every change is the mean over the (fold, seed) pairs the arm shares with its control, the same model without
augmentation. The single transforms of the ENet (Dice + CE) cover seed 0 only, i.e. 4 pairs, and are coloured by type, geometric (they
move the CT and the labels together) or intensity (CT only); the other arms cover 12 pairs (4 folds x 3 seeds) and are
coloured by recipe: joint (rotation +-10 degrees, scaling
0.9-1.1, shift up to 10%, p 0.5) or strong geometric (rotation +-15, scaling 0.8-1.2, shift 10%, p 0.7); diamonds add an elastic deformation (p 0.5). Arrows lead from the joint to the strong recipe and to the strong recipe trained for 50 instead of 25
epochs. The grey box is the seed std of the 12 control runs. Dice and HD95 are the 3D metrics on the original CT grid, mean over
the 4 organs and the 10 validation patients of a fold (one row of per_run.csv per fold and seed).
Writes augmentation_results.{png,pdf} and augmentation_results.csv (one row per point).
"""

import argparse
import csv
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from utils import CLASSES, INK, REPO, pyplot, spreads, write_csv
from plot_style import EARTH, tint  # tools/ is on the path once utils is imported

IMPROVED, IMPROVED_TEXT = (
    "#5E9142",
    "#3E6B2B",
)  # green of the region that is better on both measures
RUNS = REPO / "docs/augmentation_results/per_run.csv"
ENET, RESENC = "full_cv4_enet_dice_ce", "full_cv4_resenc_ds_unet_dice_ce"
GEOMETRIC, INTENSITY, JOINT, STRONG = "geometric", "intensity", "joint", "strong"
COLORS = {GEOMETRIC: EARTH[1], INTENSITY: "#B8862F", JOINT: EARTH[0], STRONG: "#6B3F7F"}
# label, experiment, reference, type, marker; the screening rows are seed 0 only
POINTS = [
    ("Rotation", f"{ENET}_rotation", ENET, GEOMETRIC, "o"),
    ("Scaling", f"{ENET}_scaling", ENET, GEOMETRIC, "o"),
    ("Shift", f"{ENET}_shift", ENET, GEOMETRIC, "o"),
    ("Shear", f"{ENET}_shear", ENET, GEOMETRIC, "o"),
    ("Gaussian noise", f"{ENET}_gaussian_noise", ENET, INTENSITY, "o"),
    ("Blur", f"{ENET}_blur", ENET, INTENSITY, "o"),
    ("Brightness/contrast", f"{ENET}_brightness_contrast", ENET, INTENSITY, "o"),
    ("Gamma", f"{ENET}_gamma", ENET, INTENSITY, "o"),
    ("Joint", f"{ENET}_augmented", ENET, JOINT, "o"),
    ("Joint + elastic", f"{ENET}_rot_scale_shift_elastic", ENET, JOINT, "D"),
    ("Strong", f"{ENET}_strong_geometric", ENET, STRONG, "o"),
    ("Strong + elastic", f"{ENET}_strong_geometric_elastic", ENET, STRONG, "D"),
    ("Strong, 50 epochs", f"{ENET}_strong_geometric_50ep", ENET, STRONG, "o"),
]
ARROWS = [  # from, to (labels)
    ("Joint", "Strong"),
    ("Strong", "Strong, 50 epochs"),
]
SIZE = (7.0, 3.3)  # inches, the slot on the slide
AXES = (0.95, 1.0, 5.7, 1.9)  # left, bottom, width and height of the plane in inches
XLIM, YLIM = (-0.0125, 0.040), (-3.4, 4.6)
# text offset in points and alignment of each label
LABELS = {
    "Rotation": ((-8, 0), "right"),
    "Scaling": ((0, 10), "center"),
    "Shift": ((-8, 0), "right"),
    "Shear": ((8, -10), "left"),
    "Gaussian noise": ((0, 11), "center"),
    "Blur": ((6, -14), "left"),
    "Brightness/contrast": ((-9, -2), "right"),
    "Gamma": ((0, 11), "center"),
    "Joint": ((0, -14), "center"),
    "Joint + elastic": ((9, -3), "left"),
    "Strong": ((0, -14), "center"),
    "Strong + elastic": ((-9, 0), "right"),
    "Strong, 50 epochs": ((0, 12), "center"),
}


def load_runs(path: Path = RUNS) -> dict[str, dict[tuple[int, int], dict]]:
    """Dice, HD95 and ASSD of every run, per experiment and (fold, seed).

    Args:
        path: per_run.csv of docs/augmentation_results.

    Returns:
        experiment -> (fold, seed) -> {"dice", "hd95", "assd"} (foreground) and the Dice of every organ by name.
    """
    runs: dict[str, dict] = {}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            runs.setdefault(r["experiment"], {})[int(r["fold"]), int(r["seed"])] = {
                "dice": float(r["dice_fg"]),
                "hd95": float(r["hd95_fg"]),
                "assd": float(r["assd_fg"]),
                **{organ: float(r[f"dice_{organ}"]) for organ in CLASSES.values()},
            }
    return runs


def pair_changes(arm: dict, reference: dict) -> list[tuple[float, float, float]]:
    """Change of every run two experiments share, ordered by (fold, seed).

    Args:
        arm: (fold, seed) -> metrics of the augmented experiment.
        reference: The same for its control.

    Returns:
        One (Dice change, HD95 reduction, ASSD reduction) per shared run; reductions are positive when the distance fell.
    """
    return [
        (
            arm[k]["dice"] - reference[k]["dice"],
            reference[k]["hd95"] - arm[k]["hd95"],
            reference[k]["assd"] - arm[k]["assd"],
        )
        for k in sorted(arm.keys() & reference.keys())
    ]


def paired_change(arm: dict, reference: dict) -> tuple[int, float, float]:
    """Mean change over the runs two experiments share.

    Args:
        arm: (fold, seed) -> metrics of the augmented experiment.
        reference: The same for its control.

    Returns:
        Number of shared runs, mean Dice change and mean HD95 reduction in mm (positive = lower HD95).
    """
    changes = pair_changes(arm, reference)
    n = len(changes)
    return n, sum(c[0] for c in changes) / n, sum(c[1] for c in changes) / n


def points(runs: dict) -> list[dict]:
    """The points of the figure with their paired changes.

    Args:
        runs: Result of load_runs().

    Returns:
        One dict per point: label, experiment, reference, type, marker, number of pairs, `dice_change`, `hd95_reduction`.
    """
    out = []
    for label, experiment, reference, kind, marker in POINTS:
        n, dice, hd95 = paired_change(runs[experiment], runs[reference])
        out.append(
            {"label": label, "experiment": experiment, "reference": reference, "kind": kind, "marker": marker, "n_pairs": n, "dice_change": dice, "hd95_reduction": hd95}
        )  # fmt: skip
    return out


def minus(values: list[float], signed: bool, fmt: str) -> list[str]:
    """Tick labels with a true minus sign (and a plus sign when `signed`)."""
    out = []
    for v in values:
        text = format(v, "+" + fmt if signed else fmt) if v else "0"
        out.append(text.replace("-", "\N{MINUS SIGN}"))
    return out


def draw(fig, runs: dict, colors: dict = COLORS, arrows: bool = True) -> None:
    """The change plane with its labels, the progression arrows (if `arrows`) and the key, in the given colours by type."""
    width, height = SIZE
    left, bottom, w, h = AXES
    ax = fig.add_axes([left / width, bottom / height, w / width, h / height])
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.add_patch(
        Rectangle((0, 0), XLIM[1], YLIM[1], color=tint(IMPROVED, 0.22), zorder=0, lw=0)
    )
    seed_dice, _ = spreads({k: v["dice"] for k, v in runs[ENET].items()})
    seed_hd, _ = spreads({k: v["hd95"] for k, v in runs[ENET].items()})
    ax.add_patch(
        Rectangle(
            (-seed_dice, -seed_hd),
            2 * seed_dice,
            2 * seed_hd,
            color="#CFCFCF",
            zorder=1,
            lw=0,
        )
    )
    ax.axhline(0, color=INK, lw=1, ls="--", zorder=2)
    ax.axvline(0, color=INK, lw=1, ls="--", zorder=2)
    pts = {p["label"]: p for p in points(runs)}
    for a, b in ARROWS if arrows else []:
        ax.annotate(
            "", (pts[b]["dice_change"], pts[b]["hd95_reduction"]), (pts[a]["dice_change"], pts[a]["hd95_reduction"]),
            arrowprops={"arrowstyle": "-|>", "color": "#777777", "lw": 1.2, "shrinkA": 7, "shrinkB": 7}, zorder=3,
        )  # fmt: skip
    for p in pts.values():
        color = colors[p["kind"]]
        bold = p["kind"] in (JOINT, STRONG)
        ax.scatter(
            p["dice_change"], p["hd95_reduction"], s=80, marker=p["marker"],
            facecolor=color, edgecolor=color, linewidth=2, zorder=4,
        )  # fmt: skip
        if LABELS[p["label"]] is None:
            continue
        offset, align = LABELS[p["label"]]
        ax.annotate(
            p["label"].strip().replace("/", "/\n"), (p["dice_change"], p["hd95_reduction"]), offset, textcoords="offset points",
            color=color, fontsize=11, ha=align, va="center", fontweight="bold" if bold else "normal",
        )  # fmt: skip
    xticks, yticks = [-0.01, 0, 0.01, 0.02, 0.03, 0.04], [-3, -2, -1, 0, 1, 2, 3, 4]
    ax.set_xticks(xticks, minus(xticks, True, ".2f"))
    ax.set_xticks([-0.005, 0.005, 0.015, 0.025, 0.035], minor=True)
    ax.set_yticks(yticks, minus(yticks, True, "d"))
    ax.grid(True, which="both", axis="x", color="#D9D9D9", lw=0.7, ls=(0, (3, 3)))
    ax.grid(True, axis="y", color="#D9D9D9", lw=0.7, ls=(0, (3, 3)))
    ax.set_xlabel("Dice change (higher is better)", fontsize=11)
    ax.set_ylabel("HD95 reduction, mm", fontsize=11)
    ax.tick_params(labelsize=11, length=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        XLIM[1] - 0.0005,
        0.15,
        "better on both",
        fontsize=11,
        ha="right",
        va="bottom",
        color=IMPROVED_TEXT,
        style="italic",
    )

    def key(kind, label, marker="o"):
        """Legend entry: a marker in the colour of a type, or grey when `kind` is None."""
        color = colors[kind] if kind else INK
        return Line2D(
            [], [], marker=marker, ls="", ms=8, mew=2, mfc=color, mec=color, label=label
        )

    handles = [
        key(GEOMETRIC, "Geometric"),
        key(INTENSITY, "Intensity"),
        key(JOINT, "Joint recipe"),
        key(STRONG, "Strong recipe"),
        key(None, "+ elastic deformation", "D"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=11,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.3,
        bbox_to_anchor=(0.5, 0.0),
    )


def table(runs: dict) -> list[dict]:
    """One row per point, for the csv."""
    return [
        {
            "point": p["label"].strip(), "type": p["kind"],
            "experiment": p["experiment"], "n_pairs": p["n_pairs"],
            "dice_change": round(p["dice_change"], 4), "hd95_reduction": round(p["hd95_reduction"], 3),
        }
        for p in points(runs)
    ]  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--out", type=Path, default=REPO / "figures")
    args = parser.parse_args(argv)

    runs = load_runs(args.runs)
    plt = pyplot()
    fig = plt.figure(figsize=SIZE)
    draw(fig, runs)
    fig.text(
        1 - 0.05 / SIZE[0],
        1 - 0.05 / SIZE[1],
        "Effect of augmentation on Dice and HD95 of the ENet",
        ha="right",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"augmentation_results.{ext}", dpi=300)
    write_csv(args.out / "augmentation_results.csv", table(runs))


if __name__ == "__main__":
    main()
