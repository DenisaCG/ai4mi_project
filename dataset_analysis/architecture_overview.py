"""Dice of every architecture against the upgraded baseline, one row per architecture, sized for a slide.

    python dataset_analysis/architecture_overview.py [--out figures]

Left panel: Dice averaged over the four organs. Right panel: Dice of the esophagus. Both are the mean over the 12 runs
(4 folds x 3 seeds; per run, the mean over the 10 validation patients, 3D Dice on the original CT grid). The shaded
bands around the upgraded baseline are its standard deviation over seeds (dark) and over folds (light), computed as in
the cross-validation summary; the first column gives the trainable parameters. The runs of the frozen-encoder U-Net
and the 2.5D ResEnc U-Net are not stored locally, so their runs are read from the Snellius inventory
(deck/snellius_inventory).
Writes architecture_overview.{png,pdf} and architecture_overview.csv (also HD95 and ASSD).
"""

import argparse
from pathlib import Path

from utils import (
    COLORS,
    INK,
    REPO,
    load_arch_runs,
    pyplot,
    read_csv,
    summarise,
    tint,
    write_csv,
)

ESOPHAGUS = COLORS[1]
BASELINE = "full_cv4_enet_dice_ce"
DINO = "full_cv4_dino_unet_dice_ce"

# (row label, experiment), top to bottom
ARCHITECTURES = [
    ("Upgraded baseline", BASELINE),
    ("+ 2.5D ±1 slice", "full_cv4_enet_dice_ce_25d_c1"),
    ("+ 2.5D ±2 slices", "full_cv4_enet_dice_ce_25d_c2"),
    ("U-Net", "full_cv4_unet_dice_ce"),
    ("+ attention gates", "full_cv4_attention_unet_dice_ce"),
    ("+ deep supervision", "full_cv4_ds_unet_dice_ce"),
    ("+ frozen DINOv2", DINO),
    ("ResEnc U-Net", "full_cv4_resenc_unet_dice_ce"),
    ("+ deep supervision", "full_cv4_resenc_ds_unet_dice_ce"),
    ("+ 2.5D ±1 slice", "full_cv4_resenc_ds_unet_dice_ce_25d_c1"),
]
NO_PARAMETER_COUNT = "frozen ViT-S/14\nencoder"


def collect(metrics_dir: Path, parameters: Path) -> list[dict]:
    """One result row per architecture, in figure order."""
    counts = {
        r["experiment"]: int(r["trainable_parameters"]) for r in read_csv(parameters)
    }
    table = []
    for label, experiment in ARCHITECTURES:
        rows, source = load_arch_runs(metrics_dir, experiment)
        stats = summarise(rows)
        table.append(
            {
                "architecture": label.replace("\n", " "),
                "experiment": experiment,
                "source": source,
                "trainable_parameters": counts.get(experiment),
                **stats,
            }
        )
    return table


def parameter_text(count: int | None) -> str:
    """Trainable parameters in millions, or the note for the frozen-encoder model."""
    return NO_PARAMETER_COUNT if count is None else f"{count / 1e6:.2f} M"


def draw_panel(
    ax, table: list[dict], key: str, std_prefix: str, color: str, title: str
) -> None:
    """Draw one dot per architecture and the baseline's seed and fold bands."""
    base = table[0]
    ax.axvspan(
        base[key] - base[f"{std_prefix}_fold_std"],
        base[key] + base[f"{std_prefix}_fold_std"],
        color=tint(color, 0.14),
        lw=0,
    )
    ax.axvspan(
        base[key] - base[f"{std_prefix}_seed_std"],
        base[key] + base[f"{std_prefix}_seed_std"],
        color=tint(color, 0.4),
        lw=0,
    )
    ax.axvline(base[key], color=color, lw=1.0, ls="--", zorder=1)
    top = len(table) - 1
    for i, row in enumerate(table):
        y = top - i
        ax.scatter(row[key], y, s=70, zorder=3, color=color, edgecolor="white", lw=0.8)
        ax.text(
            row[key] + 0.006,
            y,
            f"{row[key]:.3f}",
            va="center",
            fontsize=11,
            color=INK,
            zorder=4,
        )
    ax.set_title(title, fontsize=12, fontweight="bold")


def style_panel(ax, lo: float, hi: float) -> None:
    """Shared axis look: dashed grid at each tick, a few minor lines, no spines other than the bottom."""
    ax.set_xlim(lo, hi)
    ax.set_xticks(
        [round(lo + 0.04 * k, 2) for k in range(5)]
    )  # last tick sits 0.02 inside the limit
    ax.set_xticks([round(lo + 0.02 * k, 2) for k in range(9)], minor=True)
    ax.grid(axis="x", which="major", color="#C8C8C8", lw=0.8)
    ax.grid(axis="x", which="minor", color="#E4E4E4", lw=0.6, ls=(0, (2, 2)))
    ax.grid(axis="y", color="#E4E4E4", lw=0.6, ls=(0, (2, 2)))
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="x", which="minor", length=2)
    ax.tick_params(axis="y", length=0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metrics-dir", type=Path, default=REPO / "metrics")
    parser.add_argument(
        "--parameters", type=Path, default=REPO / "figures/model_parameters.csv"
    )
    parser.add_argument("--out", type=Path, default=REPO / "figures")
    args = parser.parse_args(argv)

    table = collect(args.metrics_dir, args.parameters)
    plt = pyplot()
    n = len(table)
    fig, axs = plt.subplots(
        1,
        3,
        figsize=(7.9, 4.85),
        sharey=True,
        gridspec_kw={"width_ratios": [1.25, 2.3, 2.3], "wspace": 0.12},
    )
    ax_par, ax_mean, ax_eso = axs
    draw_panel(ax_mean, table, "dice", "dice", INK, "Dice, mean of\nfour organs")
    draw_panel(
        ax_eso, table, "esophagus_dice", "esophagus", ESOPHAGUS, "Dice,\nesophagus"
    )
    style_panel(ax_mean, 0.76, 0.94)
    style_panel(ax_eso, 0.62, 0.80)
    ax_par.set_title("Trainable\nparameters", fontsize=12, fontweight="bold")
    for i, row in enumerate(table):
        ax_par.text(
            0.5,
            n - 1 - i,
            parameter_text(row["trainable_parameters"]),
            ha="center",
            va="center",
            fontsize=11,
            color=INK,
        )
    ax_par.set_xlim(0, 1)
    ax_par.set_ylim(-0.6, n - 0.4)
    ax_par.set_xticks([])
    ax_par.grid(axis="y", color="#E4E4E4", lw=0.6, ls=(0, (2, 2)))
    ax_par.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax_par.set_yticks(range(n), [r[0] for r in reversed(ARCHITECTURES)], fontsize=11)
    ax_par.tick_params(axis="y", length=0)

    handles = [
        plt.Line2D(
            [], [], marker="o", color=INK, ls="", markersize=8, label="mean of 12 runs"
        ),
        plt.Rectangle(
            (0, 0), 1, 1, color=tint(INK, 0.4), lw=0, label="baseline ± seed std"
        ),
        plt.Rectangle(
            (0, 0), 1, 1, color=tint(INK, 0.14), lw=0, label="baseline ± fold std"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=11,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.0,
    )
    fig.supxlabel("Dice (3D, original CT grid)", fontsize=12, y=0.1)
    fig.subplots_adjust(left=0.27, right=0.985, top=0.83, bottom=0.2)
    fig.text(
        0.012,
        0.5 * (0.83 + 0.2),
        "Architecture",
        rotation=90,
        ha="left",
        va="center",
        fontsize=12,
    )
    fig.text(
        ax_eso.get_position().x1,
        0.99,
        "Architectures against the upgraded baseline",
        ha="right",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"architecture_overview.{ext}")
    write_csv(
        args.out / "architecture_overview.csv",
        [
            {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}
            for row in table
        ],
    )


if __name__ == "__main__":
    main()
