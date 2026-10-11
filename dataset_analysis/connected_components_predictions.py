#!/usr/bin/env python3
"""
Connected components per label for model predictions, next to the ground truth: the same heatmaps as
dataset_analysis/profile_figures/connected_components.py (figures/comparison/connected_components.png), one row
per source.

Rows (each can be pooled over several runs, e.g. folds x seeds; every prediction volume counts once):
  Ground truth                   data/segthor_train_full/train/Patient_*/GT.nii.gz (corrected labels, 40 patients)
  Midterm ENet, 20 patients      runs/segthor_enet_dice_ce_all_corrected_ct_window_zscore_median_spacing_roi_crop
                                 (Dice+CE, full preprocessing, 25 epochs; 5 validation patients x 3 seeds)
  Midterm ENet recipe, 40 pat.   runs/full_cv4_enet_dice_ce_fold*/seed* (same recipe, 4-fold CV, 40 patients x 3 seeds)
  Upgraded ENet                  full_cv4_enet_dice_ce_strong_geometric_50ep (strong geometric augmentation,
                                 50 epochs, 4-fold CV, 40 patients x 3 seeds), volumes shared from igardner1

Rules, as in the original figure:
  2D · 4 / 2D · 8     pieces per axial slice that contains the label (edge / edge-or-corner contact)
  3D · 6 / 18 / 26    pieces of the whole label in one volume (face / +edge / +corner contact)
For 2D rules every axial slice containing the label counts once; for 3D rules every volume counts once.

Writes into <out-dir>/: connected_components_predictions.png/.pdf (all rows), connected_components_midterm.png/.pdf
(ground truth of the midterm release and the midterm ENet only, for the slides), connected_components_predictions.csv (shares),
and cache/<row>/<volume>.csv (piece counts per volume, so a rerun skips finished volumes).

Usage:
    python dataset_analysis/connected_components_predictions.py --out-dir figures/comparison --workers 16
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import generate_binary_structure, label

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import CLASSES, COLORS, NAMES  # noqa: E402  (also puts tools/ on sys.path and sets the Agg backend)
from plot_style import apply_style, decorate  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

# (rule name, dimensions, scipy connectivity rank), as in profile_figures/connected_components.py
RULES = [("2D · 4", 2, 1), ("2D · 8", 2, 2), ("3D · 6", 3, 1), ("3D · 18", 3, 2), ("3D · 26", 3, 3)]
MAX_PIECES = 3
SHARED = "/scratch-shared/igardner1/share_enet_aug"
# (row key, row name, glob patterns of label volumes)
SOURCES = [
    ("gt", "Ground truth", ["data/segthor_train_full/train/Patient_*/GT.nii.gz"]),
    ("gt_midterm", "Ground truth\n(midterm data)", ["data/segthor_part1_corrected/train/Patient_*/GT.nii.gz"]),
    ("midterm_20", "Midterm ENet\n(20 patients)",
     ["runs/segthor_enet_dice_ce_all_corrected_ct_window_zscore_median_spacing_roi_crop/seed*/volumes/val/*.nii.gz"]),
    ("midterm_40", "Midterm ENet recipe\n(40 patients, CV)",
     ["runs/full_cv4_enet_dice_ce_fold*/seed*/volumes/val/*.nii.gz"]),
    ("upgraded", "Upgraded ENet\n(strong aug, 50 ep)",
     [f"{SHARED}/full_cv4_enet_dice_ce_strong_geometric_50ep_fold*/seed*/volumes/val/*.nii.gz"]),
    ("final_unet", "Final model: ResEnc\nU-Net (strong aug, 50 ep)",
     ["runs/full_cv4_resenc_ds_unet_dice_ce_strong_geometric_50ep_fold*/seed*/volumes/val/*.nii.gz"]),
]

FOOTNOTE_RULES = ("Rules = which neighbours count as touching. 2D (one slice): 4 = sides only, 8 = sides + corners. "
                  "3D (whole scan): 6 = faces only, 18 = + edges, 26 = + corners. ")
# figure name -> (row keys, title, footnote), titles set from the results
FIGURES = {
    "connected_components_predictions": (
        ["gt", "midterm_20", "midterm_40", "upgraded", "final_unet"],
        "Each Upgrade Leaves Fewer Stray 3D Pieces, but Even the Final U-Net Predicts Only 28–45% of Organs as One "
        "Piece",
        FOOTNOTE_RULES + "Data: Full40 corrected SegTHOR validation predictions on the original CT grid; midterm "
        "20-patient run on data/segthor_part1_corrected (5 validation patients × 3 seeds). Final model: ResEnc + DS "
        "U-Net, Dice + CE, strong geometric augmentation, 50 epochs.",
    ),
    "connected_components_midterm": (
        ["gt_midterm", "midterm_20"],
        "The Midterm ENet Never Predicts the Heart or Esophagus as One 3D Piece, Though Every Real One Is",
        FOOTNOTE_RULES + "Ground truth: the 20 patients of the midterm release (data/segthor_part1_corrected, "
        "corrected labels). Midterm ENet: Dice + CE, full preprocessing, 25 epochs; its 5 validation patients × 3 "
        "seeds, predictions on the original CT grid.",
    ),
}


def piece_counts(mask: np.ndarray) -> dict[str, np.ndarray]:
    """Number of connected pieces of one label mask under every rule (copied from the profile figure script).

    Args:
        mask: Boolean label mask (x, y, z), z = axial slice index.

    Returns:
        Rule name -> piece counts: one per axial slice containing the label (2D rules), or a single count for the
        whole mask (3D rules). Empty arrays for an empty mask.
    """
    if not mask.any():
        return {name: np.array([], dtype=int) for name, _, _ in RULES}
    mask = mask[tuple(slice(i.min(), i.max() + 1) for i in np.nonzero(mask))]
    slices = np.flatnonzero(mask.any(axis=(0, 1)))
    counts = {}
    for name, dims, rank in RULES:
        structure = generate_binary_structure(dims, rank)
        if dims == 2:
            counts[name] = np.array([label(mask[:, :, z], structure)[1] for z in slices])
        else:
            counts[name] = np.array([label(mask, structure)[1]])
    return counts


def count_volume(task: tuple[str, str]) -> str:
    """Piece counts of every label in one volume, cached as csv.

    Args:
        task: (volume path, cache csv path).

    Returns:
        A one-line status message.
    """
    path, cache = task
    try:
        nii = nib.load(path)
        codes = "".join(nib.aff2axcodes(nii.affine))
        assert codes == "LPS", f"orientation {codes}"
        seg = np.asarray(nii.dataobj)
    except Exception as e:  # one unreadable volume should not lose the others
        return f"[skip] {path}: {type(e).__name__}: {e}"
    rows = [
        {"label": k, "rule": rule, "pieces": int(v)}
        for k in CLASSES
        for rule, values in piece_counts(seg == k).items()
        for v in values
    ]
    pd.DataFrame(rows, columns=["label", "rule", "pieces"]).to_csv(cache, index=False)
    return f"{path}: {seg.shape}, {codes}"


def piece_shares(counts: pd.DataFrame) -> pd.DataFrame:
    """Share of slices (2D) or volumes (3D) with each number of pieces (copied from the profile figure script).

    Args:
        counts: One row per slice or volume with columns label, rule, pieces.

    Returns:
        Percentages indexed by (label, rule), one column per piece count 1..MAX_PIECES; counts above MAX_PIECES are
        added to the last column.
    """
    capped = counts.assign(pieces=counts.pieces.clip(upper=MAX_PIECES))
    table = capped.groupby(["label", "rule", "pieces"]).size().unstack("pieces", fill_value=0)
    table = table.reindex(columns=range(1, MAX_PIECES + 1), fill_value=0)
    return 100 * table.div(table.sum(axis=1), axis=0)


def draw(shares: dict[str, pd.DataFrame], n_volumes: dict[str, int], labels: list[int], out_dir: Path,
         name_out: str, keys: list[str], title: str, footnote: str) -> None:
    """One row of heatmaps per source, one column per organ, rules down and piece counts across each heatmap.

    Args:
        shares: Row key -> piece_shares() table.
        n_volumes: Row key -> number of volumes pooled.
        labels: Organs drawn, left to right.
        out_dir: Folder to write the figure into.
        name_out: File name without extension.
        keys: Row keys of SOURCES to draw, top to bottom.
        title: Figure title.
        footnote: Figure footnote.
    """
    apply_style()
    sources = [s for k in keys for s in SOURCES if s[0] == k]
    rule_names = [name for name, _, _ in RULES]
    fig, axes = plt.subplots(len(sources), len(labels), figsize=(3.3 * len(labels) + 2.6, 2.35 * len(sources) + 1.6),
                             squeeze=False)
    for r, (key, name, _) in enumerate(sources):
        for c, k in enumerate(labels):
            ax = axes[r, c]
            table = shares[key].loc[k].reindex(rule_names).fillna(0)
            cmap = LinearSegmentedColormap.from_list(f"label{k}", ["#FFFFFF", COLORS[k]])
            ax.imshow(table.to_numpy(), cmap=cmap, vmin=0, vmax=100, aspect="auto")
            for i in range(len(rule_names)):
                for j in range(MAX_PIECES):
                    v = table.iat[i, j]
                    if v >= 0.5:
                        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                                color="white" if v > 55 else "#333333")
            # white gaps between cells, like the original seaborn linewidths
            ax.set_xticks(np.arange(-0.5, MAX_PIECES), minor=True)
            ax.set_yticks(np.arange(-0.5, len(rule_names)), minor=True)
            ax.grid(which="minor", color="white", linewidth=2)
            ax.grid(which="major", visible=False)
            ax.tick_params(which="both", length=0)
            for side in ax.spines.values():
                side.set_visible(False)
            ax.set_xticks(range(MAX_PIECES))
            ax.set_xticklabels([*map(str, range(1, MAX_PIECES)), f"{MAX_PIECES}+"] if r == len(sources) - 1 else [])
            ax.set_yticks(range(len(rule_names)))
            ax.set_yticklabels(rule_names if c == 0 else [], fontsize=10)
            if r == 0:
                ax.set_title(NAMES[k], loc="left", fontsize=14, fontweight="bold", color=COLORS[k])
            if r == len(sources) - 1:
                ax.set_xlabel("connected components")
        axes[r, 0].set_ylabel(f"{name}\n{n_volumes[key]} volumes", fontsize=11, fontweight="bold", color="#444444",
                              labelpad=10)
    decorate(
        fig,
        title,
        subtitle=(
            "Each cell = % of axial slices containing the organ (2D rows) or of volumes (3D rows) with that many "
            "connected components. Predictions pooled over all their runs (seeds, and folds where used); ground truth is one "
            "volume per patient."
        ),
        footnote_text=footnote,
    )
    fig.subplots_adjust(right=0.88, wspace=0.08, hspace=0.12)
    cax = fig.add_axes([0.905, 0.3, 0.012, 0.4])
    fig.colorbar(ScalarMappable(Normalize(0, 100), LinearSegmentedColormap.from_list("share", ["#FFFFFF", "#444444"])),
                 cax=cax, label="% of slices (2D) or volumes (3D)")
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name_out}.{ext}")
    plt.close(fig)
    print(f"wrote {out_dir / name_out}.png")


def main():
    """Count pieces in every volume (cached), pool per row, write the table and log, and draw the figure."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("figures/comparison"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    cache_root = args.out_dir / "connected_components_predictions_cache"

    volumes, tasks = {}, []
    for key, _, patterns in SOURCES:
        paths = sorted({p for pattern in patterns for p in glob.glob(pattern)})
        assert paths, f"{key}: no volumes for {patterns}"
        (cache_root / key).mkdir(parents=True, exist_ok=True)
        volumes[key] = []
        for p in paths:
            cache = cache_root / key / (p.replace("/", "_").strip("_") + ".csv")
            volumes[key].append(cache)
            if not cache.exists():
                tasks.append((p, str(cache)))
        print(f"{key}: {len(paths)} volumes", flush=True)
    print(f"{len(tasks)} volumes to count", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        for msg in pool.map(count_volume, tasks):
            print(msg, flush=True)

    shares, n_volumes, rows = {}, {}, []
    for key, name, _ in SOURCES:
        done = [c for c in volumes[key] if c.exists()]
        counts = pd.concat([pd.read_csv(c) for c in done], ignore_index=True)
        shares[key], n_volumes[key] = piece_shares(counts), len(done)
        rows.append(shares[key].reset_index().assign(source=name.replace("\n", " "), n_volumes=len(done)))
    pd.concat(rows).to_csv(args.out_dir / "connected_components_predictions.csv", index=False)
    labels = [k for k in CLASSES if all(k in s.index.get_level_values("label") for s in shares.values())]
    for name_out, (keys, title, footnote) in FIGURES.items():
        draw(shares, n_volumes, labels, args.out_dir, name_out, keys, title, footnote)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    log = {"sources": [(k, n, p) for k, n, p in SOURCES], "n_volumes": n_volumes, "git_commit": commit,
           "started": started, "finished": datetime.now(timezone.utc).isoformat()}
    (args.out_dir / "connected_components_predictions_log.json").write_text(json.dumps(log, indent=2))
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(pd.concat(rows).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
