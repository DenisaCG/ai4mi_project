"""Report for data.preprocess.normalize: builds the dataset of a config and the same one without `normalize`
(the control, same spacing), prints the stats, checks the GT PNGs are byte-identical, and plots the input-intensity
histograms before (per-volume min-max) and after (fixed HU window).
Usage: python dataset_analysis/ct_norm_report.py --config configs/segthor_enet_ce_corrected_ct_window.yaml"""
import argparse
import copy
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from plot_style import BACKGROUND_COLOR, EARTH, apply_style, decorate, legend_below  # noqa: E402
from src.config import load_config, resolve_preprocess  # noqa: E402
from src.data import ensure_sliced  # noqa: E402

EVERY = 5  # every 5th slice is enough for a histogram


def grey_values(root: Path, split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """(all pixels, foreground pixels) of the img PNGs, foreground = gt > 0."""
    imgs = sorted((root / split / "img").glob("*.png"))[::EVERY]
    allv, fg = [], []
    for f in imgs:
        img, gt = np.array(Image.open(f)), np.array(Image.open(root / split / "gt" / f.name))
        allv.append(img.ravel())
        fg.append(img[gt > 0])
    return np.concatenate(allv), np.concatenate(fg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=str(REPO / "dataset_analysis" / "results" / "ct_norm"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    control = copy.deepcopy(cfg)
    control["data"]["preprocess"].pop("normalize")
    resolve_preprocess(control)
    for c in (control, cfg):
        ensure_sliced(c)
    roots = {"before (per-volume min-max)": REPO / control["data"]["root"], "after (fixed HU window)": REPO / cfg["data"]["root"]}
    print("control dataset:", roots["before (per-volume min-max)"], "\nnormalized dataset:", roots["after (fixed HU window)"])

    stats = json.loads((roots["after (fixed HU window)"] / "ct_norm_stats.json").read_text())
    print("normalization stats:", {k: round(stats[k], 3) for k in ("lo", "hi", "mean", "std")},
          "| expected window ~(-991, 248) | train patients:", len(stats["train_patients"]))

    before, after = (roots[k] for k in roots)
    for split in ("train", "val"):
        names = sorted(p.name for p in (before / split / "gt").glob("*.png"))
        assert names == sorted(p.name for p in (after / split / "gt").glob("*.png")), f"{split}: slice lists differ"
        bad = [n for n in names if (before / split / "gt" / n).read_bytes() != (after / split / "gt" / n).read_bytes()]
        print(f"{split}: {len(names)} GT PNGs, byte-different from control: {len(bad)}", bad[:3])
        assert not bad, "labels changed"

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (title, root) in zip(axes, roots.items()):
        allv, fg = grey_values(root)
        p1, p99 = np.percentile(fg, [1, 99])
        print(f"{title}: foreground grey level p1-p99 = {p1:.0f}-{p99:.0f} (width {p99 - p1:.0f}/255), "
              f"all-voxel std {allv.std():.1f}, foreground mean {fg.mean():.1f} std {fg.std():.1f}")
        bins = np.arange(257) - 0.5
        ax.hist(allv, bins=bins, density=True, alpha=.5, label="all voxels", color=BACKGROUND_COLOR)
        ax.hist(fg, bins=bins, density=True, alpha=.7, label="foreground (labels > 0)", color=EARTH[1])
        ax.set_yscale("log")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("PNG grey level")
    axes[0].set_ylabel("Density (log scale)")
    legend_below(axes[0], ncol=2)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    path = Path(args.out) / f"{cfg['experiment']}_hist.png"
    decorate(fig, "Input intensity before and after the fixed HU window",
             f"{cfg['experiment']}  |  Grey level of the processed PNG slices, every {EVERY}th slice of the training split",
             "Foreground = voxels with a label above 0. The ground-truth PNGs are byte-identical to the control, so only the input intensity changed.")
    fig.savefig(path)
    print("saved", path)


if __name__ == "__main__":
    main()
