"""3D loss landscapes of the baseline and four candidate segmentation losses over the weights of a tiny model.

Toy image: H x W voxels, FG_FRACTION of them true foreground, with one intensity value per voxel
(foreground ~ N(1, sigma), background ~ N(0, sigma)). The model is a per-voxel logistic classifier with two
parameters: P(fg) = sigmoid(w * intensity + b). Each surface is the loss over (w, b), evaluated with the repo's
own loss classes (src.losses).

Run from the repo root: python tools/plot_loss_landscapes_weights.py
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO))
sys.path.append(str(REPO / "tools"))

from plot_loss_landscapes import CMAP, GRID_INK, K, LOSSES  # noqa: E402
from plot_style import EARTH, apply_style, decorate  # noqa: E402
from src.registry import build  # noqa: E402


def weight_landscape(loss_fn, intensity: torch.Tensor, gt: torch.Tensor, ws: np.ndarray, bs: np.ndarray) -> np.ndarray:
    """loss_fn on the softmax([0, w * intensity + b]) prediction for every (w, b); returns z[ib, iw]."""
    z = np.zeros((len(bs), len(ws)))
    for ib, b in enumerate(bs):
        for iw, w in enumerate(ws):
            logit = float(w) * intensity + float(b)
            probs = torch.softmax(torch.stack([torch.zeros_like(logit), logit], dim=1), dim=1)
            z[ib, iw] = loss_fn(probs, gt).item()
    return z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=20, help="toy image is size x size voxels")
    ap.add_argument("--fg-fraction", type=float, default=0.05)
    ap.add_argument("--sigma", type=float, default=0.3, help="intensity noise std")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w-range", type=float, nargs=2, default=(-4.0, 24.0))
    ap.add_argument("--b-range", type=float, nargs=2, default=(-20.0, 6.0))
    ap.add_argument("--n", type=int, default=61, help="grid points per axis")
    ap.add_argument("--out", type=Path,
                    default=REPO / "results" / "loss_landscapes" / "loss_landscapes_option2_weight_space.png")
    args = ap.parse_args()
    print(f"seed={args.seed}")

    n_fg = round(args.size ** 2 * args.fg_fraction)
    fg_mask = torch.zeros(1, args.size, args.size)
    fg_mask.view(-1)[:n_fg] = 1
    gen = torch.Generator().manual_seed(args.seed)
    intensity = fg_mask + args.sigma * torch.randn(fg_mask.shape, generator=gen)
    gt = torch.stack([1 - fg_mask, fg_mask], dim=1).long()

    ws, bs = np.linspace(*args.w_range, args.n), np.linspace(*args.b_range, args.n)
    ww, bb = np.meshgrid(ws, bs)

    apply_style()
    plt.rcParams.update({"grid.color": GRID_INK, "grid.linewidth": 0.7})
    fig = plt.figure(figsize=(19, 13))
    for i, (title, name, kw) in enumerate(LOSSES, start=1):
        z = weight_landscape(build("loss", name, num_classes=K, **kw), intensity, gt, ws, bs)
        zmax = z.max()  # own z-scale per panel, as in the prediction-space figure
        ib, iw = np.unravel_index(z.argmin(), z.shape)

        ax = fig.add_subplot(2, 3, i, projection="3d")
        ax.plot_surface(ww, bb, z, cmap=CMAP, vmin=0, vmax=zmax, rstride=1, cstride=1, linewidth=0,
                        antialiased=True, alpha=0.95)
        ax.scatter([ws[iw]], [bs[ib]], [z[ib, iw]], s=90, marker="*", color=EARTH[1], edgecolor="white",
                   linewidth=0.8, depthshade=False, label="Lowest loss on the grid", zorder=10)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=22)
        ax.text2D(0.5, 1.0, f"min {z[ib, iw]:.2f} at w={ws[iw]:.1f}, b={bs[ib]:.1f}  |  max {zmax:.1f}",
                  transform=ax.transAxes, ha="center", fontsize=9.5, color="#333333")
        ax.set_xlabel("w (intensity weight)", fontsize=9.5, labelpad=-2)
        ax.set_ylabel("b (bias)", fontsize=9.5, labelpad=-2)
        ax.set_zlabel("loss", fontsize=9.5, labelpad=-4)
        ax.set_zlim(0, zmax)
        ax.view_init(elev=26, azim=-52)
        ax.tick_params(labelsize=8, pad=-3)
        ax.grid(True)
    key = fig.add_subplot(2, 3, 6)
    key.axis("off")
    key.legend(*ax.get_legend_handles_labels(), loc="center", fontsize=13)

    decorate(
        fig,
        "The losses pull the same tiny model to different weights",
        subtitle=(f"Loss over the two parameters of P(fg) = sigmoid(w * intensity + b) on a toy {args.size}x{args.size} image "
                  f"with {args.fg_fraction:.0%} foreground. Each panel has its own z-scale."),
        footnote_text=(f"Intensity ~ N(1, {args.sigma}) on foreground, N(0, {args.sigma}) on background, seed {args.seed}. "
                       "Computed with the repo's own loss classes; Baseline is main.py's original CE over all classes; Dice + CE uses ce_weight = dice_weight = 1 and Dice is "
                       "foreground-only. Where the star sits on the edge of the grid, the minimum lies beyond it."),
    )
    fig.subplots_adjust(wspace=0.12, hspace=0.15, left=0.01, right=0.95)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
