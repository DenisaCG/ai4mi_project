"""3D loss landscapes of the baseline and four candidate segmentation losses on a toy binary problem (prediction space).

Toy image: H x W voxels, FG_FRACTION of them true foreground. The prediction is uniform inside each group:
  x = p = predicted foreground probability on true-foreground voxels  (1 is right)
  y = q = predicted foreground probability on true-background voxels  (0 is right, >0 is a false positive)
Every surface is evaluated with the repo's own loss classes (src.losses), not re-derived formulas.
Top row: 3D surfaces. Bottom row: the same surfaces seen from above, with labelled loss contours and arrows
showing the direction of steepest descent in (p, q) (autograd through the repo loss; direction only).

Run from the repo root: python tools/plot_loss_landscapes.py
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO))
sys.path.append(str(REPO / "tools"))

import src.losses  # noqa: E402,F401  registers losses
from plot_style import EARTH, apply_style, decorate, legend_below  # noqa: E402
from src.registry import build  # noqa: E402

K = 2  # background + one organ; class 1 is the foreground
LOSSES = [  # (panel title, registry name, kwargs)
    ("Baseline: CE all classes", "cross_entropy", {}),  # main.py's original CrossEntropy(idk=range(K))
    ("1. CE foreground only", "cross_entropy", {"idk": [1]}),
    ("2. Dice (foreground only)", "soft_dice", {}),
    ("3. CE (all) + Dice (fg)", "dice_ce", {}),
    ("4. CE (fg) + Dice (fg)", "dice_ce", {"ce_idk": [1]}),
    ("5. Dice (all classes)", "soft_dice", {"idk": [0, 1]}),  # background included, as configs/segthor_enet_dice_all_corrected
]
# single-hue sequential, sand (low loss) -> brick (high loss)
CMAP = LinearSegmentedColormap.from_list("sand_brick", ["#F3EEE4", "#E3C9A0", EARTH[0], "#5A2314"])
GRID_INK = "#9A9A9A"  # darker than plot_style's grid colour, which vanishes against the light 3D panes


def one_hot(fg_mask: torch.Tensor) -> torch.Tensor:
    return torch.stack([1 - fg_mask, fg_mask], dim=1).long()


def predict(fg_mask: torch.Tensor, p, q) -> torch.Tensor:
    """(1, 2, H, W) probabilities: foreground probability p on true-foreground voxels, q elsewhere."""
    fg_prob = torch.where(fg_mask.bool(), p, q)
    return torch.stack([1 - fg_prob, fg_prob], dim=1)


def landscape(loss_fn, fg_mask: torch.Tensor, grid: np.ndarray) -> np.ndarray:
    """loss_fn(probs, gt) on every (p, q) pair; returns z[iy, ix] with x = p, y = q."""
    gt = one_hot(fg_mask)
    z = np.zeros((len(grid), len(grid)))
    for iy, q in enumerate(grid):
        for ix, p in enumerate(grid):
            probs = predict(fg_mask, torch.tensor(float(p)), torch.tensor(float(q)))
            z[iy, ix] = loss_fn(probs, gt).item()
    return z


def descent_directions(loss_fn, fg_mask: torch.Tensor, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit vector of steepest descent (-grad of the loss w.r.t. p and q) at every (p, q) in pts x pts."""
    gt = one_hot(fg_mask)
    u, v = np.zeros((len(pts), len(pts))), np.zeros((len(pts), len(pts)))
    for iy, q in enumerate(pts):
        for ix, p in enumerate(pts):
            p_t, q_t = torch.tensor(float(p), requires_grad=True), torch.tensor(float(q), requires_grad=True)
            loss_fn(predict(fg_mask, p_t, q_t), gt).backward()
            g = np.array([p_t.grad.item(), q_t.grad.item()])
            u[iy, ix], v[iy, ix] = -g / (np.linalg.norm(g) + 1e-12)
    return u, v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=20, help="toy image is size x size voxels")
    ap.add_argument("--fg-fraction", type=float, default=0.05)
    ap.add_argument("--lim", type=float, default=0.01, help="probability grid runs over [lim, 1 - lim]")
    ap.add_argument("--n", type=int, default=41, help="grid points per axis")
    ap.add_argument("--n-arrows", type=int, default=9, help="arrows per axis in the top-down row")
    ap.add_argument("--out", type=Path,
                    default=REPO / "results" / "loss_landscapes" / "loss_landscapes_option1_gradients.png")
    args = ap.parse_args()

    n_fg = round(args.size ** 2 * args.fg_fraction)
    fg_mask = torch.zeros(1, args.size, args.size)
    fg_mask.view(-1)[:n_fg] = 1
    grid = np.linspace(args.lim, 1 - args.lim, args.n)
    xx, yy = np.meshgrid(grid, grid)
    arrow_pts = np.linspace(0.06, 0.94, args.n_arrows)
    lo, hi = args.lim, 1 - args.lim

    apply_style()
    plt.rcParams.update({"grid.color": GRID_INK, "grid.linewidth": 0.7})
    n = len(LOSSES)
    fig = plt.figure(figsize=(4.7 * n, 10.8))
    for i, (title, name, kw) in enumerate(LOSSES, start=1):
        loss_fn = build("loss", name, num_classes=K, **kw)
        z = landscape(loss_fn, fg_mask, grid)
        u, v = descent_directions(loss_fn, fg_mask, arrow_pts)
        zmax = z.max()  # own z-scale per panel, otherwise the log spike of the CE terms flattens the Dice panel
        best, trivial = z[0, -1], z[-1, -1]  # (p, q) = (hi, lo) and (hi, hi)

        ax = fig.add_subplot(2, n, i, projection="3d")
        ax.plot_surface(xx, yy, z, cmap=CMAP, vmin=0, vmax=zmax, rstride=1, cstride=1, linewidth=0,
                        antialiased=True, alpha=0.95)
        ax.scatter([hi], [lo], [best], s=90, marker="*", color=EARTH[1], edgecolor="white", linewidth=0.8,
                   depthshade=False, label="Correct prediction (p=1, q=0)", zorder=10)
        ax.scatter([hi], [hi], [trivial], s=60, marker="o", color="#222222", edgecolor="white", linewidth=0.8,
                   depthshade=False, label="Everything foreground (p=1, q=1)", zorder=10)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=22)
        ax.text2D(0.5, 1.0, f"correct {best:.2f}  |  all-foreground {trivial:.2f}  |  max {zmax:.1f}",
                  transform=ax.transAxes, ha="center", fontsize=9.5, color="#333333")
        ax.set_xlabel("p: P(fg) on fg voxels", fontsize=9.5, labelpad=-2)
        ax.set_ylabel("q: P(fg) on bg voxels", fontsize=9.5, labelpad=-2)
        ax.set_zlabel("loss", fontsize=9.5, labelpad=-4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, zmax)
        ax.view_init(elev=26, azim=-52)
        ax.tick_params(labelsize=8, pad=-3)
        ax.grid(True)

        top = fig.add_subplot(2, n, n + i)
        top.imshow(z, origin="lower", extent=(0, 1, 0, 1), cmap=CMAP, vmin=0, vmax=zmax, aspect="equal")
        cs = top.contour(xx, yy, z, levels=6, colors="#333333", linewidths=0.6, alpha=0.7)
        top.clabel(cs, fontsize=7, fmt="%.1f")
        top.quiver(arrow_pts, arrow_pts, u, v, pivot="mid", color="#222222", width=0.004, angles="uv",
                   scale=16, scale_units="width", zorder=5)
        top.scatter([hi], [lo], s=110, marker="*", color=EARTH[1], edgecolor="white", linewidth=0.8, zorder=10)
        top.scatter([hi], [hi], s=60, marker="o", color="#222222", edgecolor="white", linewidth=0.8, zorder=10)
        top.set_xlim(0, 1)
        top.set_ylim(0, 1)
        top.set_xlabel("p: P(fg) on fg voxels", fontsize=10)
        top.set_ylabel("q: P(fg) on bg voxels", fontsize=10)
        top.grid(True, axis="both", color="white", alpha=0.6, linewidth=0.7)
        top.spines[["top", "right"]].set_visible(True)
    legend_below(ax, ncol=2)

    decorate(
        fig,
        "Foreground-only cross-entropy cannot see false positives",
        subtitle=(f"Loss over a toy {args.size}x{args.size} image with {args.fg_fraction:.0%} foreground, as a function of "
                  "the predicted foreground probability on foreground voxels (x) and on background voxels (y). "
                  "Each panel has its own z-scale."),
        footnote_text=("Toy binary case (background + one organ), each voxel group predicted uniformly, computed with the repo's "
                       "own loss classes. Baseline is main.py's original CE over all classes. Dice + CE uses "
                       "ce_weight = dice_weight = 1; Dice is foreground-only except panel 5, which includes background. Bottom row: same surfaces from above; labelled "
                       "lines are loss levels, arrows point in the direction of steepest descent (direction only, not to scale)."),
    )
    fig.subplots_adjust(wspace=0.12, left=0.035, right=0.95, hspace=0.18)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
