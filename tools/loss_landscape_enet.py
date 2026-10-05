"""Loss landscapes of one trained ENet under each candidate loss (filter-normalised random directions, Li et al. 2018).

Two random directions d1, d2 are drawn around the trained weights theta*. Each filter of a direction is rescaled to the norm of
the matching filter of theta*; biases, BatchNorm and PReLU parameters are not perturbed. Every loss is evaluated on
L(theta* + a*d1 + b*d2) over an n x n grid, a, b in [-radius, radius], with the net in eval mode (BatchNorm uses its stored
statistics). One forward pass per grid point serves all losses, so the surfaces are the same weights seen through each loss.

Losses are evaluated as the training loop does: softmax probabilities, batches of data.batch_size, mean of the per-batch
losses. Resumable: the surfaces are saved after every grid row, and a rerun continues from there.

Run via jobsAndOutputs/loss_landscape/jobs/ (GPU); never on the login node.
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO))

import src.losses  # noqa: E402,F401  registers losses
import src.models  # noqa: E402,F401  registers models
from src.checkpoint import load_checkpoint  # noqa: E402
from src.config import read_yaml  # noqa: E402
from src.data import build_dataset  # noqa: E402
from src.engine import build_model  # noqa: E402
from src.registry import build  # noqa: E402

LOG = logging.getLogger("loss_landscape")
MAX_FAILED_POINTS = 20
# ce_repro_corrected_fixed: CE over all classes on data/SEGTHOR_corrected. (ce_corrected_control was trained on a different
# slicing, data/sliced_029bf0bc with retains=5, so it does not share the split of the other runs.)
DEFAULT_RUN = "runs/segthor_enet_ce_repro_corrected_fixed/seed0"


def loss_specs(k: int) -> list[tuple[str, str, str, dict]]:
    """(slug, panel title, registry name, kwargs) for the baseline and the candidate losses, as in configs/segthor_enet_*_corrected."""
    everything, fg = list(range(k)), list(range(1, k))
    return [
        ("ce_all", "Baseline: CE all classes", "cross_entropy", {"idk": everything}),
        ("ce_fg", "1. CE foreground only", "cross_entropy", {"idk": fg}),
        ("dice_fg", "2. Dice (foreground only)", "soft_dice", {"idk": fg}),
        ("ce_all_dice_fg", "3. CE (all) + Dice (fg)", "dice_ce", {"ce_idk": everything, "dice_idk": fg}),
        ("ce_fg_dice_fg", "4. CE (fg) + Dice (fg)", "dice_ce", {"ce_idk": fg, "dice_idk": fg}),
        ("dice_all", "5. Dice (all classes)", "soft_dice", {"idk": everything}),
    ]


def load_slices(cfg: dict, split: str, n_slices: int, device: torch.device):
    """Evenly spaced slices of `split` (spread over the patients), un-augmented, kept on the device."""
    ds = build_dataset(cfg | {"train": cfg["train"] | {"debug_samples": 0}}, split)
    idx = np.linspace(0, len(ds) - 1, min(n_slices, len(ds))).astype(int)
    items = [ds[i] for i in idx]
    imgs = torch.stack([it["images"] for it in items]).to(device)
    gts = torch.stack([it["gts"] for it in items]).to(torch.uint8).to(device)
    return imgs, gts, [it["stems"] for it in items]


def random_direction(net: torch.nn.Module, seed: int) -> list[torch.Tensor]:
    """Gaussian direction per parameter, each filter (dim 0) rescaled to the norm of the same filter of theta*."""
    gen = torch.Generator().manual_seed(seed)
    dirs = []
    for p in net.parameters():
        d = torch.randn(p.shape, generator=gen).to(p.device)
        if p.dim() <= 1:
            d.zero_()
        else:
            dims = tuple(range(1, p.dim()))
            d *= p.detach().norm(dim=dims, keepdim=True) / (d.norm(dim=dims, keepdim=True) + 1e-10)
        dirs.append(d)
    return dirs


@torch.no_grad()
def point_losses(net, loss_fns, imgs, gts, batch_size: int) -> np.ndarray:
    """Mean over batches of every loss at the current weights. NaN if the perturbed net outputs non-finite values."""
    acc, n_batches = torch.zeros(len(loss_fns), device=imgs.device), 0
    for s in range(0, len(imgs), batch_size):
        probs = F.softmax(net(imgs[s:s + batch_size]), dim=1)
        if not torch.isfinite(probs).all():
            return np.full(len(loss_fns), np.nan)
        gt = gts[s:s + batch_size].long()
        acc += torch.stack([fn(probs, gt) for fn in loss_fns])
        n_batches += 1
    return (acc / n_batches).cpu().numpy()


def save_state(path: Path, losses: np.ndarray, rows_done: int, settings: dict, coords: np.ndarray,
               specs: list[tuple[str, str, str, dict]]) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, losses=losses, rows_done=rows_done, settings=json.dumps(settings, sort_keys=True), coords=coords,
             slugs=np.array([s[0] for s in specs]), titles=np.array([s[1] for s in specs]))
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN, help="run dir with config.yaml and checkpoints/best.pt")
    ap.add_argument("--grid-n", type=int, default=41, help="grid points per axis (odd, so the centre is theta*)")
    ap.add_argument("--radius", type=float, default=1.0, help="a, b in [-radius, radius], in units of the filter norms")
    ap.add_argument("--n-slices", type=int, default=256)
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--seed", type=int, default=0, help="seed of the random directions")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "loss_landscape_enet")
    ap.add_argument("--smoke", action="store_true", help="5x5 grid, 16 slices, separate output dir")
    args = ap.parse_args()
    if args.smoke:
        args.grid_n, args.n_slices, args.out = 5, 16, REPO / "results" / "loss_landscape_enet_smoke"
    assert args.grid_n % 2 == 1, "use an odd --grid-n so the grid contains the trained weights"
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.out / "loss_landscape.log")])
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible: run this through sbatch on a GPU partition, not on the login node")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    run_dir = REPO / args.run
    cfg = read_yaml((run_dir / "config.yaml").read_text())
    net = build_model(cfg, device)
    best = load_checkpoint(run_dir / "checkpoints" / "best.pt")
    net.load_state_dict(best["model"])
    net.eval()
    specs = loss_specs(cfg["data"]["num_classes"])
    loss_fns = [build("loss", name, num_classes=cfg["data"]["num_classes"], **kw) for _, _, name, kw in specs]
    imgs, gts, stems = load_slices(cfg, args.split, args.n_slices, device)
    bs, n = cfg["data"]["batch_size"], args.grid_n
    coords = np.linspace(-args.radius, args.radius, n)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    manifest = {"started": time.strftime("%F %T"), "commit": commit, "args": {k: str(v) for k, v in vars(args).items()},
                "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                "run": args.run, "best_epoch": best["epoch"], "trained_loss": cfg["loss"], "data_root": cfg["data"]["root"],
                "n_slices": len(stems), "first_stem": stems[0], "last_stem": stems[-1],
                "losses": {slug: {"name": name, "kwargs": kw} for slug, _, name, kw in specs}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    LOG.info("commit %s, %s epoch %d, %d %s slices, grid %dx%d over [-%g, %g], seed %d", commit[:8], args.run, best["epoch"],
             len(stems), args.split, n, n, args.radius, args.radius, args.seed)

    path = args.out / "landscape.npz"
    settings = {"run": args.run, "epoch": best["epoch"], "grid_n": n, "radius": args.radius, "n_slices": len(stems),
                "split": args.split, "seed": args.seed, "losses": [s[0] for s in specs]}
    losses, rows_done = np.full((len(specs), n, n), np.nan), 0
    if path.exists():
        saved = np.load(path)
        if json.loads(str(saved["settings"])) != settings:
            raise SystemExit(f"{path} was made with different settings; remove it or change --out")
        losses, rows_done = saved["losses"], int(saved["rows_done"])
    if rows_done == n:
        LOG.info("already complete: %s", path)
        return

    params = list(net.parameters())
    theta = [p.detach().clone() for p in params]
    d1, d2 = random_direction(net, 2 * args.seed), random_direction(net, 2 * args.seed + 1)
    failed = 0
    for ib in range(rows_done, n):
        t0 = time.time()
        for ia in range(n):
            with torch.no_grad():
                for p, t, a, b in zip(params, theta, d1, d2):
                    p.copy_(t + coords[ia] * a + coords[ib] * b)
            try:
                losses[:, ib, ia] = point_losses(net, loss_fns, imgs, gts, bs)
            except (RuntimeError, AssertionError):
                failed += 1
                LOG.exception("grid point a=%.3f b=%.3f failed (%d/%d allowed)", coords[ia], coords[ib], failed,
                              MAX_FAILED_POINTS)
                losses[:, ib, ia] = np.nan
                if failed > MAX_FAILED_POINTS:
                    raise
        save_state(path, losses, ib + 1, settings, coords, specs)
        LOG.info("row %d/%d done in %.1fs", ib + 1, n, time.time() - t0)
    centre = losses[:, n // 2, n // 2]
    LOG.info("loss at theta* (centre) per loss: %s", dict(zip(settings["losses"], np.round(centre, 4).tolist())))
    manifest["finished"] = time.strftime("%F %T")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    LOG.info("done -> %s", path)


if __name__ == "__main__":
    main()
