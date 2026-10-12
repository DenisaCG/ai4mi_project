"""Test-time augmentation on a finished run: no retraining, same checkpoint, same validation patients.

    python -m src.eval_tta --run runs/full_cv4_resenc_ds_unet_dice_ce_strong_geometric_50ep_fold0/seed0

Each validation slice is predicted several times with a small geometric change of the input (the kinds of change the model
was trained with: rotation, scaling, shift), every softmax output is mapped back onto the original frame and the
probabilities are averaged, counting at each pixel only the passes whose input covered it. The averaged label slices are
stitched onto the CT grid and scored in 3D exactly like src.evaluate (Dice, HD95, ASSD against the original GT).

Variants (each includes the untouched pass):
  none    the plain prediction; must reproduce eval/metrics_3d.csv
  rot     + rotations of +-ROT degrees
  scale   + zoom out / in by SCALE
  shift   + shifts of +-SHIFT of the width, in x and y
  all     rot + scale + shift
  flip    + a left-right mirror (NOT a transform the model was trained with: a reference for nnU-Net-style mirroring)
Writes <run>/eval/metrics_3d_tta.csv (variant, patient, class, dice, hd95, assd).
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from src.config import REPO, read_yaml
from src.data import build_dataset, patient_of
from src.evaluate import stitch
from src.metrics_3d import METRICS, volume_metrics
from src.run import read_json, resolve_device

ROT, SCALE, SHIFT = 10.0, 0.1, 0.05
IDENTITY = {}


def theta(angle: float = 0.0, scale: float = 1.0, tx: float = 0.0, ty: float = 0.0, flip: bool = False) -> torch.Tensor:
    """2x3 matrix for F.affine_grid: output pixel p samples the input at theta @ p (normalised coordinates, y down).
    scale > 1 zooms in, a shift of f (fraction of the width) is 2 f in normalised units."""
    a = math.radians(angle)
    m = torch.tensor([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]) / scale
    if flip:
        m = m @ torch.diag(torch.tensor([-1.0, 1.0]))
    return torch.cat([m, torch.tensor([[2 * tx], [2 * ty]])], dim=1)


TRANSFORMS = {
    "rot_p": theta(angle=ROT), "rot_m": theta(angle=-ROT),
    "scale_dn": theta(scale=1 - SCALE), "scale_up": theta(scale=1 + SCALE),
    "shift_xp": theta(tx=SHIFT), "shift_xm": theta(tx=-SHIFT), "shift_yp": theta(ty=SHIFT), "shift_ym": theta(ty=-SHIFT),
    "flip": theta(flip=True),
}
GROUPS = {"rot": ["rot_p", "rot_m"], "scale": ["scale_dn", "scale_up"],
          "shift": ["shift_xp", "shift_xm", "shift_yp", "shift_ym"], "flip": ["flip"]}
VARIANTS = {"none": [], "rot": GROUPS["rot"], "scale": GROUPS["scale"], "shift": GROUPS["shift"],
            "all": GROUPS["rot"] + GROUPS["scale"] + GROUPS["shift"], "flip": GROUPS["flip"]}


def warp(x: torch.Tensor, th: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    """x (B, C, H, W) resampled by th (bilinear); pixels outside x get `fill`."""
    grid = F.affine_grid(th.to(x).expand(x.shape[0], 2, 3), list(x.shape), align_corners=False)
    return F.grid_sample(x - fill, grid, mode="bilinear", padding_mode="zeros", align_corners=False) + fill


def unwarp(y: torch.Tensor, th: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Map the output y of a warped input back onto the original frame. Returns (y, coverage in 0..1)."""
    inv = torch.linalg.inv(torch.cat([th, torch.tensor([[0.0, 0.0, 1.0]])], dim=0))[:2].to(y)
    grid = F.affine_grid(inv.expand(y.shape[0], 2, 3), list(y.shape), align_corners=False)
    back = F.grid_sample(y, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    cover = F.grid_sample(torch.ones_like(y[:, :1]), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return back, cover


def fill_value(cfg: dict) -> float:
    """What the warps put outside the image: air in z-score units (the fill the training augmentation uses), else 0."""
    pre = cfg["data"].get("preprocess") or {}
    if pre.get("normalize") != "ct_window_zscore":
        return 0.0
    z = json.loads((REPO / cfg["data"]["root"] / "ct_norm_stats.json").read_text())
    return (z["lo"] - z["mean"]) / z["std"]


@torch.no_grad()
def average_probs(net, x: torch.Tensor, names: list[str], fill: float) -> torch.Tensor:
    """Softmax of x (B, C, H, W) averaged with the softmax of each warped copy named in `names`, mapped back and
    weighted by the part of the frame each copy covered. `names=[]` is the plain softmax."""
    total = net(x).softmax(dim=1)
    weight = torch.ones_like(total[:, :1])
    for name in names:
        back, cover = unwarp(net(warp(x, TRANSFORMS[name], fill)).softmax(dim=1), TRANSFORMS[name])
        total, weight = total + back, weight + cover  # `back` is already scaled by the coverage (zero padding)
    return total / weight


@torch.no_grad()
def predict_variants(net, images: torch.Tensor, fill: float, device: torch.device, batch: int = 16) -> dict[str, np.ndarray]:
    """Label slices (Z, H, W) per variant for one patient's slices (Z, C, H, W)."""
    out = {v: [] for v in VARIANTS}
    for chunk in images.split(batch):
        x = chunk.to(device)
        plain = net(x).softmax(dim=1)
        passes = {"identity": (plain, torch.ones_like(plain[:, :1]))}
        for name in {n for names in VARIANTS.values() for n in names}:
            back, cover = unwarp(net(warp(x, TRANSFORMS[name], fill)).softmax(dim=1), TRANSFORMS[name])
            passes[name] = (back, cover)  # `back` is already scaled by the coverage (zero padding): not multiplied again
        for v, names in VARIANTS.items():
            total = sum(passes[n][0] for n in ["identity"] + names)
            weight = sum(passes[n][1] for n in ["identity"] + names)
            out[v].append((total / weight).argmax(dim=1).cpu().numpy().astype(np.uint8))
    return {v: np.concatenate(parts) for v, parts in out.items()}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = p.parse_args(argv)
    from src.predict_test import load_net  # here, not at the top: predict_test imports this module

    run, device = args.run.resolve(), resolve_device(args.device)
    cfg, net = load_net(run, device)
    cfg = cfg | {"train": cfg["train"] | {"debug_samples": 0}}
    d = cfg["data"]
    root, pre = REPO / d["root"], d.get("preprocess") or {}
    fill = fill_value(cfg)

    loader = torch.utils.data.DataLoader(build_dataset(cfg, "val"), batch_size=16, shuffle=False, num_workers=0)
    stems, images = defaultdict(list), defaultdict(list)
    for b in loader:
        for stem, img in zip(b["stems"], b["images"]):
            patient = patient_of(stem, d["patient_regex"])
            stems[patient].append(stem)
            images[patient].append(img)

    rows = []
    for patient in sorted(stems):
        order = sorted(range(len(stems[patient])), key=lambda i: stems[patient][i])
        labels = predict_variants(net, torch.stack([images[patient][i] for i in order]), fill, device)
        ref = nib.load(REPO / d["source_pattern"].format(patient=patient))
        spacing = tuple(float(s) for s in ref.header.get_zooms()[:3])
        gt = np.asarray(ref.dataobj)
        crop = read_json(root / "roi_crop" / f"{patient}.json") if pre.get("crop") else None
        for variant, slices in labels.items():
            vol = stitch(dict(enumerate(slices)), ref, patient, bool(pre.get("resample")), crop)
            for k in range(1, d["num_classes"]):
                rows.append({"variant": variant, "patient": patient, "class_idx": k, "class_name": d["class_names"][k],
                             **volume_metrics(gt == k, vol == k, spacing)})
        print(f"{patient}: {len(slices)} slices, {len(VARIANTS)} variants scored", flush=True)

    out = run / "eval" / "metrics_3d_tta.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # the plain variant must reproduce the run's own evaluation
    ref_csv = run / "eval" / "metrics_3d.csv"
    if ref_csv.exists():
        mine = {(r["patient"], r["class_idx"]): float(r["dice"]) for r in rows if r["variant"] == "none"}
        theirs = {(r["patient"], int(r["class_idx"])): float(r["dice"]) for r in csv.DictReader(ref_csv.open()) if r["split"] == "val"}
        diff = max(abs(mine[k] - theirs[k]) for k in theirs)
        print(f"plain prediction vs eval/metrics_3d.csv: max |dice difference| = {diff:.2e}")
        if diff > 1e-3:  # a different batch size can flip a voxel on a tie; anything larger is a bug
            raise SystemExit("the plain TTA variant does not reproduce the run's own evaluation")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
