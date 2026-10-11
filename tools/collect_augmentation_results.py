"""Collect every augmentation experiment's 3D validation metrics into three CSVs (per run, per arm, paired deltas).

Usage: python -I tools/collect_augmentation_results.py   (reads metrics/<exp>_fold*/seed*/summary.json, writes docs/augmentation_results/)
"""
import csv
import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "augmentation_results"
P = "full_cv4_"
METRICS = ["dice_fg", "dice_esophagus", "dice_heart", "dice_trachea", "dice_aorta", "hd95_fg", "assd_fg"]

# (experiment, label, model, group, augmentation, epochs)
ARMS = [
    ("enet_ce", "BASE ENet, no aug", "BASE ENet", "control", "none", 25),
    ("enet_ce_augmented", "BASE ENet, joint rot+scale+shift", "BASE ENet", "joint", "rot+scale+shift p0.5", 25),
    ("enet_dice_ce", "ENet, no aug", "Improved ENet", "control", "none", 25),
    ("enet_dice_ce_rotation", "ENet, rotation", "Improved ENet", "single (seed 0)", "rotation +-10 p0.5", 25),
    ("enet_dice_ce_scaling", "ENet, scaling", "Improved ENet", "single (seed 0)", "scaling 0.9-1.1 p0.5", 25),
    ("enet_dice_ce_shift", "ENet, shift", "Improved ENet", "single (seed 0)", "shift 10% p0.5", 25),
    ("enet_dice_ce_shear", "ENet, shear", "Improved ENet", "single (seed 0)", "shear +-5 p0.5", 25),
    ("enet_dice_ce_gaussian_noise", "ENet, Gaussian noise", "Improved ENet", "single (seed 0)", "noise sigma 0.05 p0.5", 25),
    ("enet_dice_ce_blur", "ENet, Gaussian blur", "Improved ENet", "single (seed 0)", "blur sigma 0.5-1 p0.2", 25),
    ("enet_dice_ce_brightness_contrast", "ENet, brightness/contrast", "Improved ENet", "single (seed 0)", "b/c 0.75-1.25 p0.15", 25),
    ("enet_dice_ce_gamma", "ENet, gamma", "Improved ENet", "single (seed 0)", "gamma 0.7-1.5 p0.2", 25),
    ("enet_dice_ce_augmented", "ENet, joint rot+scale+shift", "Improved ENet", "joint", "rot+scale+shift p0.5", 25),
    ("enet_dice_ce_nnunet_rot_scale_shift", "ENet, nnU-Net-rate rot+scale+shift (seed 0)", "Improved ENet", "joint variant",
     "rot +-10 p0.2, scale 0.7-1.4 p0.2, shift 2% p0.2", 25),
    ("enet_dice_ce_nnunet_rot_scale_shift_noise", "ENet, nnU-Net-rate + noise (seed 0)", "Improved ENet", "joint variant",
     "same + noise sigma 0.05 p0.1", 25),
    ("enet_dice_ce_rot_scale_shift_elastic", "ENet, joint + elastic", "Improved ENet", "joint variant",
     "rot+scale+shift p0.5 + elastic p0.5", 25),
    ("enet_dice_ce_strong_geometric", "ENet, strong geometric", "Improved ENet", "strong",
     "rot +-15, scale 0.8-1.2, shift 10%, p0.7", 25),
    ("enet_dice_ce_strong_geometric_elastic", "ENet, strong geometric + elastic", "Improved ENet", "strong variant",
     "strong + elastic p0.5", 25),
    ("enet_dice_ce_extreme_geometric", "ENet, extreme geometric", "Improved ENet", "strong variant",
     "rot +-20, scale 0.7-1.3, shift 10%, p0.9", 25),
    ("enet_dice_ce_strong_geometric_50ep", "ENet, strong geometric, 50 epochs", "Improved ENet", "strong variant",
     "strong, 50 epochs", 50),
    ("resenc_ds_unet_dice_ce", "ResEnc DS U-Net, no aug", "ResEnc DS U-Net", "control", "none", 25),
    ("resenc_ds_unet_dice_ce_augmented", "ResEnc DS U-Net, joint rot+scale+shift", "ResEnc DS U-Net", "joint",
     "rot+scale+shift p0.5", 25),
    ("resenc_ds_unet_dice_ce_strong_geometric", "ResEnc DS U-Net, strong geometric", "ResEnc DS U-Net", "strong",
     "rot +-15, scale 0.8-1.2, shift 10%, p0.7", 25),
]
# (arm, reference) pairs for paired deltas; matched on (fold, seed)
PAIRS = [
    ("enet_ce_augmented", "enet_ce"), ("enet_dice_ce_augmented", "enet_dice_ce"),
    ("enet_dice_ce_rot_scale_shift_elastic", "enet_dice_ce_augmented"),
    ("enet_dice_ce_strong_geometric", "enet_dice_ce_augmented"), ("enet_dice_ce_strong_geometric", "enet_dice_ce"),
    ("enet_dice_ce_strong_geometric_elastic", "enet_dice_ce_strong_geometric"),
    ("enet_dice_ce_extreme_geometric", "enet_dice_ce_strong_geometric"),
    ("enet_dice_ce_strong_geometric_50ep", "enet_dice_ce_strong_geometric"),
    ("enet_dice_ce_nnunet_rot_scale_shift_noise", "enet_dice_ce_nnunet_rot_scale_shift"),
    ("resenc_ds_unet_dice_ce_augmented", "resenc_ds_unet_dice_ce"),
    ("resenc_ds_unet_dice_ce_strong_geometric", "resenc_ds_unet_dice_ce_augmented"),
    ("resenc_ds_unet_dice_ce_strong_geometric", "resenc_ds_unet_dice_ce"),
]
for single in ("rotation", "scaling", "shift", "shear", "gaussian_noise", "blur", "brightness_contrast", "gamma"):
    PAIRS.append((f"enet_dice_ce_{single}", "enet_dice_ce"))


def load(exp):
    runs = {}
    for f in (REPO / "metrics").glob(f"{P}{exp}_fold*/seed*/summary.json"):
        fold, seed = int(f.parts[-3].rsplit("fold", 1)[1]), int(f.parts[-2][4:])
        ev = json.loads(f.read_text()).get("eval")
        if ev:
            runs[(fold, seed)] = {m: ev[f"val_{m}"] for m in METRICS} | {"best_epoch": ev.get("best_epoch")}
    return runs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {a[0]: load(a[0]) for a in ARMS}
    with open(OUT / "per_run.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["experiment", "label", "model", "group", "fold", "seed"] + METRICS + ["best_epoch"])
        for exp, label, model, group, _, _ in ARMS:
            for (fold, seed), r in sorted(data[exp].items()):
                w.writerow([P + exp, label, model, group, fold, seed] + [f"{r[m]:.6f}" for m in METRICS] + [r["best_epoch"]])
    with open(OUT / "per_arm.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["experiment", "label", "model", "group", "augmentation", "epochs", "n_runs"]
                   + [f"{m}_{s}" for m in METRICS for s in ("mean", "sd")])
        for exp, label, model, group, aug, epochs in ARMS:
            rs = list(data[exp].values())
            if not rs:
                continue
            row = [P + exp, label, model, group, aug, epochs, len(rs)]
            for m in METRICS:
                v = [r[m] for r in rs]
                row += [f"{st.mean(v):.6f}", f"{st.stdev(v) if len(v) > 1 else 0:.6f}"]
            w.writerow(row)
    with open(OUT / "paired_deltas.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "reference", "n_pairs", "dice_fg_delta_mean", "dice_fg_delta_sd", "wins", "hd95_fg_delta_mean", "assd_fg_delta_mean"])
        for a, b in PAIRS:
            keys = sorted(set(data[a]) & set(data[b])) if a in data and b in data else []
            if not keys:
                continue
            d = [data[a][k]["dice_fg"] - data[b][k]["dice_fg"] for k in keys]
            w.writerow([P + a, P + b, len(keys), f"{st.mean(d):.6f}", f"{st.stdev(d) if len(d) > 1 else 0:.6f}",
                        sum(x > 0 for x in d),
                        f"{st.mean(data[a][k]['hd95_fg'] - data[b][k]['hd95_fg'] for k in keys):.4f}",
                        f"{st.mean(data[a][k]['assd_fg'] - data[b][k]['assd_fg'] for k in keys):.4f}"])


if __name__ == "__main__":
    main()
