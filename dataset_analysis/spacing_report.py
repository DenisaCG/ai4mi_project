"""Report for data.preprocess.resample: native vs median-spacing volumes of a config, from the built dataset.
Writes a per-patient table and two figures: spacing/shape overview, and one example patient before/after.
Usage: python dataset_analysis/spacing_report.py --config configs/segthor_enet_ce_corrected_median_spacing.yaml"""
import argparse
import copy
import csv
import sys
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from plot_style import EARTH  # noqa: E402
from src.config import load_config, resolve_preprocess  # noqa: E402
from src.data import ensure_sliced  # noqa: E402

TRAIN, VAL, MED = EARTH[1], EARTH[2], "#171717"  # earth palette: teal train, ochre val, ink target line
VIEW = (-200, 300)  # display window only, not used for training


def zooms(path: Path) -> np.ndarray:
    return np.abs(nib.load(str(path)).affine[:3, :3]).sum(0)


def patient_rows(source_dir: Path, resampled_dir: Path, train_ids: set[str]) -> list[dict]:
    rows = []
    for p in sorted(resampled_dir.glob("Patient_*")):
        native, res = nib.load(str(source_dir / "train" / p.name / f"{p.name}.nii.gz")), nib.load(str(p / f"{p.name}.nii.gz"))
        dn, dr = zooms(source_dir / "train" / p.name / f"{p.name}.nii.gz"), zooms(p / f"{p.name}.nii.gz")
        rows.append({"patient": p.name, "split": "train" if p.name in train_ids else "val",
                     "native_dx": dn[0], "native_dz": dn[2], "native_shape": native.shape,
                     "res_dx": dr[0], "res_dz": dr[2], "res_shape": res.shape,
                     "native_mm_per_px": native.shape[0] * dn[0] / 256, "res_mm_per_px": res.shape[0] * dr[0] / 256})
    return rows


def overview(rows: list[dict], target: list[float], path: Path, title: str) -> None:
    names = [r["patient"].replace("Patient_", "") for r in rows]
    colors = [TRAIN if r["split"] == "train" else VAL for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5))
    panels = [("In-plane spacing dx (mm)", [r["native_dx"] for r in rows], target[0]),
              ("Slice thickness dz (mm)", [r["native_dz"] for r in rows], target[2]),
              ("Axial slices per volume", [r["native_shape"][2] for r in rows], None),
              ("Final PNG mm per pixel (512·dx / 256)", [r["native_mm_per_px"] for r in rows], None)]
    after = [None, None, [r["res_shape"][2] for r in rows], [r["res_mm_per_px"] for r in rows]]
    for ax, (label, vals, tgt), aft in zip(axes.ravel(), panels, after):
        ax.bar(x - (0.2 if aft else 0), vals, width=0.4 if aft else 0.7, color=colors, label="native" if aft else None)
        if aft:
            ax.bar(x + 0.2, aft, width=0.4, color=colors, alpha=.45, hatch="//", edgecolor="white", label="after median spacing")
            ax.legend(loc="upper right")
        if tgt is not None:
            ax.axhline(tgt, color=MED, ls="--", lw=1.5)
            ax.text(len(rows) - .5, tgt, f" target {tgt:.3f}", va="bottom", ha="right", color=MED, fontsize=11)
        ax.set_xticks(x, names)
        ax.set_xlabel("patient")
        ax.set_title(label)
    axes[0, 0].plot([], [], "s", color=TRAIN, label="train")
    axes[0, 0].plot([], [], "s", color=VAL, label="val")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def show(ax, arr: np.ndarray, spacing: tuple[float, float], title: str, view: bool = True) -> None:
    """arr (rows, cols) shown at physical size: extent in mm."""
    h, w = arr.shape
    kw = {"vmin": VIEW[0], "vmax": VIEW[1]} if view else {"vmin": 0, "vmax": 255}
    ax.imshow(arr, cmap="gray", extent=[0, w * spacing[1], h * spacing[0], 0], **kw)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("mm")


def example(row: dict, source_dir: Path, resampled_dir: Path, roots: dict[str, Path], path: Path) -> None:
    pid = row["patient"]
    nat, res = nib.load(str(source_dir / "train" / pid / f"{pid}.nii.gz")), nib.load(str(resampled_dir / pid / f"{pid}.nii.gz"))
    gt = np.asarray(nib.load(str(source_dir / "train" / pid / "GT.nii.gz")).dataobj)
    ct_n, ct_r = np.asarray(nat.dataobj), np.asarray(res.dataobj)
    dn, dr = zooms(source_dir / "train" / pid / f"{pid}.nii.gz"), zooms(resampled_dir / pid / f"{pid}.nii.gz")
    cx, _, cz = (int(np.rint(c.mean())) for c in np.nonzero(gt == 2))  # heart centre, same physical place in both
    to_res = lambda i, a: int(np.clip(np.rint((i + .5) * dn[a] / dr[a] - .5), 0, ct_r.shape[a] - 1))
    rx, rz = to_res(cx, 0), to_res(cz, 2)
    split = row["split"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    for r, (label, ct, d, ix, iz, root) in enumerate([("native", ct_n, dn, cx, cz, roots["native"]),
                                                       ("median spacing", ct_r, dr, rx, rz, roots["median"])]):
        show(axes[r, 0], ct[:, :, iz].T, (d[1], d[0]), f"{label}: axial z={iz}, {ct.shape[0]}x{ct.shape[1]} at {d[0]:.2f} mm")
        show(axes[r, 1], ct[ix].T, (d[2], d[1]), f"{label}: sagittal, {ct.shape[2]} slices at {d[2]:.2f} mm")
        png = root / split / "img" / f"{pid}_{iz:04d}.png"
        show(axes[r, 2], np.array(Image.open(png)), (row["native_mm_per_px"],) * 2, f"{label}: the 256x256 PNG the network sees", view=False)
    fig.suptitle(f"{pid} ({split}): same heart-centred location before and after resampling, drawn at physical size", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="a config with data.preprocess.resample")
    ap.add_argument("--out", default=str(REPO / "dataset_analysis" / "results" / "median_spacing"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    p = cfg["data"]["preprocess"]
    assert p.get("resample"), "config has no data.preprocess.resample"
    native = copy.deepcopy(cfg)
    native["data"]["preprocess"].pop("resample")
    resolve_preprocess(native)
    for c in (native, cfg):
        ensure_sliced(c)
    root_n, root_m = REPO / native["data"]["root"], REPO / cfg["data"]["root"]
    source_dir = REPO / p["source_dir"]
    train_ids = {f.name.rsplit("_", 1)[0] for f in (root_m / "train" / "gt").glob("*.png")}
    rows = patient_rows(source_dir, root_m / "resampled" / "train", train_ids)
    target = np.median([[r["res_dx"], r["res_dx"], r["res_dz"]] for r in rows if r["split"] == "train"], axis=0).tolist()
    print("native dataset:", root_n, "\nmedian-spacing dataset:", root_m)
    for r in rows:
        print(f"{r['patient']} {r['split']:5s} dx {r['native_dx']:.3f}->{r['res_dx']:.3f} dz {r['native_dz']:.2f}->{r['res_dz']:.2f} "
              f"shape {r['native_shape']}->{r['res_shape']} mm/px {r['native_mm_per_px']:.3f}->{r['res_mm_per_px']:.3f}")
    print(f"target spacing ~ {np.round(target, 3).tolist()} mm | max |mm/px change| in final PNG: "
          f"{max(abs(r['res_mm_per_px'] / r['native_mm_per_px'] - 1) for r in rows):.4f} (relative)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "spacing_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    overview(rows, target, out / "spacing_overview.png", f"Median-spacing resampling ({cfg['experiment']})")
    for tag, axis, t in (("inplane", "dx", target[0]), ("slice_thickness", "dz", target[2])):
        worst = max(rows, key=lambda r: abs(np.log(r[f"native_{axis}"] / t)))
        print(f"example patient (largest {axis} change):", worst["patient"])
        example(worst, source_dir, root_m / "resampled" / "train", {"native": root_n, "median": root_m},
                out / f"spacing_example_{tag}.png")
    print("saved", out)


if __name__ == "__main__":
    main()
