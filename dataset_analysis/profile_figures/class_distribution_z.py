#!/usr/bin/env python3
"""
Class distribution along the superior–inferior axis, to show where 2.5D context helps.

Every patient's slices are aligned to the most superior heart slice (z = 0 mm,
z increasing toward the head) and pooled in 2.5 mm bins, the median slice
spacing the 2.5D runs are resampled to. Writes into <out-dir>/class_distribution_z/:
  class_distribution_z.png/.pdf  top: % of patients whose slice contains each organ;
                                 bottom: % of the slice's pixels the organ covers,
                                 on slices that contain it (median and IQR).
                                 Shaded bands at each organ's median top and bottom
                                 edge show the ±2.5/±5/±10 mm reach of the 2.5D
                                 context (c1 with stride 1/2/4).
  class_distribution_z.csv       one row per patient, slice and organ.
  class_distribution_z_log.json  patients, spacing, orientation, git commit, time.

Reads only GT.nii.gz of every Patient_* folder in --data-dir.

Usage:
    python dataset_analysis/profile_figures/class_distribution_z.py \
        --data-dir data/segthor_train_full/train --out-dir figures/dataset_40
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from plot_style import (  # noqa: E402
    EARTH_LABEL_COLORS,
    apply_style,
    decorate,
    legend_below,
    tint,
)
from utils import CLASSES, NAMES  # noqa: E402

BIN_MM = 2.5
MIN_PATIENTS = 10  # bins covered by fewer patients are dropped
ANCHOR = 2  # heart: z = 0 at its most superior slice
# 2.5D context reach (mm) at 2.5 mm slices: c1 with stride 1, 2, 4; widest first
CONTEXT_MM = [(10.0, 0.18), (5.0, 0.30), (2.5, 0.45)]


def patient_rows(gt_path: Path) -> tuple[pd.DataFrame, dict]:
    """Per-slice pixel counts of every organ in one patient, aligned to the anchor organ.

    Args:
        gt_path: Patient's GT.nii.gz.

    Returns:
        One row per slice and organ (patient, slice, z_mm, label, pixels, pct_pixels),
        and the patient's geometry for the log.
    """
    nii = nib.load(gt_path)
    codes = "".join(nib.aff2axcodes(nii.affine))
    assert codes == "LPS", f"{gt_path}: orientation {codes}, expected LPS"
    seg = np.asarray(nii.dataobj)
    n_slices = seg.shape[2]
    # world z (mm, toward superior in LPS) of each slice centre
    z = (nii.affine @ np.c_[np.zeros((n_slices, 2)), np.arange(n_slices), np.ones(n_slices)].T)[2]
    pixels = {c: (seg == c).sum(axis=(0, 1)) for c in CLASSES}
    anchor = pixels[ANCHOR] > 0
    assert anchor.any(), f"{gt_path}: no {CLASSES[ANCHOR]} voxels to anchor on"
    z = z - z[anchor].max()
    pid = gt_path.parent.name
    n_px = seg.shape[0] * seg.shape[1]
    rows = pd.concat(
        pd.DataFrame(
            {
                "patient": pid,
                "slice": np.arange(n_slices),
                "z_mm": z,
                "label": c,
                "pixels": pixels[c],
                "pct_pixels": 100 * pixels[c] / n_px,
            }
        )
        for c in CLASSES
    )
    geometry = {
        "patient": pid,
        "orientation": codes,
        "spacing_mm": [float(s) for s in nii.header.get_zooms()[:3]],
        "shape": list(seg.shape),
        "labels_present": [int(c) for c in CLASSES if pixels[c].any()],
    }
    return rows, geometry


def binned(slices: pd.DataFrame) -> pd.DataFrame:
    """Pool slices into BIN_MM bins: presence across patients and size where present.

    Args:
        slices: Output rows of patient_rows for every patient.

    Returns:
        One row per bin and organ: z_mm (bin centre), n_patients covering the bin,
        presence_pct, and median/q25/q75 of pct_pixels over patients with the organ there.
    """
    df = slices.assign(z_mm=np.round(slices.z_mm / BIN_MM) * BIN_MM)
    # one value per patient per bin: present if any slice in the bin has it, size = mean over those slices
    per_patient = (
        df.groupby(["label", "z_mm", "patient"])
        .agg(present=("pixels", lambda p: (p > 0).any()), pct=("pct_pixels", lambda p: p[p > 0].mean()))
        .reset_index()
    )
    out = (
        per_patient.groupby(["label", "z_mm"])
        .agg(
            n_patients=("patient", "size"),
            presence_pct=("present", lambda p: 100 * p.mean()),
            pct_median=("pct", "median"),
            pct_q25=("pct", lambda p: p.quantile(0.25)),
            pct_q75=("pct", lambda p: p.quantile(0.75)),
        )
        .reset_index()
    )
    return out[out.n_patients >= MIN_PATIENTS]


def organ_edges(slices: pd.DataFrame) -> pd.DataFrame:
    """Median (over patients) most superior and most inferior z of every organ.

    Args:
        slices: Output rows of patient_rows for every patient.

    Returns:
        One row per label with columns top_mm and bottom_mm.
    """
    present = slices[slices.pixels > 0]
    per_patient = present.groupby(["label", "patient"]).z_mm.agg(top_mm="max", bottom_mm="min")
    return per_patient.groupby("label").median()


def draw(bins: pd.DataFrame, edges: pd.DataFrame, labels: list[int], n_patients: int, out_dir: Path) -> None:
    """Draw the presence and pixel-share panels with 2.5D context bands.

    Args:
        bins: Output of binned.
        edges: Output of organ_edges.
        labels: Labels with voxels in at least one patient.
        n_patients: Number of patients pooled.
        out_dir: Folder to write the figure into.
    """
    apply_style()
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True, height_ratios=[1, 1.15])

    for reach, amount in CONTEXT_MM:
        for c in labels:
            for edge in edges.loc[c, ["top_mm", "bottom_mm"]]:
                for ax in (top, bottom):
                    ax.axvspan(edge - reach, edge + reach, color=tint(EARTH_LABEL_COLORS[c], amount), lw=0, zorder=0)

    for c in labels:
        color = EARTH_LABEL_COLORS[c]
        b = bins[bins.label == c]
        top.plot(b.z_mm, b.presence_pct, color=color, lw=2.2, label=NAMES[c], zorder=3)
        b = b.dropna(subset=["pct_median"])
        bottom.fill_between(b.z_mm, b.pct_q25, b.pct_q75, color=tint(color, 0.35), lw=0, zorder=2)
        bottom.plot(b.z_mm, b.pct_median, color=color, lw=2.2, zorder=3)

    top.set_ylim(0, 102)
    top.set_ylabel("patients with organ\nin slice (%)")
    top.set_title("Where each organ appears", loc="left", fontsize=13, fontweight="bold")
    bottom.set_yscale("log")
    bottom.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    bottom.set_ylabel("organ pixels / slice pixels (%)\nlog scale")
    bottom.set_title(
        "How much of the slice it covers, where present (median, band = 25–75% of patients)",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    bottom.set_xlabel("z relative to top of the heart (mm)   ←  superior   ·   inferior  →")
    seen = bins[bins.presence_pct > 0].z_mm
    bottom.set_xlim(seen.min() - 15, seen.max() + 15)
    bottom.invert_xaxis()  # superior on the left, reading head to feet
    top.axvline(0, color="#999999", lw=0.8, ls="--", zorder=1)
    bottom.axvline(0, color="#999999", lw=0.8, ls="--", zorder=1)

    # empty fills, only so the legend shows the band shades (in grey, since each organ has its own tint)
    for reach, amount in CONTEXT_MM:
        top.fill_between([], [], color=tint("#8C8C8C", amount), lw=0, label=f"±{reach:g} mm (stride {int(reach / BIN_MM)})")
    legend_below(top, ncol=len(labels) + len(CONTEXT_MM))

    spans = (edges.top_mm - edges.bottom_mm).loc[labels]
    decorate(
        fig,
        f"Organs Span {spans.min():.0f}–{spans.max():.0f} mm Along z; a 2.5D Input Sees Only ±2.5–10 mm",
        subtitle=(
            f"{n_patients} patients aligned at the top of the heart, pooled in {BIN_MM:g} mm bins. "
            "Shaded bands mark each organ's median top and bottom edge, widened by the reach of a "
            "3-slice 2.5D input (c1) at stride 1, 2 and 4."
        ),
        footnote_text=(
            "Data: data/segthor_train_full/train, corrected labels (aorta separate). Presence = share of patients "
            "whose scan covers the bin that have the organ in it; pixel share is over the full 512×512 slice, "
            f"before ROI crop. Heart presence jumps to 100% at z = 0 by construction (it is the anchor). Bins covered by fewer than {MIN_PATIENTS} patients are left out."
        ),
    )
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"class_distribution_z.{ext}")
    plt.close(fig)
    print(f"wrote {out_dir / 'class_distribution_z.png'}")


def main():
    """Measure every patient, write the table and log, and draw the figure."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/segthor_train_full/train"))
    ap.add_argument("--out-dir", type=Path, default=Path("figures/dataset_40"))
    args = ap.parse_args()
    out_dir = args.out_dir / "class_distribution_z"
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()

    tables, geometry, failed = [], [], []
    for gt_path in sorted(args.data_dir.glob("Patient_*/GT.nii.gz")):
        try:
            rows, geo = patient_rows(gt_path)
        except Exception as e:  # log and skip one bad patient instead of losing the run
            print(f"[skip] {gt_path}: {type(e).__name__}: {e}")
            failed.append({"patient": gt_path.parent.name, "error": f"{type(e).__name__}: {e}"})
            continue
        print(f"{geo['patient']}: {geo['orientation']} spacing {geo['spacing_mm']} shape {geo['shape']}")
        tables.append(rows)
        geometry.append(geo)
    slices = pd.concat(tables, ignore_index=True)
    slices.to_csv(out_dir / "class_distribution_z.csv", index=False)

    labels = [c for c in CLASSES if (slices[slices.label == c].pixels > 0).any()]
    bins = binned(slices[slices.label.isin(labels)])
    edges = organ_edges(slices)
    draw(bins, edges, labels, len(geometry), out_dir)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    log = {
        "data_dir": str(args.data_dir),
        "git_commit": commit,
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(),
        "bin_mm": BIN_MM,
        "min_patients": MIN_PATIENTS,
        "anchor": CLASSES[ANCHOR],
        "labels_drawn": [CLASSES[c] for c in labels],
        "organ_edges_mm": {CLASSES[c]: edges.loc[c].to_dict() for c in labels},
        "patients": geometry,
        "failed": failed,
    }
    (out_dir / "class_distribution_z_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps({k: log[k] for k in ("labels_drawn", "organ_edges_mm", "failed")}, indent=2))


if __name__ == "__main__":
    main()
