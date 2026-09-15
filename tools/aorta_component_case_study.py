#!/usr/bin/env python3
"""
Follow-up to the connected_components.png check: aorta shows 2 components
(scipy.ndimage.label default, 6-connectivity) in Patient_02/09/16.

Verdict: connectivity-definition artifact, not a real annotation gap or an
anatomical split. The aortic arch curves so that, in most axial slices, the
ascending and descending aorta appear as two separate round cross-sections
(normal anatomy, visible in every thoracic CT) -- they are one continuous
vessel only through the arch. At the exact junction slices the 3D voxel
connection between the ascending-side and descending-side voxel columns is
corner/edge-only, which scipy's default structure (face-adjacency only, i.e.
6-connectivity in 3D) does not count as connected. Full 26-connectivity
(structure=np.ones((3,3,3))) merges them back into 1 component for all three
patients -- see connected_components.png's updated panel.

Usage:
    python tools/aorta_component_case_study.py --out-dir figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import label

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_style, decorate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/segthor_part1/train"))
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    ap.add_argument("--patient", default="Patient_02")
    args = ap.parse_args()
    apply_style()

    ct = np.asarray(nib.load(str(args.data_dir / args.patient / f"{args.patient}.nii.gz")).dataobj)
    seg = np.asarray(nib.load(str(args.data_dir / args.patient / "GT.nii.gz")).dataobj)
    mask = seg == 1

    struct26 = np.ones((3, 3, 3), dtype=int)
    _, n6 = label(mask)
    _, n26 = label(mask, structure=struct26)

    # pick 4 representative z-slices: one blob (arch), two just-split, mid-split, back to one
    z_candidates = list(range(mask.shape[2]))
    blob_counts = []
    for z in z_candidates:
        sl = mask[:, :, z]
        if sl.any():
            _, n = label(sl)
        else:
            n = 0
        blob_counts.append(n)
    blob_counts = np.array(blob_counts)
    two_blob_z = np.where(blob_counts == 2)[0]
    one_blob_before = np.where((blob_counts == 1) & (np.arange(len(blob_counts)) < two_blob_z.min()))[0]
    one_blob_after = np.where((blob_counts == 1) & (np.arange(len(blob_counts)) > two_blob_z.max()))[0]
    z_show = [one_blob_before[-1] if len(one_blob_before) else two_blob_z.min(),
              two_blob_z[len(two_blob_z) // 4],
              two_blob_z[len(two_blob_z) // 2],
              one_blob_after[0] if len(one_blob_after) else two_blob_z.max()]
    titles = ["Below arch: 1 blob\n(single descending tube)" if len(one_blob_before) else "z=%d" % z_show[0],
              "Arch region: 2 blobs\n(ascending + descending)",
              "Arch region: 2 blobs",
              "Above arch: 1 blob\n(ascending + arch fused)" if len(one_blob_after) else "z=%d" % z_show[3]]

    fig, axes = plt.subplots(1, 4, figsize=(13, 4))
    for ax, z, title in zip(axes, z_show, titles):
        crop = mask[180:330, 150:330, z]
        ct_crop = ct[180:330, 150:330, z]
        ax.imshow(ct_crop.T, cmap="gray", vmin=-200, vmax=400, origin="lower")
        overlay = np.ma.masked_where(~crop.T, crop.T)
        ax.imshow(overlay, cmap="autumn", alpha=0.6, origin="lower")
        ax.set_title(f"{title}\n(z={z})", fontsize=9)
        ax.axis("off")

    decorate(fig, f"{args.patient}: The Aorta Never Actually Splits -- The Arch Just Curves",
             subtitle=f"scipy.ndimage.label: 6-connectivity finds {n6} components, 26-connectivity finds {n26}",
             footnote_text="Ascending + descending aorta are two round cross-sections in most axial slices -- "
                            "normal thoracic anatomy. They join only at the arch, where the 3D voxel link is "
                            "corner-only, so face-only (6-) connectivity misses it. Full (26-) connectivity "
                            "correctly reads this as one vessel.")
    fig.savefig(args.out_dir / f"aorta_component_case_study_{args.patient}.png")
    plt.close(fig)
    print(f"n6={n6} n26={n26} z_show={z_show}")
    print(f"Wrote aorta_component_case_study_{args.patient}.png")


if __name__ == "__main__":
    main()
