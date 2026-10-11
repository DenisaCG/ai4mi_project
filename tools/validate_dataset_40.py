"""Audit the 40-patient dataset tables: row counts, unique ids, physical bounds, independent recounts, input immutability."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import generate_binary_structure, label

sys.path.append(str(Path(__file__).resolve().parent))
from dataset_profile import LABELS

REPO = Path(__file__).resolve().parents[1]
N_PATIENTS, N_ORIGINAL = 40, 20
N_PAIRS = len(LABELS) * (len(LABELS) - 1) // 2
HU_REGIONS = 1 + len(LABELS)  # all voxels + one per organ
N_FOLDS = 4
EXPECTED_FIGURES = (
    "geometry_per_patient.png",
    "hu_per_patient.png",
    "organ_volume_per_patient.png",
    "esophagus_components_per_patient.png",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dir", type=Path, default=REPO / "data/segthor_train_full/train"
    )
    ap.add_argument(
        "--midterm-dir", type=Path, default=REPO / "data/segthor_part1_corrected/train"
    )
    ap.add_argument("--out-dir", type=Path, default=REPO / "figures/dataset_40")
    args = ap.parse_args()
    out = args.out_dir
    geo = pd.read_csv(out / "patient_geometry.csv")
    hu = pd.read_csv(out / "patient_intensity.csv")
    lab = pd.read_csv(out / "patient_labels.csv")
    adj = pd.read_csv(out / "patient_adjacency.csv")
    wide = pd.read_csv(out / "patient_wide.csv")
    folds = pd.read_csv(out / "fold_preprocessing.csv")
    check = pd.read_csv(out / "midterm_vs_full_first20.csv")
    cmp = pd.read_csv(out / "comparison.csv")

    # Row counts and ids
    on_disk = sorted(p.name for p in args.data_dir.glob("Patient_*"))
    midterm = sorted(p.name for p in args.midterm_dir.glob("Patient_*"))
    require(
        len(on_disk) == N_PATIENTS,
        f"{len(on_disk)} patients on disk, expected {N_PATIENTS}",
    )
    require(
        len(midterm) == N_ORIGINAL and set(midterm) <= set(on_disk),
        "midterm ids are not a subset of the full set",
    )
    for name, df, per_patient in (
        ("geometry", geo, 1),
        ("wide", wide, 1),
        ("intensity", hu, HU_REGIONS),
        ("labels", lab, len(LABELS)),
        ("adjacency", adj, N_PAIRS),
    ):
        require(
            len(df) == per_patient * N_PATIENTS,
            f"{name}: {len(df)} rows, expected {per_patient * N_PATIENTS}",
        )
        require(
            sorted(df.patient.unique()) == on_disk,
            f"{name}: patient ids differ from the folder",
        )
    require(geo.patient.is_unique and wide.patient.is_unique, "duplicate patient ids")
    require(
        not lab.duplicated(["patient", "label"]).any(), "duplicate patient/label rows"
    )
    require(
        not hu.duplicated(["patient", "region"]).any(), "duplicate patient/region rows"
    )
    require(
        not adj.duplicated(["patient", "pair"]).any(), "duplicate patient/pair rows"
    )
    require(
        geo.group.value_counts().to_dict()
        == {"original_20": N_ORIGINAL, "added_20": N_PATIENTS - N_ORIGINAL},
        "group sizes are not 20/20",
    )
    require(
        set(geo.loc[geo.group == "original_20", "patient"]) == set(midterm),
        "original_20 is not the midterm id set",
    )
    require(
        len(folds) == N_FOLDS + 1,
        f"{len(folds)} rows in fold_preprocessing, expected {N_FOLDS} folds + midterm",
    )
    require(len(check) == N_ORIGINAL, "midterm check does not cover 20 patients")

    # Physical bounds
    require(
        geo.pixel_size_left_right_mm.between(0.3, 2.0).all()
        and geo.pixel_size_anterior_posterior_mm.between(0.3, 2.0).all(),
        "in-plane pixel size outside 0.3-2 mm",
    )
    require(
        geo.slice_spacing_superior_inferior_mm.between(0.5, 10).all(),
        "slice spacing outside 0.5-10 mm",
    )
    require(
        (geo[["matrix_left_right", "matrix_anterior_posterior", "n_slices"]] > 0)
        .all()
        .all(),
        "non-positive matrix size",
    )
    require(
        (
            geo.voxels
            == geo.matrix_left_right * geo.matrix_anterior_posterior * geo.n_slices
        ).all(),
        "voxel count != matrix product",
    )
    require(
        np.allclose(
            geo.scan_length_mm, geo.n_slices * geo.slice_spacing_superior_inferior_mm
        ),
        "scan length != slices x spacing",
    )
    require(
        geo.scan_length_mm.between(100, 1500).all(), "scan length outside 100-1500 mm"
    )
    for axis_codes in geo.axis_codes.unique():
        require(
            axis_codes[0] in "LR" and axis_codes[1] in "AP" and axis_codes[2] in "SI",
            f"axes {axis_codes} do not map to left-right, anterior-posterior, superior-inferior",
        )
    require(
        (geo.axis_codes == geo.label_axis_codes).all(),
        "image and label orientation differ",
    )
    require(
        hu["p0.5"].between(-1100, 4000).all()
        and hu["p99.5"].between(-1100, 4000).all(),
        "HU percentile outside -1100..4000",
    )
    ordered = hu.dropna(subset=["median"])
    require(
        (
            (ordered["p0.5"] <= ordered["median"])
            & (ordered["median"] <= ordered["p99.5"])
        ).all(),
        "p0.5 <= median <= p99.5 violated",
    )
    require((hu["std"].dropna() >= 0).all(), "negative HU std")
    require(
        (lab.share_of_scan_pct.between(0, 100)).all()
        and (lab.share_of_foreground_pct.between(0, 100)).all(),
        "share outside 0-100 %",
    )
    require(
        np.allclose(lab.groupby("patient").share_of_foreground_pct.sum(), 100),
        "foreground shares do not sum to 100 %",
    )
    require(
        (lab.present == (lab.voxels > 0)).all(),
        "presence flag disagrees with voxel count",
    )
    require(
        ((lab.components_face >= 1) == (lab.present == 1)).all(),
        "component count disagrees with presence",
    )
    require(
        (lab.components_face_edge_corner <= lab.components_face).all(),
        "26-connected pieces exceed 6-connected pieces",
    )
    require(
        lab.share_slices_2plus_components_pct.dropna().between(0, 100).all(),
        "2+ component share outside 0-100 %",
    )
    require(
        (adj.shared_face_area_mm2 >= 0).all()
        and (adj.touching == (adj.shared_face_area_mm2 > 0)).all(),
        "adjacency inconsistent",
    )
    require(
        folds.roi_window_px.mod(folds.roi_multiple_px).eq(0).all(),
        "ROI window is not a multiple of its rounding step",
    )
    require(folds.window_lo_hu.lt(folds.window_hi_hu).all(), "HU window lo >= hi")
    require(
        cmp.quantity.is_unique and len(cmp) == len(wide.columns) - 2,
        "comparison does not cover every wide-table quantity once",
    )

    # Independent recount from the NIfTI files (voxel counts, volume, components for the esophagus)
    face = generate_binary_structure(3, 1)
    for _, g in geo.iterrows():
        pid = g.patient
        seg_img = nib.load(args.data_dir / pid / "GT.nii.gz")
        ct_img = nib.load(args.data_dir / pid / f"{pid}.nii.gz")
        require(
            ct_img.shape
            == seg_img.shape
            == (g.matrix_left_right, g.matrix_anterior_posterior, g.n_slices),
            f"{pid}: shape mismatch",
        )
        require(
            np.allclose(ct_img.affine, seg_img.affine),
            f"{pid}: CT and GT affines differ",
        )
        require(
            "".join(nib.aff2axcodes(ct_img.affine)) == g.axis_codes,
            f"{pid}: axis codes differ",
        )
        seg = np.asarray(seg_img.dataobj)
        zooms = seg_img.header.get_zooms()[:3]
        require(
            math.isclose(g.slice_spacing_superior_inferior_mm, zooms[2], rel_tol=1e-6),
            f"{pid}: slice spacing differs",
        )
        counts = np.bincount(seg.ravel().astype(np.int64), minlength=max(LABELS) + 1)
        require(set(np.unique(seg)) <= {0, *LABELS}, f"{pid}: unexpected label values")
        for k in LABELS:
            row = lab[(lab.patient == pid) & (lab.label == k)].iloc[0]
            require(
                int(row.voxels) == int(counts[k]),
                f"{pid} label {k}: voxel count differs from recount",
            )
            require(
                math.isclose(
                    row.volume_ml,
                    counts[k] * float(np.prod(zooms)) / 1000,
                    rel_tol=1e-6,
                ),
                f"{pid} label {k}: volume differs",
            )
        pieces = label(seg == 1, structure=face)[1]
        require(
            int(lab[(lab.patient == pid) & (lab.label == 1)].components_face.iloc[0])
            == pieces,
            f"{pid}: esophagus component recount differs",
        )
    require(
        (lab[lab.label == max(LABELS)].present == 1).all(),
        "aorta missing in some patient of the corrected set",
    )

    # Inputs unchanged since the tables were built
    inputs = pd.read_csv(out / "inputs.csv")
    for r in inputs.itertuples():
        stat = Path(r.path).stat()
        require(
            stat.st_size == r.size_bytes and stat.st_mtime_ns == r.mtime_ns,
            f"input changed since analysis: {r.path}",
        )

    figures = sorted(p.name for p in out.glob("*.png"))
    require(
        figures == sorted(EXPECTED_FIGURES),
        f"figures differ from the expected set: {figures}",
    )
    report = {
        "status": "passed",
        "patients": len(on_disk),
        "geometry_rows": len(geo),
        "intensity_rows": len(hu),
        "label_rows": len(lab),
        "adjacency_rows": len(adj),
        "comparison_quantities": len(cmp),
        "inputs_checked": len(inputs),
        "figures": figures,
        "midterm_identical_patients": int(
            check[
                [
                    "geometry_equal",
                    "label_voxels_equal",
                    "components_equal",
                    "histograms_equal",
                ]
            ]
            .all(axis=1)
            .sum()
        ),
    }
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
