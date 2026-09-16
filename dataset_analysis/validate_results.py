"""Audit generated table identities, physical quantities, metric bounds and input immutability."""
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np

from style import FIGURES, COLORS
from figures import select_shape_examples
from utils import parser, paths, read_csv


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    args = parser(__doc__).parse_args()
    _, original, _, output = paths(args)
    tables = output / "tables"
    inventory = read_csv(tables / "patient_inventory.csv")
    shapes = read_csv(tables / "shape_descriptors_3d.csv")
    run = json.loads((output / "dataset_run.json").read_text())
    organs = read_csv(tables / "patient_class_stats.csv")
    slices = read_csv(tables / "slice_class_stats.csv")
    baseline = read_csv(tables / "baseline_slice_metrics.csv")
    patients = read_csv(tables / "baseline_patient_metrics.csv")
    ids = {r["patient_id"] for r in inventory}
    inv = {r["patient_id"]: r for r in inventory}
    require(len(ids) == len(inventory), "Duplicate patients")
    if not run["subset"]:
        require(ids == {p.parent.name for p in original.glob("Patient_*/GT.nii.gz")}, "Original patient coverage mismatch")
    require(len(shapes) == 3 * len(ids), "Incorrect shape descriptor row count")
    shape_map = {(r["patient_id"], int(r["class_id"])): r for r in shapes}
    require(set(shape_map) == {(p, k) for p in ids for k in (1, 2, 3)}, "Invalid shape patient/organ keys")
    # Independent checks use per-axis marginal counts rather than coordinate means.
    for patient in sorted(ids):
        nii = nib.load(original / patient / "GT.nii.gz")
        data = np.asanyarray(nii.dataobj)
        require(nib.aff2axcodes(nii.affine) == ("L", "P", "S"), "Revise scan-position captions for non-LPS data")
        require(np.allclose(nii.affine[2, :2], 0) and nii.affine[2, 2] > 0,
                "This scan-position figure requires an axial superior-pointing slice axis")
        require(nii.header.get_xyzt_units()[0] == "mm", "Physical units are not millimetres")
        for k in (1, 2, 3):
            row = shape_map[patient, k]
            mask = data == k
            count = int(np.count_nonzero(mask))
            require(count > 0, f"Unexpected empty annotated organ: {patient}, {k}")
            marginals = [np.count_nonzero(mask, axis=tuple(j for j in range(3) if j != a)) for a in range(3)]
            centre = np.array([np.dot(np.arange(len(m)), m) / count for m in marginals])
            world = nii.affine[:3, :3] @ centre + nii.affine[:3, 3]
            expected_volume = count * float(np.prod(nii.header.get_zooms()))
            require(int(row["voxel_count"]) == count and math.isclose(float(row["volume_mm3"]), expected_volume, rel_tol=1e-6),
                    "Invalid shape volume")
            require(math.isclose(float(row["volume_ml"]), expected_volume / 1000, rel_tol=1e-6), "Invalid mL conversion")
            require(np.allclose([float(row[f"centroid_{a}"]) for a in "ijk"], centre), "Invalid voxel centroid")
            require(np.allclose([float(row[f"centroid_world_{a}_mm"]) for a in "xyz"], world), "Invalid affine centroid transform")
            positive = np.flatnonzero(marginals[2])
            expected_extent = (positive[-1] - positive[0] + 1) * nii.header.get_zooms()[2]
            require(math.isclose(float(row["si_extent_mm"]), expected_extent, rel_tol=1e-6), "Invalid SI extent")
            normalized = centre[2] / (data.shape[2] - 1) if data.shape[2] > 1 else .5
            require(0 <= float(row["normalized_si_centroid"]) <= 1 and math.isclose(float(row["normalized_si_centroid"]), normalized),
                    "Invalid normalized SI centroid")
            require(row["axis_codes"] == "LPS", "Incorrect recorded orientation")
    require(len(organs) == 3 * len(ids), "Unexpected patient-class count")
    require(len(slices) == 3 * sum(int(r["num_slices"]) for r in inventory), "Unexpected slice-class count")
    require(len(baseline) == 3 * sum(int(r["num_slices"]) for r in inventory if r["split"] == "val"), "Unexpected baseline count")
    for rows, keys in ((organs, ("patient_id", "class_id")),
                       (slices, ("patient_id", "slice_index", "class_id")),
                       (baseline, ("patient_id", "slice_index", "class_id")),
                       (patients, ("patient_id", "class_id"))):
        require(len({tuple(r[k] for k in keys) for r in rows}) == len(rows), "Duplicate table keys")
        require(all(r["class_id"] in ("1", "2", "3") for r in rows), "Unexpected class in analysis")
    for r in organs:
        i = inv[r["patient_id"]]
        expected = int(r["voxel_count"]) * float(i["voxel_volume_mm3"])
        require(math.isclose(float(r["volume_mm3"]), expected, rel_tol=1e-9), "Incorrect physical volume")
        require(0 <= expected <= float(i["image_volume_mm3"]), "Impossible volume")
        require(0 <= float(r["occupied_slice_fraction"]) <= 1, "Invalid occupied-slice fraction")
    for r in slices:
        require(r["split"] == inv[r["patient_id"]]["split"], "Split mismatch")
        require(0 <= float(r["normalized_z"]) <= 1, "Invalid z")
        require(math.isclose(float(r["relative_area"]), int(r["pixel_area"]) / (int(r["height"]) * int(r["width"]))), "Invalid pixel fraction")
        require((r["present"] == "True") == (int(r["pixel_area"]) > 0), "Presence/area mismatch")
    features = {(r["stem"], r["class_id"]): r for r in slices}
    for r in baseline:
        require(int(features[r["stem"], r["class_id"]]["pixel_area"]) == int(r["gt_area"]), "GT area join mismatch")
        g, p, inter = int(r["gt_area"]), int(r["pred_area"]), int(r["intersection"])
        require(0 <= inter <= min(g, p), "Impossible intersection")
        require(int(r["fp_pixels"]) == p - inter and int(r["fn_pixels"]) == g - inter, "Invalid FP/FN")
        if r["joint_empty"] == "True":
            require(g == p == 0 and r["dice"] == "", "Joint empty must be undefined")
        else:
            require(0 <= float(r["dice"]) <= 1 and math.isclose(float(r["dice"]), 2 * inter / (g + p)), "Invalid Dice")
    for r in patients:
        if r["reconstruction_available"] == "True":
            require(0 <= float(r["dice_3d"]) <= 1, "Invalid 3D Dice")
    for stage in ("dataset", "baseline"):
        for r in read_csv(tables / f"{stage}_inputs.csv"):
            stat = Path(r["path"]).stat()
            require(stat.st_size == int(r["size_bytes"]) and stat.st_mtime_ns == int(r["mtime_ns"]),
                    f"Input changed since analysis: {r['path']}")
    for split in ("train", "val", "all"):
        rs = [r for r in read_csv(tables / "class_frequency_original.csv") if r["split"] == split]
        if rs:
            require(math.isclose(sum(float(r["fraction_all_voxels"]) for r in rs), 1), "Frequency does not sum to one")
    report = {"status": "passed", "patients": len(ids), "patient_class_rows": len(organs), "shape_descriptor_rows": len(shapes),
              "orientation_verified": "LPS; axial slices increase superiorly",
              "slice_class_rows": len(slices), "baseline_slice_class_rows": len(baseline),
              "baseline_patient_class_rows": len(patients),
              "figures": len(list((output / "plots").glob("*.png"))),
              "example_sheets": len(list((output / "examples").glob("*.png")))}
    require({p.name for p in (output / "plots").iterdir() if p.is_file()} == set(FIGURES),
            "Missing curated figures or obsolete files remain")
    examples = read_csv(tables / 'shape_example_selection.csv')
    expected = select_shape_examples(shapes)
    require(len(examples) == 9, 'Expected nine primary shape examples')
    require({p.name for p in (output / 'examples').iterdir() if p.is_file()} ==
            {r['filename'] for r in expected}, 'Missing or unexpected example files')
    for actual, chosen in zip(examples, expected):
        require(all(actual[key] == str(value) for key, value in chosen.items()),
                'Example selection or displayed source descriptors differ from shape CSV')
        k, patient = int(actual['class_id']), actual['patient_id']
        require(actual['organ_color'] == COLORS[k], 'Inconsistent example organ color')
        root = original / patient
        ct, gt = nib.load(root / f'{patient}.nii.gz'), nib.load(root / 'GT.nii.gz')
        require(ct.shape == gt.shape and np.allclose(ct.affine, gt.affine), 'Misaligned CT and GT')
        require(np.allclose(gt.affine[:3, :3], np.diag(np.diag(gt.affine[:3, :3]))), 'Example orientation is not axial')
        mask = np.asanyarray(gt.dataobj) == k
        for axis, view, coordinate in ((0, 'sagittal', 'i'), (1, 'coronal', 'j'), (2, 'axial', 'k')):
            index = int(actual[f'{view}_index'])
            require(index == int(np.rint(float(chosen[f'centroid_{coordinate}']))), 'Incorrect centroid plane')
            require(np.take(mask, index, axis=axis).any(), 'Centroid plane misses organ')
    report['example_selection_and_alignment'] = 'passed'
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
