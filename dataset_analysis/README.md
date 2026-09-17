# Dataset Analysis

A guide to SegTHOR anatomy, the targets seen by the 2D ENet, and the saved
original baseline's performance. Seven quantitative figures are complemented by
nine original-CT/ground-truth examples explaining the physical shape descriptors.

## Data

- **20 patients:** 15 training and five validation.
- **Annotated organs:** esophagus (1), heart (2), trachea (3).
- **Aorta/class 4 is missing**; background is label 0.
- Original NIfTI GT supplies physical 3D descriptors; original CT supplies example backgrounds.
- Processed 256×256 masks supply slice size/location measurements.
- Saved original ENet validation predictions supply baseline metrics; no training is run.

## Main Figures

The seven PNGs in `results/plots/` use consistent colors: esophagus purple, heart
blue, trachea teal, and background gray. Each figure labels its source and cohort.

### `class_distribution.png`

Shows original-GT voxel imbalance, separating overall background/foreground from foreground composition.

**Main observation:** 98.96% is background; foreground is 73.35% heart, 23.07% esophagus, and 3.58% trachea.

### `shape_descriptors_3d.png`

Shows 60 patient–organ points by volume (mL), SI extent (mm), and normalized SI centroid.

**Main observation:** Heart occupies the larger-volume region, esophagus has longer SI spans, and trachea is smaller and relatively superior.

### `shape_descriptors_summary.png`

Shows coordinated distributions of the three descriptors, including every patient.

**Main observation:** Median volumes are 255.72/862.05/38.77 mL and SI extents 274.50/96.25/122.50 mm for esophagus/heart/trachea, with substantial patient variation.

### `target_area_through_scan.png`

Shows processed target-area percentage through normalized scan position, including absent-target slices, using patient-bin medians and IQRs.

**Main observation:** Heart cross-sections are larger and concentrated earlier; trachea appears farther superiorly, while esophagus spans more of the scan.

### `baseline_3d_dice_by_class.png`

Shows each validation patient's original-grid volumetric Dice and the equal-patient organ mean.

**Main observation:** Mean Dice is 0.695 for heart, 0.468 for esophagus, and 0.257 for trachea; Patient_19 has zero trachea Dice.

### `baseline_dice_vs_target_size.png`

Shows GT-positive slice Dice by processed target-area bin, averaging within patients before averaging patients equally.

**Main observation:** Heart has a strong positive area association, esophagus a weaker increase, and trachea no clean monotonic trend; contributor counts vary by bin.

### `baseline_dice_by_organ_position.png`

Shows patient-grouped GT-positive slice Dice at the beginning (20%), middle (60%), and end (20%) of each organ's annotated extent.

**Main observation:** Heart and esophagus perform better in the middle; trachea does not show the same decline at both ends.

## Shape Descriptors

- **Volume:** original GT voxel count × absolute spatial-affine determinant, converted from mm³ to mL.
- **Centroid:** mean foreground voxel indices, also stored in world coordinates through the NIfTI affine.
- **SI extent:** inferior-to-superior occupied-cell span in mm, including gaps and voxel thickness; not centreline length.
- **Normalized SI centroid:** world SI centroid scaled between inferior/superior scan voxel-centre limits (0–1); not anatomical registration.

## Qualitative Examples

`results/examples/` contains exactly nine GT figures, named
`<organ>_<small|typical|large>_volume.png`.

Selection uses `shape_descriptors_3d.csv`: minimum volume, closest to the class
median, and maximum volume. Ties resolve by patient ID (median distances compared
to 9 decimal places). These are descriptive examples, not clinical categories.

Each figure shows axial, coronal, and sagittal original CT views at the nearest
voxel plane to the organ centroid, with only that organ's GT overlaid in its
standard color. White crosses mark the projected centroid. The colored coronal
bracket marks the **whole 3D mask's SI span**, which can exceed its span in that plane.

Titles report volume (all occupied voxels), SI extent (full physical span), and
normalized SI centroid (relative scan location), directly from the descriptor CSV.
The selection and displayed values are recorded in `tables/shape_example_selection.csv`.

All examples use **−160 to 240 HU** (level 40, width 400), a consistent soft-tissue
window that also leaves the air-filled trachea distinct. Windowing only affects
display. Physical aspect ratios are preserved; transverse crops add 45 mm of
context and coronal/sagittal views retain full scan SI coverage.

CT/GT shapes and affines are checked before rendering. The verified axial L/P/S
orientation supports the direction labels: R/L = right/left, A/P = anterior/posterior,
and I/S = inferior/superior. These views use index-order orientation, as labeled.

## Important Tables

CSV files are in `results/tables/`:

- `shape_descriptors_3d.csv`: 60 patient–organ rows with physical shape descriptors.
- `patient_inventory.csv`: one row per patient with split, scan geometry, and label counts.
- `patient_class_stats.csv`: patient–organ counts, physical measurements, and slice coverage.
- `slice_class_stats.csv`: processed slice–organ target sizes and relative positions.
- `baseline_patient_metrics.csv`: validation patient–organ 3D Dice and slice summaries.
- `baseline_slice_metrics.csv`: validation slice–organ Dice, overlap, and empty-case indicators.

## Run and Validation

Submit the existing CPU-only workflow from the repository root:

```bash
sbatch jobsAndOutputs/baseline/jobs/dataset_analysis.job
```

Logs go to `jobsAndOutputs/baseline/outputs/dataset_analysis_<jobid>.out`.
The job runs numerical tests, both analyses, example generation, and the results
audit. `results/validation.json` reports coverage, geometry/metric checks, and
example selection/alignment checks. Input manifests and run JSONs retain provenance.

## Caveats

- Only five validation patients; adjacent slices are correlated.
- Class 4/aorta is intentionally omitted; background includes unannotated anatomy.
- Normalized SI position depends on scan coverage, not anatomical registration.
- SI extent is not centreline length; a centroid plane need not contain every part of an organ.
- IQR bands show descriptive spread, not confidence intervals; size bins can contain different patients.
- Exploratory associations do not establish causality or clinical normal ranges.
- Joint-empty Dice is undefined; slice-quality trends use GT-positive slices.
