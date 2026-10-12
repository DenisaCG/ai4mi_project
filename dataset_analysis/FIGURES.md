# SegTHOR dataset-profile figures

Generated below against the 20-patient corrected release
`data/segthor_part1_corrected/train`, where label values are 0 (background), 1
(esophagus), 2 (heart), 3 (trachea), 4 (aorta) and every label has voxels in
all 20 patients. In the original release `data/segthor_part1/train` the aorta
is not annotated separately (label 4 has 0 voxels in every patient and label 1
holds the esophagus and aorta together); it enters only the before/after
figures in `figures/comparison`. Every figure script runs unchanged against
either release (see `tools/run_all_figures.py`) -- labels are referred to only
by number, and each script detects which ones actually have voxels rather than
assuming either case.

Regenerate: run `python tools/dataset_profile.py --data-dir
data/segthor_part1_corrected/train --out-dir figures/profile` once to write the tables
(`patients.csv`, `labels.csv`, `label_slices.csv`, `scan_slices.csv`,
`label_pairs.csv`, `intensity_histograms.npz`), then run each figure script
below with the profile directory it reads from -- or run all of them in one
command with `tools/run_all_figures.py` (see its docstring for the before/after flags).

Shared conventions: label colors come from `tools/plot_style.py:EARTH_LABEL_COLORS`,
the palette every figure script here uses -- label 1 teal (`#2E5E6E`), label 2 brick
(`#B5533C`), label 3 ochre (`#C9A15B`), label 4 olive (`#5E9142`); the unlabeled
background is warm sand (`#CFC7B8`). Spatial axes are left–right (x),
anterior–posterior (y), superior–inferior (z), in mm.
Every image and label volume in the dataset is LPS-oriented (x increases
toward patient left, y toward posterior, z toward superior).

## Checks without a figure

Verified against `patients.csv` / `labels.csv`:

- Image/label grid integrity passes for all 20 patients: image and label
  arrays have matching shape and voxel spacing, and identical affines (max
  affine element difference across all patients is 0.0); both image and
  label axis codes are LPS in every patient.
- Labels present are exactly {0, 1, 2, 3, 4} in every one of the 20 patients
  of the corrected release (the original release has {0, 1, 2, 3}).
- No empty slices inside a label's slice range: for labels 1, 2, 3 and 4 in
  every patient, every slice between that label's first and last slice
  contains at least one voxel of it (0 empty slices, all patients).
- Crop to nonzero (the box around every voxel above the scan's minimum
  value) keeps 100% of label voxels in every patient: no voxel of labels
  1-3 lies outside the box. The box itself covers 97.5-100% of the image
  (mean 99.2%), so cropping removes at most 2.5% of voxels.

## Scan geometry

**File:** `figures/profile/scan_geometry/scan_geometry.png`
**Script:** `dataset_analysis/profile_figures/scan_geometry.py --profile-dir figures/profile`
**Shows:** Six dot-histogram panels — pixel size, slice spacing, spacing
ratio, number of slices, image width and scan length — with a red triangle
marking the median.
**Unit and pooling:** One dot per patient per panel (20 dots), stacked at
bins centered on multiples of a fixed width per panel.
**Useful for:** Choosing a common resample target spacing and expected
patch footprint.

**File:** `figures/profile/scan_geometry/grid_per_patient.png`
**Script:** same command as above (both figures are written by `main()`).
**Shows:** One row per patient, sorted by pixel size then slice spacing, one
column per geometry measurement (x/y/z spacing, x/y/z voxel counts, x/y/z
extent in mm, total voxels); the bottom row is the median of all patients.
**Unit and pooling:** One dot per patient per column; bottom row is the
per-column median across all 20 patients.
**Useful for:** Spotting which patients are geometric outliers before fixing
a resampling target.

**File:** `figures/profile/scan_geometry/field_of_view.png`
**Script:** same command as above (all three figures are written by `main()`).
**Shows:** Left, every distinct axial field-of-view size drawn to true scale
as nested concentric squares (outline thickness = how many scans share that
size), with the smallest one filled by a real axial slice from that patient.
Right, histograms of slice count, slice spacing and scan length, colored to
match each field-of-view size.
**Unit and pooling:** One outline per distinct field-of-view size (mm,
rounded), grouping the 20 patients into however many sizes actually occur;
one histogram bar per patient for slice count/spacing/length.
**Useful for:** Seeing the actual physical size differences between scans at
a glance, not just as numbers in a table.

## Label HU distribution

**File:** `figures/profile/label_hu_distribution/label_hu_distribution.png`
**Script:** `dataset_analysis/profile_figures/label_hu_distribution.py --profile-dir figures/profile`
**Shows:** The HU distribution of every label with voxels, all labels together as opaque
outlined bars (one color per organ, named in the legend; the heart lighter, the others strong) in
thousand voxels per 10 HU bin, summed over all patients. The bars are not stacked: each starts at zero
and the smaller bar of a bin is drawn in front, so a taller bar shows only the part above
it. The HU axis runs between the two datasets: the first is drawn upwards and the second
hangs downwards from it. The stretch between air and soft tissue (HU -900 to -200, marked
with a purple band and break marks) is drawn 6 times narrower. The background is not shown. Black lines mark nnU-Net's statistics of the labeled voxels of the last dataset only, every
patient weighted equally as in its fingerprint: the 0.5th and 99.5th percentile, the mean and the median.
**Unit and pooling:** From the exact 1 HU histograms of every voxel inside the
group, summed across all patients.
**Compare two datasets:** pass two folders; the first is drawn upwards and the
second mirrored downwards from the same zero line, sharing one x axis and one
color per label, so a label only the second has (the aorta) appears only in
the lower half:
`--profile-dir figures/segthor_part1/profile figures/full_release/profile
--names "3 labels" "4 labels" --out-dir figures/comparison`.
**Useful for:** Seeing which labels make up the voxels at each HU value, and
before/after adding a label.

## Organ position in 3D

**File:** `figures/profile/organ_position_3d/organ_position_3d.png`
**Script:** `dataset_analysis/profile_figures/organ_position_3d.py --profile-dir figures/profile`
**Shows:** One 3D panel per organ (esophagus, aorta, heart, trachea) and a last panel with all organs, all on the
same axes and view (left–right, anterior–posterior, superior–inferior in mm; a grid line every 50 mm),
with positions relative to each patient's heart center. In an organ panel every patient has a thin
box (the bounding box of the label) and a dot at the organ's center; the bold box is the median box.
The last panel has the median box of every organ, lightly filled, and the median center of each organ.
**Unit and pooling:** One box and one center per patient and organ (20 patients), from `labels.csv`;
median boxes take the median start and size on each axis.
**Useful for:** Reading where each organ typically sits and how far it reaches along the body axis,
and how the organs overlap, without the label numbers or the clutter of every box in one space.

## Connected components

**File:** `figures/profile/connected_components/connected_components.png`; before/after the label fix:
`figures/comparison/connected_components.png`
**Script:** `dataset_analysis/profile_figures/connected_components.py --data-dir data/segthor_part1_corrected/train --profile-dir figures/profile`;
before/after: add `--before-data-dir data/segthor_part1/train --names "3 labels (aorta merged)" "4 labels (aorta separate)" --out-dir figures/comparison`
**Shows:** One heatmap per organ (esophagus, heart, trachea, aorta where annotated) of the
share of slices (2D rules) or patients (3D rules) that have 1, 2 or 3+ connected components,
under five connectivity rules: 2D 4- and 8-connectivity within an axial slice, and 3D 6-,
18- and 26-connectivity over the whole label. With `--before-data-dir` there are two
rows, that dataset on top and `--data-dir` below, so the effect of the corrected labels
reads down a column; an organ whose shares are the same in both datasets (heart and
trachea) is drawn in the top row only, and an arrow runs from the esophagus heatmap of the top row to the
aorta heatmap of the second row, marking that the aorta was part of the esophagus label.
**Unit and pooling:** 2D rows pool every axial slice containing the label across all
patients (one component count per slice); 3D rows pool one component count per patient (whole 3D
mask).
**Useful for:** Choosing post-processing connectivity (e.g. keep-largest-component) and
checking whether 2D or 3D processing changes how fragmented a label looks.

## Label correction before and after

**File:** `figures/comparison/label_makeup_rings.png`
**Script:** `dataset_analysis/profile_figures/label_makeup_rings.py --profile-dir figures/before/profile figures/profile --names "3 labels (aorta merged)" "4 labels (aorta separate)" --out-dir figures/comparison`
**Shows:** A big ring of all scan voxels of all patients, split into background and labeled voxels.
A gray panel zooms into the thin labeled sliver and holds one small ring per release, split by
organ, with each organ's share of the labeled voxels written on its wedge and the pooled labeled
voxel count in the middle. With 3 labels the teal wedge is the merged esophagus and aorta label.
**Unit and pooling:** Voxels of all patients pooled, from `labels.csv` and `patients.csv` of each
release; percentages of the small rings are shares of the labeled voxels, the big ring is a share
of all voxels.
**Useful for:** Reading the label make-up, background included, before and after the correction.

**File:** `figures/comparison/label_correction_example.png`, and each half alone as `label_correction_example_before.png` and `label_correction_example_after.png`
**Script:** `dataset_analysis/profile_figures/label_correction_example.py --patient Patient_02 --out-dir figures/comparison`
**Shows:** One patient's axial CT slice and 3D surfaces under both label sets. Before, the
esophagus label is two face-connected pieces (the larger one is the aorta); after, the esophagus and
the aorta are one piece each. The slice is where the smaller of the esophagus and aorta areas is
largest; patient and slice are written in the corner of each slice, and a box with a
line to each blob names it.
**Unit and pooling:** One patient, one slice, and the meshes of all its slices (marching cubes on the
labels, lightly smoothed).
**Useful for:** Showing what the correction changed in one case.

**File:** `figures/comparison/label_example_2D.png` and `figures/comparison/label_example_3D.png`
**Script:** `dataset_analysis/profile_figures/label_example.py --patient Patient_18 --out-dir figures/comparison`
**Shows:** One patient's four organs, once on an axial CT slice (the one where the smallest organ is
largest) and once as 3D surfaces seen from the patient's side; patient and slice are written under
each figure.
**Unit and pooling:** One patient, corrected labels only.
**Useful for:** Showing the four annotated organs on a real scan.
