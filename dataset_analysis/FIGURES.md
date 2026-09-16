# SegTHOR dataset-profile figures

20 patients, `data/segthor_part1/train/Patient_*/GT.nii.gz`. Label values are 0
(background), 1, 2, 3; label 4 is part of the annotation scheme but has 0
voxels in every one of the 20 patients, so it never appears below. Labels are
referred to only by number.

Regenerate: run `python tools/dataset_profile.py --data-dir
data/segthor_part1/train --out-dir figures/profile` once to write the tables
(`patients.csv`, `labels.csv`, `label_slices.csv`, `scan_slices.csv`,
`label_pairs.csv`, `intensity_histograms.npz`), then run each figure script
below with the profile directory it reads from.

Shared conventions: label 1 is teal (`#2A9D8F`), label 2 orange (`#E76F51`),
label 3 amber (`#E9A23B`); background/other surfaces are grey where drawn.
Spatial axes are left–right (x), anterior–posterior (y), superior–inferior
(z), in mm. Every image and label volume in the dataset is LPS-oriented (x
increases toward patient left, y toward posterior, z toward superior).

## Checks without a figure

Verified against `patients.csv` / `labels.csv`:

- Image/label grid integrity passes for all 20 patients: image and label
  arrays have matching shape and voxel spacing, and identical affines (max
  affine element difference across all patients is 0.0); both image and
  label axis codes are LPS in every patient.
- Labels present are exactly {0, 1, 2, 3} in every one of the 20 patients;
  label 4 is never found (0 voxels in every patient).
- No empty slices inside a label's slice range: for labels 1, 2 and 3 in
  every patient, every slice between that label's first and last slice
  contains at least one voxel of it (0 empty slices, all patients).
- Crop to nonzero (the box around every voxel above the scan's minimum
  value) does **not** keep 100% for every patient: it keeps 97.5-100% of
  the image's voxels per patient (mean 99.2%), reaching exactly 100% for
  only 3 of the 20.

## 01-03 Scan geometry

**File:** `figures/profile/01-03_scan_geometry/scan_geometry.png`
**Script:** `dataset_analysis/profile_figures/f01_03_scan_geometry.py --profile-dir figures/profile`
**Shows:** Six dot-histogram panels — pixel size, slice spacing, spacing
ratio, number of slices, image width and scan length — with a red triangle
marking the median.
**Unit and pooling:** One dot per patient per panel (20 dots), stacked at
bins centred on multiples of a fixed width per panel.
**Useful for:** Choosing a common resample target spacing and expected
patch footprint.

**File:** `figures/profile/01-03_scan_geometry/grid_per_patient.png`
**Script:** same command as above (both figures are written by `main()`).
**Shows:** One row per patient, sorted by pixel size then slice spacing, one
column per geometry measurement (x/y/z spacing, x/y/z voxel counts, x/y/z
extent in mm, total voxels); the bottom row is the median of all patients.
**Unit and pooling:** One dot per patient per column; bottom row is the
per-column median across all 20 patients.
**Useful for:** Spotting which patients are geometric outliers before fixing
a resampling target.

## 04 Scan intensity

**File:** `figures/profile/04_scan_intensity/slices_3d.png`
**Script:** `dataset_analysis/profile_figures/f04_scan_intensity/slices_3d.py --patient Patient_01`
**Shows:** Eight evenly spaced axial slices of one patient, stacked in 3D at
their true height, coloured by HU on a fixed scale and nearly transparent
near -1000 HU.
**Unit and pooling:** Single patient (Patient_01), full-resolution pixels;
not aggregated across patients.
**Useful for:** A visual sanity check of orientation, spacing and intensity
range before trusting the pooled statistics elsewhere.

**File:** `figures/profile/04_scan_intensity/bands_along_height.png`
**Script:** `dataset_analysis/profile_figures/f04_scan_intensity/bands_along_height.py --data-dir data/segthor_part1/train`
**Shows:** For each of the 20 patients, one panel of the median, middle 50%
and middle 90% of slice HU values plotted against height above the lowest
slice, with nnU-Net's clip values as dashed lines.
**Unit and pooling:** One panel per patient; each point along the x-axis
summarises every pixel of that slice (full resolution, no subsampling).
**Useful for:** Seeing how intensity varies along the scanned length, e.g.
whether a height-dependent normalisation would matter.

**File:** `figures/profile/04_scan_intensity/circle_grid.png`
**Script:** `dataset_analysis/profile_figures/f04_scan_intensity/circle_grid.py --profile-dir figures/profile`
**Shows:** Rows are the 20 scans, columns are fixed HU bands from -1000 HU
up; circle size and colour (log scale) give the percentage of that scan's
voxels in the band.
**Unit and pooling:** One circle per patient x HU band, from the exact 1 HU
histogram of the whole scan (every voxel).
**Useful for:** Comparing the overall intensity-band distribution, including
the size of the -1000 padding value, across all patients at once.

**File:** `figures/profile/04_scan_intensity/centre_edge_outside.png`
**Script:** `dataset_analysis/profile_figures/f04_scan_intensity/centre_edge_outside.py --profile-dir figures/profile`
**Shows:** Left, one patient's scan as eight stacked slices coloured by
region (outside the scanned area, and its outer/middle/central third by
distance from the area's edge). Right, one panel per region with every
patient's HU histogram as a translucent step fill and the pooled 20-patient
histogram as a dark outline.
**Unit and pooling:** Region membership from a per-slice distance transform
of the scanned area; each patient's histogram covers every pixel of that
region across all its slices (full resolution); the pooled line sums raw
counts across patients, not an average of percentages.
**Useful for:** Checking whether intensity near the edge of the scanned
field differs from its centre, relevant to field-of-view cropping.

**File:** `figures/profile/04_scan_intensity/nnunet_normalisation.png`
**Script:** `dataset_analysis/profile_figures/f04_scan_intensity/nnunet_normalisation.py --profile-dir figures/profile`
**Shows:** Top, one patient's slice stack before and after nnU-Net's CT
normalisation (clip to its foreground-sample percentiles, then subtract mean
and divide by standard deviation). Bottom, whole-scan HU histograms for all
20 patients before and after that normalisation.
**Unit and pooling:** Top row is one patient at full resolution; bottom row
overlays one whole-scan histogram per patient (20 lines), from the exact 1
HU histograms; the after-normalisation panel's end bins are exactly the mass
the clip removes.
**Useful for:** Confirming what nnU-Net's normalisation actually does to the
visible range and how much mass sits at the clip boundaries.

**File:** `figures/profile/04_scan_intensity/scan_vs_labels.png`
**Script:** `dataset_analysis/profile_figures/f04_scan_intensity/scan_vs_labels.py --profile-dir figures/profile`
**Shows:** One small panel per patient: the whole-scan HU distribution drawn
above zero and the distribution restricted to labels 1-3 mirrored below it,
with a grey band for the across-patient average whole-scan distribution and
dashed nnU-Net clip lines.
**Unit and pooling:** Per patient, voxels rebinned into 20 fixed-width HU
ranges from -1000 to 1600 HU, from the exact 1 HU histograms (log scale on
percent of voxels).
**Useful for:** Seeing how much of the labelled-tissue intensity range
overlaps the whole-scan range, informing normalisation/clip choices.

**File:** `figures/profile/04_scan_intensity/spike_and_ridges.png`
**Script:** `dataset_analysis/profile_figures/f04_scan_intensity/spike_and_ridges.py --profile-dir figures/profile`
**Shows:** Left, a lollipop per patient of the percentage of that scan's
voxels exactly at -1000 HU (the padding value outside the scanned field).
Right, an overlapping ridge plot, one row per patient, of the remaining
voxels' HU distribution in 10 HU bands, height capped for readability.
**Unit and pooling:** One point per patient (left); one ridge row per
patient (right), each built only from that patient's own non-(-1000)
voxels, from the exact 1 HU "scan" histograms.
**Useful for:** Quantifying how much of each scan is the fixed padding value
versus signal, relevant to foreground/background sampling.

## 05 Label intensity

**File:** `figures/profile/05_label_intensity/label_intensity.png`
**Script:** `dataset_analysis/profile_figures/f05_label_intensity.py --profile-dir figures/profile`
**Shows:** Ridge rows — one per patient plus one pooled "all patients" row —
of HU histograms for labels 1, 2 and 3, with nnU-Net's foreground sample
overlaid as a dashed line and its clip values as vertical dashed lines.
**Unit and pooling:** One row per patient (plus one pooled row); each row is
built from the exact 1 HU histograms of that patient's label 1/2/3 voxels,
rebinned to 10 HU; the dashed line is that patient's own 5M-voxel nnU-Net
foreground sample; the clip lines come from the pooled sample's 0.5th/99.5th
percentile.
**Useful for:** Choosing per-class clip/normalisation parameters and judging
how consistent each label's intensity is across patients.

## 06-07 Label size and intensity

**File:** `figures/profile/06-07_label_size_intensity/label_size_intensity.png`
**Script:** `dataset_analysis/profile_figures/f06_07_label_size_intensity.py --profile-dir figures/profile --data-dir data/segthor_part1/train`
**Shows:** Top, every patient's label 1, 2 and 3 surface at the same
physical scale, sorted left to right from smallest to largest volume
(shading light to dark). Bottom, overlapping HU histograms for background
and labels 1-3, pooled across patients.
**Unit and pooling:** One 3D surface mesh per patient per label (marching
cubes on a resampled mask); the histogram panel pools every patient's voxels
for each of background/label 1/2/3 into one histogram per group.
**Useful for:** Relating a label's physical size/shape variability to its
intensity distribution in one view, e.g. whether small instances also read
differently in HU.

## 07 Label size

**File:** `figures/profile/07_label_size/label_shapes_3d.png`
**Script:** `dataset_analysis/profile_figures/f07_label_shapes_and_sizes.py --profile-dir figures/profile --data-dir data/segthor_part1/train`
**Shows:** One 3D panel per patient with labels 1, 2 and 3 rendered at
identical physical scale and camera angle.
**Unit and pooling:** One mesh set per patient (surfaces from marching cubes
on the raw mask, no resampling); 20 panels, not aggregated.
**Useful for:** Visually comparing the shape and relative position of the
three labels across all patients at a glance.

**File:** `figures/profile/07_label_size/label_size_per_patient.png`
**Script:** same command as above (`main()` writes both figures).
**Shows:** One row per patient, three columns (voxel count, volume in mL, %
of the scan's voxels), each with three dots for labels 1, 2 and 3; the
bottom row is the median across patients.
**Unit and pooling:** One dot per patient x label; volume in mL is voxel
count times voxel volume, from `labels.csv`.
**Useful for:** Setting expected per-class foreground volume, e.g. for
sampling or class-balance choices.

## 08-11 Slices and change

**File:** `figures/profile/08-11_slices_and_change/slices_and_change_bands.png`
**Script:** `dataset_analysis/profile_figures/f08_11_slices_and_change.py --profile-dir figures/profile`
**Shows:** Top left, a stacked histogram of how many axial slices contain
each label, one count per patient. Top right, the median with middle-50%/
middle-90% bands of each label's area (normalised to its own peak slice)
against position within the label's slice range (0% lowest to 100% highest
slice), the three labels overlapped. Bottom, the same layout for
slice-to-slice area change.
**Unit and pooling:** Top-left bars are one count per patient (20 per
label). The bands are computed per label across all 20 patients at each of
100 relative positions; rescaling position to 0-100% lets patients with
different slice counts line up. Area change = 100 x (area - area of the
slice below) / area of the slice below.
**Useful for:** Reading typical along-length shape (e.g. tapering near the
ends) and how abruptly cross-sectional area changes between neighbouring
slices — relevant to slice thickness and 2D-vs-3D modelling choices.

**File:** `figures/profile/08-11_slices_and_change/slices_and_change_joint.png`
**Script:** same command as above.
**Shows:** The same slice-count histogram on top, then one 2D histogram per
label of every neighbouring-slice pair: position within the label's range
(x) against slice-to-slice area change (y), with a marginal histogram of the
change alongside.
**Unit and pooling:** Per-slice-pair counts pooled across all 20 patients —
every neighbouring slice pair within a label's range contributes one point;
darker cells hold more pairs.
**Useful for:** Seeing where along a label's length area changes most, using
every slice pair rather than a per-patient summary.

## 09 Label bounding box

**File:** `figures/profile/09_label_bounding_box/label_bbox_size_3d.png`
**Script:** `dataset_analysis/profile_figures/f09_label_bounding_box.py --profile-dir figures/profile`
**Shows:** One panel per label; every patient's 3D bounding-box wireframe
drawn around a shared centre, so only size and shape (not position) differ,
with the median box drawn bold; axes are left–right / anterior–posterior /
superior–inferior in mm.
**Unit and pooling:** One wireframe box per patient per label, 20 overlaid
per panel. Box size along an axis = (last - first voxel index + 1) x voxel
spacing.
**Useful for:** Reading off typical and extreme physical extents per label
and axis, e.g. the minimum patch size needed to contain a label.

**File:** `figures/profile/09_label_bounding_box/label_bbox_position_3d.png`
**Script:** same command as above (`main()` writes both figures).
**Shows:** All three labels' bounding boxes in one shared 3D space, each
patient's boxes shifted so that patient's label 2 centre sits at the origin.
**Unit and pooling:** One wireframe box per patient per label (20 patients x
3 labels), all referenced to that patient's own label 2 centre.
**Useful for:** Seeing typical relative position and extent overlap between
the three labels, e.g. for anchoring a crop around one of them.

## 12 Connected components

**File:** `figures/profile/12_connected_components/connected_components.png`
**Script:** `dataset_analysis/profile_figures/f12_connected_components.py --data-dir data/segthor_part1/train --profile-dir figures/profile`
**Shows:** Three heatmaps, one per label, of the share of slices (2D rules)
or patients (3D rules) that have 1, 2 or 3+ connected pieces, under five
connectivity rules: 2D 4- and 8-connectivity within an axial slice, and 3D
6-, 18- and 26-connectivity over the whole label.
**Unit and pooling:** 2D rows pool every axial slice containing the label
across all 20 patients (one piece-count per slice); 3D rows pool one
piece-count per patient (whole 3D mask). All 20 patients are pooled into
each heatmap.
**Useful for:** Choosing post-processing connectivity (e.g. keep-largest-
component) and checking whether 2D or 3D processing changes how fragmented
a label looks. Key number: 3 of 20 patients have label 1 split into 2 pieces
under 3D 6-connectivity (face-adjacency only); all 20 are a single piece
once diagonal adjacency is allowed (18- or 26-connectivity). Labels 2 and 3
are a single 3D piece in all 20 patients under every 3D rule; both still
show slices with 2 (and, for label 3, occasionally 3) pieces under the 2D
rules.

## 13 Label pairs

**File:** `figures/profile/13_label_pairs/label_pairs_contact_3d.png`
**Script:** `dataset_analysis/profile_figures/f13_label_pairs.py --profile-dir figures/profile --data-dir data/segthor_part1/train`
**Shows:** Every patient's three labels as faint grey surfaces, with the
part of each label's surface that lies within one voxel of another label
painted in that pair's colour; a strip under each row of 10 patients gives
each pair's shared border area. Patients are sorted by total shared border
area.
**Unit and pooling:** One mesh set per patient (surfaces from marching cubes
on a resampled mask); shared border area per patient per label pair comes
from `label_pairs.csv` (a distance/face-sharing computation), one value per
patient x pair.
**Useful for:** Judging which label pairs are typically adjacent and by how
much, relevant to boundary-aware loss weighting or post-processing between
neighbouring classes.
