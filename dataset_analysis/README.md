# Dataset Analysis

<<<<<<< HEAD
A guide to SegTHOR anatomy, the targets seen by the 2D ENet, and how the baseline
performs on the **original** labels versus the **corrected** labels (aorta annotated
separately). Every figure below is shown for both label versions, side by side.
=======
This analysis covers annotated classes 1 (esophagus), 2 (heart), 3 (trachea),
and 4 (aorta). In the original course release the aorta has no label of
its own (label 4 is empty and label 1 holds the esophagus and aorta together);
the corrected release splits them. Every script here reports which classes are
actually present in the data it's given rather than assuming either case, and
an absent class is left out of quality summaries rather than counted as zero.
Background means the supplied label 0, not necessarily anatomically empty
space.

For the full figure set (dataset profile, nnU-Net planner checks, EDA) in one
command, see `tools/run_all_figures.py` (documented in `tools/README.md`),
which also runs this directory's `analyze_dataset.py`/`analyze_baseline.py`
when given `--processed-dir`/`--predictions`.
>>>>>>> 35c00e72cc8eb216ef8f05d638d790bad9933a18

## Data

- **20 patients:** 15 training and five validation.
- **Original labels** (`data/segthor_part1/`, `data/SEGTHOR/`): esophagus (1), heart (2), trachea (3). Aorta (4) has zero voxels; the aorta is merged into label 1, which is why the original esophagus looks about 5x too large.
- **Corrected labels** (`data/segthor_part1_corrected/`, `data/SEGTHOR_corrected/`): esophagus (1), heart (2), trachea (3), aorta (4).
- Background is label 0 in both. Original NIfTI GT supplies physical 3D descriptors and the CT backgrounds of the examples; processed 256×256 masks supply slice size/location measurements.
- The Dice and slice-level baseline figures use one seeded pipeline run per label version: the run with the median foreground Dice of its 9 runs (rule fixed before looking at the plots). That is `runs/segthor_enet_ce_repro/seed2` for the original labels and `runs/segthor_enet_ce_repro_corrected_fixed/seed7` for the corrected labels, which was trained after the stale-PNG repair (see Caveats).
- The before/after Dice figure uses every seeded pipeline run: original labels `runs/segthor_enet_ce_repro/seed0-8` (9 runs) vs corrected labels `runs/segthor_enet_ce_repro_corrected_fixed/seed0-8` (9 runs). No training happens here.

Results live in three folders under `results/`:

| Folder | Contents |
|---|---|
| `repro_original_seed0/` | Analysis of the original labels against pipeline seed 2, the median run ("before") |
| `repro_corrected_seed0/` | Analysis of the corrected labels against pipeline seed 7, the median run, trained on the repaired data ("after"). Folder names keep `seed0` for history |
| `before_after/` | Direct before/after comparison figures and tables |

Colors are fixed per organ in every figure and come from the shared theme in `tools/plot_style.py` (earth palette): esophagus teal, heart brick, trachea ochre, aorta olive, background sand. The baseline figures are titled "Original ENet baseline" in both folders; that wording predates the correction and refers to the ENet recipe, not the label version.

## Before vs After

<table>
<tr>
<td width="50%"><img src="results/before_after/plots/before_after_class_balance.png" width="100%"></td>
<td width="50%"><img src="results/before_after/plots/before_after_dice_3d.png" width="100%"></td>
</tr>
<tr>
<td valign="top"><b>Ground truth.</b> Esophagus falls from 23.1% to 4.1% of annotated voxels (2.36 M to 0.42 M) and aorta appears at 18.9% (1.93 M). Heart (7.50 M) and trachea (0.37 M) are unchanged. Pooled, the old esophagus ground truth was 5.7x larger than the corrected one. <code>before_after/tables/label_changes.csv</code> confirms the correction is almost purely a split of old label 1: of 1,942,033 changed voxels, 15,629 (0.8%) are not a clean esophagus/aorta split, and only 3 voxels changed outside old label 1.</td>
<td valign="top"><b>3D Dice per training run, before → after (mean over 9 runs each).</b> Each dot is one run's mean over the 5 validation patients. Esophagus 0.53 → 0.18, heart 0.77 → 0.78, trachea 0.52 → 0.29, aorta – → 0.49. Heart is unchanged. On the original labels no run fails: 0 of 9 at Dice 0 for both esophagus (range 0.44–0.57) and trachea (0.41–0.59). On the corrected labels 3 of 9 runs never learn the esophagus (seeds 0, 3, 8) and 2 of 9 never learn the trachea (seeds 3 and 8, which also fail on the esophagus), although the trachea ground truth is identical in both versions. With 9 runs per side this is suggestive, not established (Fisher exact p = 0.21 and 0.47). Esophagus values are not like-for-like: the old target also contained the aorta. The failures do not come from the repaired stale labels (same outcome in 9 of 9 paired seeds).</td>
</tr>
<tr>
<td colspan="2"><img src="results/before_after/plots/before_after_example_slice.png" width="60%"></td>
</tr>
<tr>
<td colspan="2"><b>One example slice (Patient_01, axial slice 145, largest aorta cross-section).</b> Left: in the original labels the whole aorta carries label 1 (drawn in the esophagus color). Middle: the corrected labels. Right: exactly the voxels whose label changed, coloured by their new label; here all of them become aorta.</td>
</tr>
<tr>
<td colspan="2"><img src="results/before_after/plots/before_after_metrics_by_experiment_solid.png" width="100%"></td>
</tr>
<tr>
<td colspan="2"><b>Dice, HD95 and ASSD for every experiment, absolute values (mean over the 5 validation patients and the experiment's seeds).</b> One row per experiment, one dot per organ and a black line for the overall (the mean over organs, its value printed on top); the grey line spans the organs. Rows: the uncorrected labels (9 seeds, no aorta), the corrected-label baseline (9 seeds; Dice 0.18 / 0.78 / 0.29 / 0.49, HD95 41 / 35 / 72 / 33 mm, ASSD 10.5 / 8.6 / 20.0 / 8.1 mm for esophagus / heart / trachea / aorta), the loss variants (row names coloured by whether the loss covers the foreground only or foreground + background), preprocessing, 512x512 (native slices instead of 256x256, same cross-entropy over all classes) and nnU-Net 2D and 3D. The baseline and 512 row names state their loss. A row without finished runs is drawn faded with a dotted line at invented placeholder values and switches to real values once its runs have <code>eval/metrics_3d.csv</code>. <code>before_after_metrics_by_experiment_shaded.png</code> is the same figure with the organ dots and lines shaded so the overall line stands out. Esophagus is not like-for-like across label versions. The right-hand column counts collapsed organ-seeds (an organ with 3D Dice below 0.1 in a seed) out of organs x seeds. All values are in <code>before_after/tables/metrics_3d_by_experiment.csv</code>.</td>
</tr>
</table>

## Main Figures

Each pair shows the original labels (left) and corrected labels (right), and the text
underneath compares them. The Dice and slice-level baseline figures come from one seeded pipeline run per side (the median run of 9: original labels seed 2, corrected labels seed 7). Runs differ a lot, so read them as one example; the spread across all 9 runs is in Before vs After.

<table>
<tr><th width="50%">Before: original labels</th><th width="50%">After: corrected labels</th></tr>

<tr>
<td><img src="results/repro_original_seed0/plots/class_distribution.png" width="100%"></td>
<td><img src="results/repro_corrected_seed0/plots/class_distribution.png" width="100%"></td>
</tr>
<tr><td colspan="2"><b><code>class_distribution.png</code></b> shows original-GT voxel imbalance, separating overall background/foreground from foreground composition. Background is 98.96% in both. Foreground is 73.35% heart, 23.07% esophagus, 3.58% trachea before, and 73.46% heart, 18.88% aorta, 4.08% esophagus, 3.58% trachea after. Heart and trachea have identical voxel counts (7,496,021 and 365,773).</td></tr>

<tr>
<td><img src="results/repro_original_seed0/plots/shape_descriptors_3d.png" width="100%"></td>
<td><img src="results/repro_corrected_seed0/plots/shape_descriptors_3d.png" width="100%"></td>
</tr>
<tr><td colspan="2"><b><code>shape_descriptors_3d.png</code></b> shows every patient–organ point with annotated voxels by volume (mL), SI extent (mm) and normalized SI centroid. Heart is the large-volume, inferior structure (median 862 mL, centroid 0.34) and trachea the small, superior one (39 mL, 0.65). After the correction the esophagus shrinks to a median 46 mL and sits in the same mid-scan region as the new aorta (median 202 mL, centroid 0.47 vs 0.49).</td></tr>

<tr>
<td><img src="results/repro_original_seed0/plots/shape_descriptors_summary.png" width="100%"></td>
<td><img src="results/repro_corrected_seed0/plots/shape_descriptors_summary.png" width="100%"></td>
</tr>
<tr><td colspan="2"><b><code>shape_descriptors_summary.png</code></b> shows the same three descriptors as coordinated distributions, including every patient. Median volumes esophagus/heart/trachea are 255.72/862.05/38.77 mL before and 45.79/862.05/38.77 mL after (aorta 202.41 mL); median SI extents are 274.50/96.25/122.50 mm before and 236.25/96.25/122.50 mm after (aorta 216.25 mm). Patient variation is substantial.</td></tr>

<tr>
<td><img src="results/repro_original_seed0/plots/target_area_through_scan.png" width="100%"></td>
<td><img src="results/repro_corrected_seed0/plots/target_area_through_scan.png" width="100%"></td>
</tr>
<tr><td colspan="2"><b><code>target_area_through_scan.png</code></b> shows processed target area (% of slice) through normalized scan position, including absent-target slices, as patient-bin medians with IQR bands. After the correction: heart peaks at 2.8% around position 0.35, trachea at 0.12% around 0.65, esophagus stays flat near 0.065% between 0.25 and 0.75, and aorta rises to 0.42% at 0.65 and is absent above 0.65. The original esophagus is far larger (peak 0.48% at 0.65) because label 1 included the aorta.</td></tr>

<tr>
<td><img src="results/repro_original_seed0/plots/baseline_3d_dice_by_class.png" width="100%"></td>
<td><img src="results/repro_corrected_seed0/plots/baseline_3d_dice_by_class.png" width="100%"></td>
</tr>
<tr><td colspan="2"><b><code>baseline_3d_dice_by_class.png</code></b> shows each validation patient's original-grid volumetric Dice and the equal-patient organ mean (seed 0). Means are esophagus 0.562, heart 0.797, trachea 0.483 before, and esophagus 0.221, heart 0.819, trachea 0.360, aorta 0.466 after. Patient_19's trachea has Dice 0.000 in both. These are single runs; the spread across all 9 runs per side is in Before vs After. Esophagus is not comparable across versions.</td></tr>

<tr>
<td><img src="results/repro_original_seed0/plots/baseline_dice_vs_target_size.png" width="100%"></td>
<td><img src="results/repro_corrected_seed0/plots/baseline_dice_vs_target_size.png" width="100%"></td>
</tr>
<tr><td colspan="2"><b><code>baseline_dice_vs_target_size.png</code></b> shows GT-positive slice Dice by processed target-area bin (five quantile bins, patients averaged equally). Heart rises strongly with size in both versions (0.18 to 0.92 before, 0.23 to 0.92 after). Esophagus rises with size before (0.43 to 0.63) but stays low and flat after (0.09 to 0.21). Trachea shows no clean trend in either version. Aorta scores 0.25 in its smallest bin, peaks at 0.59 and is 0.41 to 0.46 in the larger bins. Some bins have only 2 contributing patients.</td></tr>

<tr>
<td><img src="results/repro_original_seed0/plots/baseline_dice_by_organ_position.png" width="100%"></td>
<td><img src="results/repro_corrected_seed0/plots/baseline_dice_by_organ_position.png" width="100%"></td>
</tr>
<tr><td colspan="2"><b><code>baseline_dice_by_organ_position.png</code></b> shows patient-grouped GT-positive slice Dice at the beginning (20%), middle (60%) and end (20%) of each organ's annotated extent. Heart is best in the middle in both versions (0.90 before, 0.92 after). Esophagus is best in the middle before (0.34 / 0.57 / 0.39) and low and flat after (0.14 / 0.20 / 0.22). Trachea is 0.29 / 0.52 / 0.53 before and 0.23 / 0.41 / 0.24 after. Aorta is 0.15 / 0.48 / 0.43, worst at the beginning.</td></tr>
</table>

## Shape Descriptors

- **Volume:** original GT voxel count × absolute spatial-affine determinant, converted from mm³ to mL.
- **Centroid:** mean foreground voxel indices, also stored in world coordinates through the NIfTI affine.
- **SI extent:** inferior-to-superior occupied-cell span in mm, including gaps and voxel thickness; not centreline length.
- **Normalized SI centroid:** world SI centroid scaled between inferior/superior scan voxel-centre limits (0–1); not anatomical registration.

## Qualitative Examples

`results/repro_corrected_seed0/examples/` contains twelve GT figures, named
`<organ>_<small|typical|large>_volume.png`; the corrected labels are shown below.
`results/repro_original_seed0/examples/` has the nine equivalents for the original labels
(no aorta; its esophagus examples carry the merged label 1).

<table>
<tr><th></th><th>Small volume</th><th>Typical volume</th><th>Large volume</th></tr>
<tr><th>Esophagus</th>
<td><img src="results/repro_corrected_seed0/examples/esophagus_small_volume.png" width="100%"></td>
<td><img src="results/repro_corrected_seed0/examples/esophagus_typical_volume.png" width="100%"></td>
<td><img src="results/repro_corrected_seed0/examples/esophagus_large_volume.png" width="100%"></td></tr>
<tr><th>Heart</th>
<td><img src="results/repro_corrected_seed0/examples/heart_small_volume.png" width="100%"></td>
<td><img src="results/repro_corrected_seed0/examples/heart_typical_volume.png" width="100%"></td>
<td><img src="results/repro_corrected_seed0/examples/heart_large_volume.png" width="100%"></td></tr>
<tr><th>Trachea</th>
<td><img src="results/repro_corrected_seed0/examples/trachea_small_volume.png" width="100%"></td>
<td><img src="results/repro_corrected_seed0/examples/trachea_typical_volume.png" width="100%"></td>
<td><img src="results/repro_corrected_seed0/examples/trachea_large_volume.png" width="100%"></td></tr>
<tr><th>Aorta</th>
<td><img src="results/repro_corrected_seed0/examples/aorta_small_volume.png" width="100%"></td>
<td><img src="results/repro_corrected_seed0/examples/aorta_typical_volume.png" width="100%"></td>
<td><img src="results/repro_corrected_seed0/examples/aorta_large_volume.png" width="100%"></td></tr>
</table>

Selection uses `shape_descriptors_3d.csv`: minimum volume, closest to the class
median, and maximum volume. Ties resolve by patient ID (median distances compared
to 9 decimal places). Organs with no annotation in a dataset (aorta, original labels)
get no examples. These are descriptive examples, not clinical categories.

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

CSV files are in `results/repro_<original|corrected>_seed0/tables/` (all git-ignored; only PNGs are tracked):

- `shape_descriptors_3d.csv`: 80 patient–organ rows with physical shape descriptors (the original version has 20 empty aorta rows).
- `patient_inventory.csv`: one row per patient with split, scan geometry, and label counts.
- `patient_class_stats.csv`: patient–organ counts, physical measurements, and slice coverage.
- `slice_class_stats.csv`: processed slice–organ target sizes and relative positions.
- `baseline_patient_metrics.csv`: validation patient–organ 3D Dice and slice summaries.
- `baseline_slice_metrics.csv`: validation slice–organ Dice, overlap, and empty-case indicators.

In `results/before_after/tables/`:

- `label_changes.csv`: per patient, esophagus voxels before/after, aorta voxels, and how many changed voxels are not a split of old label 1.
- `dice_3d_before_after.csv`: per organ, the number of runs, mean / min / max 3D Dice, and the number of runs at exactly 0, before and after.
- `metrics_3d_by_experiment.csv`: per organ (and overall) and metric (Dice, HD95, ASSD), the mean for the corrected-label baseline and for every experiment marker, with its number of seeds and whether the value is a placeholder (blank mean = organ had no ground truth).

## Run and Validation

Submit the CPU-only workflows from the repository root, in this order:

```bash
sbatch jobsAndOutputs/baseline/jobs/dataset_analysis_before_after.job   # both label versions (ONLY=corrected or ONLY=original runs one)
sbatch jobsAndOutputs/baseline/jobs/baseline_median_runs.job          # baseline stage from the median pipeline runs (run after the job above)
sbatch jobsAndOutputs/baseline/jobs/compare_before_after.job            # the before/after figures (all seeds; the --compare lines in the job list the experiment rows of the metrics figure)
```

The first job stitches each run's validation predictions to 3D, then runs numerical
tests, both analyses, example generation, and the results audit for each version. Logs go to
`jobsAndOutputs/baseline/outputs/`. Each folder's `validation.json` reports coverage,
geometry/metric checks, and example selection/alignment checks; input manifests and run
JSONs retain provenance. Pass `--predictions` and `--reconstructed-volumes` to
`analyze_baseline.py` explicitly. Without them it scores the old 3-organ predictions in
`results/segthor/ce`, which is what the older `dataset_analysis.job` and
`dataset_analysis_corrected.job` still do.

## Caveats

<<<<<<< HEAD
- Only five validation patients; adjacent slices are correlated. The baseline figures (everything except the before/after Dice figure) each use a single seeded pipeline run, the median of 9; the before/after Dice figure uses all 9 runs per side.
- **Small organs fail to train more often with the corrected labels.** No original-label run fails (0 of 9 at Dice 0 for esophagus and trachea), but in the corrected runs the esophagus is never learned in 3 of 9 seeds and the trachea in 2 of 9 (3D Dice exactly 0). Seeds 3 and 8 fail on both organs. Even the corrected runs that do learn the trachea average 0.38 (7 runs) against 0.52 before, on identical trachea labels. The difference is not statistically firm at 9 runs per side (p = 0.21 esophagus, 0.47 trachea) and the cause is untested. Restoring the stale labels does not change which seeds fail. Learned runs turn the esophagus on late (epochs 11-21 of 25).
- Esophagus results are not comparable across label versions: the original target also contained the aorta.
- **Stale processed labels were repaired.** `data/SEGTHOR_corrected` had held the old merged label 1 for Patient_02 (train) and Patient_15 (val). It was re-sliced on 2026-09-19 (`check_processed_gt.py`) and all corrected runs and analyses here were retrained/rebuilt afterwards. Results from before the repair are not comparable.
- Background includes unannotated anatomy. Normalized SI position depends on scan coverage, not anatomical registration.
- SI extent is not centreline length; a centroid plane need not contain every part of an organ.
- IQR bands show descriptive spread, not confidence intervals; size bins can contain different patients.
- Exploratory associations do not establish causality or clinical normal ranges.
- Joint-empty Dice is undefined; slice-quality trends use GT-positive slices.
=======
It follows the existing genoa/module/conda setup. Logs go to the existing
`jobsAndOutputs/baseline/outputs`; CSVs, plots and examples go exclusively to
`dataset_analysis/results`. Scripts overwrite their own named outputs but never
write to data, predictions, checkpoints, or existing training logs. Run manifests
record arguments, library versions, code hashes, Git commit, and source size/mtime.
Results and caches are ignored by Git. A dataset run must precede a baseline run.

The tests import the figure scripts, so they need the full `requirements.txt`
(pandas, seaborn, scipy); rebuild the environment with `env.job` if it predates it.

For numerical checks and a deterministic small end-to-end run:

```bash
python -B dataset_analysis/test_analysis.py
python -B dataset_analysis/analyze_dataset.py --max-patients 2 --output-dir dataset_analysis/results/smoke
python -B dataset_analysis/analyze_baseline.py --output-dir dataset_analysis/results/smoke
python -B dataset_analysis/validate_results.py --output-dir dataset_analysis/results/smoke
```

## Measurement conventions

- `patient_inventory.csv`, `patient_class_stats.csv`, and
  `class_frequency_original.csv` use ORIGINAL NIfTI labels. Physical volume is
  voxel count times the absolute determinant of the affine's spatial matrix;
  mm units and CT/GT geometry are checked. mL = mm³ / 1000. Bounding-box extent
  includes the full occupied voxel cells. Occupied-slice count can differ from
  first-to-last span when labels contain gaps. Background's fraction of
  foreground is undefined, recorded as blank.
- `slice_class_stats.csv` uses PROCESSED 256×256 PNGs, retaining every slice for
  each annotated class. Areas are pixels or fractions of image pixels, not mm².
  `slice_presence_summary.csv` gives pooled slice summaries by train/val/all.
- Scan-relative z is `index / (Z - 1)`; for Z=1 it is 0. This is not anatomical
  registration. Processed class-relative z is measured between its own first
  and last nonempty processed slices. It is undefined on GT-empty slices;
  a one-slice class gets 0.5 (middle). First/last distances are in slice indices.
- Area-versus-z plots include zeros. Slices are first averaged within each
  patient/bin; the line and IQR summarize those patient means. Tables retain
  both patient-bin values and train/val/all aggregates. IQR is descriptive,
  not a confidence interval. Five validation patients are not hundreds of
  independent observations despite their many correlated slices.
- `baseline_slice_metrics.csv` pairs GT and predictions by exact stem. Dice is
  `2*intersection/(GT area + predicted area)`, with no smoothing. Joint-empty
  Dice is undefined (blank), not 1. FP-only and complete-miss Dice are 0.
  Main Dice summaries use GT-positive slices; FP-only and joint-empty counts
  are reported separately. Predictions may contain class 4, but it has no
  metric row; predicting it on classes 1–3 still creates false negatives.
- Area bins use within-class quintiles with duplicate edges collapsed. Z bins
  use 10 equal scan intervals; class-extent bins use [0,.2), [.2,.8), [.8,1].
  Final bin endpoints are inclusive. Both pooled-slice and equal-patient
  summaries are saved in `baseline_binned_summary.csv`; detailed patient/bin
  summaries are in `baseline_patient_bins.csv`. Plotted shading is the IQR
  of patient mean Dice within the bin, not a slice-level confidence interval.
- `baseline_patient_metrics.csv` separates GT-positive mean/median slice Dice
  (processed grid) from full-mask 3D Dice (original grid). Existing reconstructed
  labels must have matching geometry and match the current prediction PNGs
  after the repository's exact 2× nearest-neighbor expansion. No reconstruction
  or inference is launched. Missing reconstructed files leave 3D cells blank.
- Example selections are deterministic (ties follow patient/slice order).
  Each class has six panels: smallest, nearest median, largest, lowest Dice,
  highest Dice and lowest-Dice extremity. The same slice can satisfy multiple
  criteria. Cyan is GT, orange is prediction. Crops retain PNG coordinates.

## Scope and limitations

Original anatomy and processed appearance are deliberately separate. Resizing
can alter small targets, extent and boundaries; expanding predictions to 512²
cannot recover lost detail. Frequency summaries pool voxels, whereas physical
volumes account for varying spacing. Validation slice distributions pool slices,
while explicitly named patient summaries give patients equal weight. These are
exploratory measurements, not a finalized course metric suite or a new method.

No dependency installation, GPU, training, complex morphology, clustering or
model modification is required. NumPy and standard-library CSV avoid adding
Pandas to the current environment.
>>>>>>> 35c00e72cc8eb216ef8f05d638d790bad9933a18
