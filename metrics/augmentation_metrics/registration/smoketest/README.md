# Fold-0 registration smoketest

One pair from the production cyclic plan: Patient_26 -> Patient_19, pairing seed 42. `volumes/train/Synthetic_f0_00/` preserves the CT/GT produced by job 27793262. `reports/qc_existing.json` records their disk-readback geometry, intensity, organ volumes and actual real-fold ROI retention. No production registration manifest or cache was touched.

A second execution of the same pair writes CT/GT to `volumes/rerun/train/Synthetic_f0_00/` without overwriting the original. `reports/rerun_comparison.json` counts any differences. `reports/transform.h5` is the safely flattened, reread and point-verified transform for the rerun pair; the old incomplete file is `reports/transform.failed_27793262.h5`. If the pairs differ, use the rerun figures and preprocessing report to interpret the transform.

`reports/qc.json` adds exact settings, optimizer outcomes, stage runtimes, source checksums, and deformation statistics.

`figures/registration_overview.png` compares CTs on the same target physical grid; `figures/organ_overlays.png` selects organ-present slices on each native grid; `figures/deformation.png` shows the B-spline component, full mapping, and full Jacobian on a coarse 4 x 4 x 5 mm target grid. Nonpositive Jacobians there indicate possible folding; a positive coarse-grid result cannot exclude folding between samples.

The original saved pair has finite CT values from -1000 to 3071 HU, contains all four labels, and has exactly the target CT geometry. All four organs retain 100% of their resampled label voxels after the real fold-0 ROI crop. The repeat also retains 100%, but it differs from the original in 35,263,607 CT voxels and 112,501 GT voxels. Its rigid stage ended at the configured 150-iteration limit; its B-spline stage met function tolerance after 26 iterations, versus 13 in the prior job. The cause of the differing optimizer outcomes has not been established. `reports/transform.h5` and the deformation diagnostics belong to the **repeat**, not the original pair.

On the coarse diagnostic grid, the repeat's B-spline displacement reaches 12.74 mm; the full-transform Jacobian ranges from 0.846 to 1.225, with no nonpositive samples. The full displacement includes the large rigid shift between different scanner origins. The source column in `registration_overview.png` is black because the source and target physical grids do not overlap without registration; inspect the native source in `organ_overlays.png` instead. The overlays of warped labels on the target CT show possible anatomical misalignment, particularly around the heart and upper mediastinum. A human should review both original and repeat organ overlays before treating either registered pair as acceptable training data.

The 120-pair extrapolation multiplies this pair's registration-and-write runtime by 120. It excludes this pilot's figure and QC overhead; patient-to-patient runtime can vary. Automated checks do not establish anatomical plausibility. Inspect all figures before reuse.

This QC completion was run with `sbatch jobsAndOutputs/augmentation/jobs/registration/smoketest.job` from the repository root after the original CT/GT pair from job 27793262 was present. The completed no-augmentation ENet and ResEnc + DS U-Net fold-0 control logs both recorded NVIDIA H100 GPUs; this registration generation uses only CPU resources.
