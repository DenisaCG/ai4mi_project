# Full 40-patient SegTHOR training set versus the midterm 20

Source: `data/segthor_train_full/train` (corrected labels). `original_20` are the patient ids in
`data/segthor_part1_corrected/train` (the midterm set), `added_20` are the other 20 ids, `all_40` is both.
Everything is computed per patient first (`patient_*.csv`, `patient_wide.csv`); the tables below summarise those.
Old values in "Slide numbers that change" are recomputed with the same code from the midterm folder itself, not read off the slides.

## Are the first 20 patients the same files as at midterm?

Compared `20` patients between the two folders (geometry, per-organ voxel counts, 3D component counts, exact 1 HU histograms of the whole scan and each organ): geometry_equal = 20/20, label_voxels_equal = 3/20, components_equal = 14/20, histograms_equal = 3/20.

Organ voxel counts of the same patient id, full set minus midterm folder (patients that differ, total voxels): esophagus: 17/20 patients, +9028 voxels; heart: 0/20 patients, +0 voxels; trachea: 0/20 patients, +0 voxels; aorta: 17/20 patients, +6592 voxels. The CT volumes are compared through the geometry and histogram columns of `midterm_vs_full_first20.csv`.

## Group comparison (min / median / max over patients)

`differs` is True when the two groups are constant and unequal, or when a two-sided Mann-Whitney U test of
original_20 against added_20 gives p < 0.05. This is one test per quantity, 81 quantities, so about 4 rows are expected to flag by chance alone.
`outside` is how many of the 20 added patients lie outside the original 20's [min, max].

| quantity | original_20 | added_20 | all_40 | median added - original | outside | p | differs |
|---|---|---|---|---|---|---|---|
| pixel_size_left_right_mm | 0.9 / 0.98 / 1.37 | 0.98 / 0.98 / 1.37 | 0.9 / 0.98 / 1.37 | 0 | 0 | 0.647 | False |
| pixel_size_anterior_posterior_mm | 0.9 / 0.98 / 1.37 | 0.98 / 0.98 / 1.37 | 0.9 / 0.98 / 1.37 | 0 | 0 | 0.647 | False |
| slice_spacing_superior_inferior_mm | 2 / 2.5 / 2.5 | 2 / 2.5 / 2.5 | 2 / 2.5 / 2.5 | 0 | 0 | 1 | False |
| matrix_left_right | 512 / 512 / 512 | 512 / 512 / 512 | 512 / 512 / 512 | 0 | 0 |  | False |
| matrix_anterior_posterior | 512 / 512 / 512 | 512 / 512 / 512 | 512 / 512 / 512 | 0 | 0 |  | False |
| n_slices | 147 / 175 / 284 | 148 / 177.5 / 263 | 147 / 177.5 / 284 | 2.5 | 0 | 0.989 | False |
| scan_length_mm | 326 / 421.25 / 615 | 362 / 426.75 / 526 | 326 / 426.75 / 615 | 5.5 | 0 | 0.925 | False |
| voxels | 38535168 / 45875200 / 74448896 | 38797312 / 46530560 / 68943872 | 38535168 / 46530560 / 74448896 | 655360 | 0 | 0.989 | False |
| hu_all_voxels_p0.5 | -1000 / -1000 / -1000 | -1000 / -1000 / -1000 | -1000 / -1000 / -1000 | 0 | 0 |  | False |
| hu_aorta_p0.5 | -60 / -20.5 / 107 | -48.25 / -19 / 162 | -60 / -19 / 162 | 1.5 | 2 | 0.386 | False |
| hu_esophagus_p0.5 | -1000 / -618.5 / -147 | -1000 / -752.75 / -139.62 | -1000 / -684.07 / -139.62 | -134.25 | 1 | 0.543 | False |
| hu_heart_p0.5 | -312 / -143.5 / -90 | -282 / -147 / -78 | -312 / -146.5 / -78 | -3.5 | 1 | 0.745 | False |
| hu_trachea_p0.5 | -1000 / -993.5 / -951 | -1000 / -997 / -960 | -1000 / -996.5 / -951 | -3.5 | 0 | 0.66 | False |
| hu_all_voxels_p99.5 | 261 / 500 / 658 | 328 / 437.5 / 773 | 261 / 479.5 / 773 | -62.5 | 1 | 0.636 | False |
| hu_aorta_p99.5 | 75 / 179 / 545.32 | 83 / 170 / 356 | 75 / 170 / 545.32 | -9 | 0 | 0.417 | False |
| hu_esophagus_p99.5 | 72 / 115.29 / 190.59 | 84 / 108.11 / 261.06 | 72 / 110.4 / 261.06 | -7.18 | 2 | 0.925 | False |
| hu_heart_p99.5 | 74 / 169.51 / 388 | 74 / 116.5 / 427 | 74 / 139.5 / 427 | -53.01 | 1 | 0.766 | False |
| hu_trachea_p99.5 | -306.34 / -76.36 / 320 | -442.97 / -117.6 / 1.91 | -442.97 / -102 / 320 | -41.24 | 4 | 0.156 | False |
| hu_all_voxels_mean | -817.42 / -739.95 / -620.6 | -831.32 / -711.17 / -542.32 | -831.32 / -721.42 / -542.32 | 28.78 | 2 | 0.0962 | False |
| hu_aorta_mean | 33.39 / 43.75 / 201.91 | 29.58 / 43.99 / 255.52 | 29.58 / 43.75 / 255.52 | 0.24 | 5 | 0.715 | False |
| hu_esophagus_mean | -203.12 / 4.59 / 42.82 | -201.83 / -1.26 / 59.82 | -203.12 / 0.14 / 59.82 | -5.85 | 1 | 0.617 | False |
| hu_heart_mean | -0.47 / 27.78 / 110.37 | -21.53 / 23.92 / 148.88 | -21.53 / 25.98 / 148.88 | -3.85 | 6 | 0.903 | False |
| hu_trachea_mean | -895.44 / -809.74 / -652.31 | -916.74 / -814.24 / -689.76 | -916.74 / -814.24 / -652.31 | -4.49 | 2 | 0.968 | False |
| hu_all_voxels_median | -996 / -982 / -913 | -998 / -973.5 / -809 | -998 / -977.5 / -809 | 8.5 | 2 | 0.163 | False |
| hu_aorta_median | 34 / 42.5 / 209 | 30 / 44.5 / 260 | 30 / 43 / 260 | 2 | 4 | 0.56 | False |
| hu_esophagus_median | 4 / 21.5 / 51 | -3 / 26 / 64 | -3 / 22 / 64 | 4.5 | 2 | 0.645 | False |
| hu_heart_median | 22 / 38.5 / 133 | -4 / 36 / 174 | -4 / 38 / 174 | -2.5 | 6 | 0.903 | False |
| hu_trachea_median | -985 / -925.5 / -749 | -966 / -904.5 / -827 | -985 / -917.5 / -749 | 21 | 0 | 0.213 | False |
| hu_all_voxels_std | 348.3 / 419.23 / 467.69 | 346.07 / 432.96 / 464.95 | 346.07 / 427.09 / 467.69 | 13.73 | 1 | 0.172 | False |
| hu_aorta_std | 15.69 / 31 / 67.93 | 18.31 / 27.28 / 40.77 | 15.69 / 29.04 / 67.93 | -3.73 | 0 | 0.229 | False |
| hu_esophagus_std | 49.29 / 95.79 / 391.93 | 43.48 / 134.18 / 364.61 | 43.48 / 108.81 / 391.93 | 38.39 | 2 | 0.323 | False |
| hu_heart_std | 30.14 / 58.2 / 116.12 | 27.81 / 56.32 / 108.29 | 27.81 / 57.02 / 116.12 | -1.87 | 1 | 0.903 | False |
| hu_trachea_std | 142.01 / 231.66 / 397.69 | 117.13 / 202.97 / 287.6 | 117.13 / 219.64 / 397.69 | -28.7 | 3 | 0.148 | False |
| aorta_voxels | 39783 / 87903.5 / 231512 | 42740 / 99934 / 140752 | 39783 / 93593 / 231512 | 12030.5 | 0 | 0.695 | False |
| esophagus_voxels | 10521 / 19231.5 / 42545 | 10078 / 21233.5 / 71106 | 10078 / 20401.5 / 71106 | 2002 | 2 | 0.839 | False |
| heart_voxels | 183705 / 365749 / 767246 | 223689 / 362970 / 703594 | 183705 / 365280 / 767246 | -2778.5 | 0 | 0.989 | False |
| trachea_voxels | 7209 / 16948 / 36293 | 9516 / 15895.5 / 20413 | 7209 / 16600.5 / 36293 | -1052.5 | 0 | 0.262 | False |
| aorta_volume_ml | 94.85 / 203.09 / 465.16 | 141.32 / 211.16 / 290.74 | 94.85 / 206.15 / 465.16 | 8.07 | 0 | 0.598 | False |
| esophagus_volume_ml | 31.35 / 45.85 / 86.86 | 30.59 / 44.66 / 169.53 | 30.59 / 45.8 / 169.53 | -1.19 | 2 | 0.925 | False |
| heart_volume_ml | 437.99 / 862.05 / 1829.26 | 709.29 / 804.92 / 1677.5 | 437.99 / 846.98 / 1829.26 | -57.13 | 0 | 0.655 | False |
| trachea_volume_ml | 27.59 / 38.77 / 72.92 | 21.66 / 36.31 / 48.67 | 21.66 / 38 / 72.92 | -2.46 | 3 | 0.25 | False |
| aorta_present | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 0 | 0 |  | False |
| esophagus_present | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 0 | 0 |  | False |
| heart_present | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 0 | 0 |  | False |
| trachea_present | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 0 | 0 |  | False |
| aorta_share_of_scan_pct | 0.08 / 0.17 / 0.59 | 0.09 / 0.18 / 0.29 | 0.08 / 0.18 / 0.59 | 0.01 | 0 | 0.425 | False |
| esophagus_share_of_scan_pct | 0.02 / 0.04 / 0.07 | 0.02 / 0.04 / 0.14 | 0.02 / 0.04 / 0.14 | 0 | 2 | 0.797 | False |
| heart_share_of_scan_pct | 0.45 / 0.72 / 1.6 | 0.47 / 0.77 / 1.5 | 0.45 / 0.76 / 1.6 | 0.05 | 0 | 0.425 | False |
| trachea_share_of_scan_pct | 0.01 / 0.03 / 0.09 | 0.02 / 0.04 / 0.05 | 0.01 / 0.04 / 0.09 | 0 | 0 | 0.561 | False |
| aorta_share_of_foreground_pct | 10.72 / 18.23 / 28.73 | 13.07 / 18.06 / 25.76 | 10.72 / 18.19 / 28.73 | -0.17 | 0 | 0.946 | False |
| esophagus_share_of_foreground_pct | 2.67 / 3.89 / 7.51 | 2.71 / 4.01 / 12.39 | 2.67 / 4.01 / 12.39 | 0.12 | 1 | 0.735 | False |
| heart_share_of_foreground_pct | 63.48 / 73.28 / 84.77 | 66.61 / 74.48 / 82.63 | 63.48 / 73.69 / 84.77 | 1.2 | 0 | 0.508 | False |
| trachea_share_of_foreground_pct | 1.83 / 3.51 / 5.97 | 1.59 / 3.18 / 4.47 | 1.59 / 3.3 / 5.97 | -0.34 | 1 | 0.102 | False |
| aorta_components_face | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 0 | 0 |  | False |
| esophagus_components_face | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 0 | 0 |  | False |
| heart_components_face | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 0 | 0 |  | False |
| trachea_components_face | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 0 | 0 |  | False |
| aorta_components_face_edge_corner | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 0 | 0 |  | False |
| esophagus_components_face_edge_corner | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 0 | 0 |  | False |
| heart_components_face_edge_corner | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 0 | 0 |  | False |
| trachea_components_face_edge_corner | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 0 | 0 |  | False |
| aorta_slices_with_organ | 78 / 91 / 113 | 81 / 88.5 / 116 | 78 / 89 / 116 | -2.5 | 3 | 0.989 | False |
| esophagus_slices_with_organ | 80 / 99.5 / 136 | 79 / 90 / 122 | 79 / 94 / 136 | -9.5 | 1 | 0.0564 | False |
| heart_slices_with_organ | 27 / 42 / 62 | 27 / 38 / 45 | 27 / 40 / 62 | -4 | 0 | 0.0447 | True |
| trachea_slices_with_organ | 35 / 52 / 65 | 38 / 46 / 61 | 35 / 49 / 65 | -6 | 0 | 0.0252 | True |
| aorta_share_slices_2plus_components_pct | 10.89 / 21.91 / 29.55 | 14.04 / 20.67 / 30.59 | 10.89 / 21.56 / 30.59 | -1.24 | 1 | 0.646 | False |
| esophagus_share_slices_2plus_components_pct | 0 / 0 / 1.16 | 0 / 0 / 0 | 0 / 0 / 1.16 | 0 | 0 | 0.342 | False |
| heart_share_slices_2plus_components_pct | 0 / 0 / 0 | 0 / 0 / 2.38 | 0 / 0 / 2.38 | 0 | 1 | 0.342 | False |
| trachea_share_slices_2plus_components_pct | 11.11 / 15.83 / 21.74 | 6.52 / 15.54 / 20.41 | 6.52 / 15.67 / 21.74 | -0.29 | 1 | 0.957 | False |
| adjacency_esophagus-aorta_shared_face_area_mm2 | 0 / 84.42 / 1008.4 | 0 / 21.66 / 1125.11 | 0 / 67.93 / 1125.11 | -62.76 | 1 | 0.222 | False |
| adjacency_esophagus-heart_shared_face_area_mm2 | 0 / 197.85 / 787.09 | 0 / 197.48 / 1133.27 | 0 / 197.48 / 1133.27 | -0.37 | 1 | 0.534 | False |
| adjacency_esophagus-trachea_shared_face_area_mm2 | 53.14 / 345.38 / 1773.28 | 39.63 / 348.99 / 1239.97 | 39.63 / 345.38 / 1773.28 | 3.61 | 2 | 0.579 | False |
| adjacency_heart-aorta_shared_face_area_mm2 | 518.3 / 989.86 / 1714.71 | 695.34 / 952.52 / 1204.41 | 518.3 / 965.34 / 1714.71 | -37.33 | 0 | 0.646 | False |
| adjacency_heart-trachea_shared_face_area_mm2 | 0 / 0 / 36.16 | 0 / 0 / 115.24 | 0 / 0 / 115.24 | 0 | 1 | 0.724 | False |
| adjacency_trachea-aorta_shared_face_area_mm2 | 0 / 0 / 21.32 | 0 / 0 / 0.95 | 0 / 0 / 21.32 | 0 | 0 | 1 | False |
| adjacency_esophagus-aorta_touching | 0 / 1 / 1 | 0 / 1 / 1 | 0 / 1 / 1 | 0 | 0 | 0.447 | False |
| adjacency_esophagus-heart_touching | 0 / 1 / 1 | 0 / 1 / 1 | 0 / 1 / 1 | 0 | 0 | 1 | False |
| adjacency_esophagus-trachea_touching | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 0 | 0 |  | False |
| adjacency_heart-aorta_touching | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 0 | 0 |  | False |
| adjacency_heart-trachea_touching | 0 / 0 / 1 | 0 / 0 / 1 | 0 / 0 / 1 | 0 | 0 | 0.654 | False |
| adjacency_trachea-aorta_touching | 0 / 0 / 1 | 0 / 0 / 1 | 0 / 0 / 1 | 0 | 0 | 1 | False |

**Differ (2):** `heart_slices_with_organ`, `trachea_slices_with_organ`

**Do not differ (79):** `pixel_size_left_right_mm`, `pixel_size_anterior_posterior_mm`, `slice_spacing_superior_inferior_mm`, `matrix_left_right`, `matrix_anterior_posterior`, `n_slices`, `scan_length_mm`, `voxels`, `hu_all_voxels_p0.5`, `hu_aorta_p0.5`, `hu_esophagus_p0.5`, `hu_heart_p0.5`, `hu_trachea_p0.5`, `hu_all_voxels_p99.5`, `hu_aorta_p99.5`, `hu_esophagus_p99.5`, `hu_heart_p99.5`, `hu_trachea_p99.5`, `hu_all_voxels_mean`, `hu_aorta_mean`, `hu_esophagus_mean`, `hu_heart_mean`, `hu_trachea_mean`, `hu_all_voxels_median`, `hu_aorta_median`, `hu_esophagus_median`, `hu_heart_median`, `hu_trachea_median`, `hu_all_voxels_std`, `hu_aorta_std`, `hu_esophagus_std`, `hu_heart_std`, `hu_trachea_std`, `aorta_voxels`, `esophagus_voxels`, `heart_voxels`, `trachea_voxels`, `aorta_volume_ml`, `esophagus_volume_ml`, `heart_volume_ml`, `trachea_volume_ml`, `aorta_present`, `esophagus_present`, `heart_present`, `trachea_present`, `aorta_share_of_scan_pct`, `esophagus_share_of_scan_pct`, `heart_share_of_scan_pct`, `trachea_share_of_scan_pct`, `aorta_share_of_foreground_pct`, `esophagus_share_of_foreground_pct`, `heart_share_of_foreground_pct`, `trachea_share_of_foreground_pct`, `aorta_components_face`, `esophagus_components_face`, `heart_components_face`, `trachea_components_face`, `aorta_components_face_edge_corner`, `esophagus_components_face_edge_corner`, `heart_components_face_edge_corner`, `trachea_components_face_edge_corner`, `aorta_slices_with_organ`, `esophagus_slices_with_organ`, `aorta_share_slices_2plus_components_pct`, `esophagus_share_slices_2plus_components_pct`, `heart_share_slices_2plus_components_pct`, `trachea_share_slices_2plus_components_pct`, `adjacency_esophagus-aorta_shared_face_area_mm2`, `adjacency_esophagus-heart_shared_face_area_mm2`, `adjacency_esophagus-trachea_shared_face_area_mm2`, `adjacency_heart-aorta_shared_face_area_mm2`, `adjacency_heart-trachea_shared_face_area_mm2`, `adjacency_trachea-aorta_shared_face_area_mm2`, `adjacency_esophagus-aorta_touching`, `adjacency_esophagus-heart_touching`, `adjacency_esophagus-trachea_touching`, `adjacency_heart-aorta_touching`, `adjacency_heart-trachea_touching`, `adjacency_trachea-aorta_touching`

## Preprocessing the cross-validation actually used

| | roi window (px) | largest required (px, patient) | target spacing (mm) | window lo..hi (HU) | train patients |
|---|---|---|---|---|---|
| fold0 | 288 | 258 (Patient_20) | 0.9766 x 0.9766 x 2.5 | -982.0 .. 270.0 | 30 |
| fold1 | 320 | 286 (Patient_30) | 0.9766 x 0.9766 x 2.5 | -981.9 .. 274.0 | 30 |
| fold2 | 320 | 286 (Patient_30) | 0.9766 x 0.9766 x 2.5 | -972.7 .. 277.0 | 30 |
| fold3 | 320 | 286 (Patient_30) | 0.9766 x 0.9766 x 2.5 | -983.8 .. 262.0 | 30 |
| midterm dataset | 288 | 258 (Patient_20) | 0.9766 x 0.9766 x 2.5 | -993.2 .. 259.0 | 15 |

Midterm slide values: window 288 x 288, 0.98 x 0.98 x 2.5 mm, -993 to 259 HU.

## Slide numbers that change

Old = the 20 patients of the midterm folder, new = all 40 of the full folder. Medians are of per-patient values.

| number | old | new |
|---|---|---|
| patients with aorta voxels | 20 of 20 | 40 of 40 |
| aorta voxels, summed over patients | 1926407 | 3820699 |
| in-plane pixel size, median (mm) | 0.9766 x 0.9766 | 0.9766 x 0.9766 |
| slice spacing, min / median / max (mm) | 2 / 2.5 / 2.5 | 2 / 2.5 / 2.5 |
| number of slices, min / median / max | 147 / 175 / 284 | 147 / 177.5 / 284 |
| scan length, min / median / max (mm) | 326 / 421 / 615 | 326 / 427 / 615 |
| target spacing (mm) | 0.9766 x 0.9766 x 2.5 | per fold: 0.9766 x 0.9766 x 2.5; 0.9766 x 0.9766 x 2.5; 0.9766 x 0.9766 x 2.5; 0.9766 x 0.9766 x 2.5 |
| crop window (px) | 288 | per fold: 288, 320, 320, 320 |
| HU window (lo .. hi) | -993.2 .. 259.0 | per fold: -982.0 .. 270.0; -981.9 .. 274.0; -972.7 .. 277.0; -983.8 .. 262.0 |
| esophagus volume, min / median / max (mL) | 29.8 / 45.8 / 86.7 | 30.6 / 45.8 / 169.5 |
| heart volume, min / median / max (mL) | 438 / 862 / 1829.3 | 438 / 847 / 1829.3 |
| trachea volume, min / median / max (mL) | 27.6 / 38.8 / 72.9 | 21.7 / 38 / 72.9 |
| aorta volume, min / median / max (mL) | 93.4 / 202.4 / 466.3 | 94.8 / 206.1 / 465.2 |
| esophagus voxel share of foreground, pooled over patients (%) | 4.08 | 4.31 |
| heart voxel share of foreground, pooled over patients (%) | 73.46 | 73.76 |
| trachea voxel share of foreground, pooled over patients (%) | 3.58 | 3.29 |
| aorta voxel share of foreground, pooled over patients (%) | 18.88 | 18.64 |
| esophagus voxel share of scan, median (%) | 0.04 | 0.041 |
| heart voxel share of scan, median (%) | 0.723 | 0.764 |
| trachea voxel share of scan, median (%) | 0.035 | 0.035 |
| aorta voxel share of scan, median (%) | 0.173 | 0.181 |
| esophagus 3D components (face connectivity), min / median / max | 1 / 1 / 9 | 1 / 1 / 1 |
| heart 3D components (face connectivity), min / median / max | 1 / 1 / 1 | 1 / 1 / 1 |
| trachea 3D components (face connectivity), min / median / max | 1 / 1 / 1 | 1 / 1 / 1 |
| aorta 3D components (face connectivity), min / median / max | 1 / 1 / 67 | 1 / 1 / 1 |
| esophagus share of axial slices with 2+ components, median (%) | 0 | 0 |
| heart share of axial slices with 2+ components, median (%) | 0 | 0 |
| trachea share of axial slices with 2+ components, median (%) | 15.8 | 15.7 |
| aorta share of axial slices with 2+ components, median (%) | 22.4 | 21.6 |
| patients where esophagus-aorta share a voxel face | 12 of 20 | 32 of 40 |
| patients where esophagus-heart share a voxel face | 19 of 20 | 38 of 40 |
| patients where esophagus-trachea share a voxel face | 20 of 20 | 40 of 40 |
| patients where heart-aorta share a voxel face | 20 of 20 | 40 of 40 |
| patients where heart-trachea share a voxel face | 3 of 20 | 5 of 40 |
| patients where trachea-aorta share a voxel face | 2 of 20 | 2 of 40 |

Components are 6-connected in 3D (`components_face`; 26-connected counts are in `patient_labels.csv`). The 2+ components share uses 4-connected pieces per axial slice, over the slices that contain the organ.

## Figures

Rows are patients (Patient_01 at the top), circles are original_20 and squares are added_20. Axes are named left-right, anterior-posterior, superior-inferior.

- `geometry_per_patient.png`: pixel size (left-right, anterior-posterior), slice spacing, matrix size, number of slices and scan length (superior-inferior) per patient.
- `hu_per_patient.png`: HU median (marker) with p0.5 to p99.5 (line) and mean (tick) per patient, for all voxels and inside each organ.
- `organ_volume_per_patient.png`: organ volume in mL per patient, one panel per organ.
- `esophagus_components_per_patient.png`: number of 3D connected components of the esophagus, and share of its axial slices with 2+ components, per patient.

## Files

`patient_geometry.csv`, `patient_intensity.csv` (long: patient x region), `patient_labels.csv` (long: patient x organ), `patient_adjacency.csv` (long: patient x organ pair), `patient_wide.csv`, `comparison.csv`, `fold_preprocessing.csv`, `midterm_vs_full_first20.csv`, `inputs.csv` (size and mtime of every input), `run_log.json`. Regenerate with `jobsAndOutputs/dataset_40/jobs/dataset_40.job`; check with `tools/validate_dataset_40.py`.

## Class distribution along z

**File:** `figures/dataset_40/class_distribution_z/class_distribution_z.png` (+ `.pdf`, per-slice table `class_distribution_z.csv`, run log `class_distribution_z_log.json`)
**Script:** `dataset_analysis/profile_figures/class_distribution_z.py --data-dir data/segthor_train_full/train --out-dir figures/dataset_40` (job: `jobsAndOutputs/dataset_40/jobs/class_distribution_z.job`)
**Shows:** Top, % of patients whose slice at each height contains each organ; bottom, % of the 512×512 slice the organ covers on slices that contain it (median, 25–75% band, log scale). Shaded bands at each organ's median top and bottom edge show the ±2.5/±5/±10 mm reach of a 3-slice 2.5D input (context 1, stride 1/2/4).
**Unit and pooling:** z in mm relative to each patient's most superior heart slice (world z from the affine, LPS), 2.5 mm bins; one value per patient per bin, then pooled over the patients whose scan covers the bin (bins with fewer than 10 are dropped). Corrected labels, 40 patients.
**Useful for:** Seeing how little of an organ's z-extent (90–224 mm) a 2.5D context window covers, where organs start and stop, and how small the esophagus and trachea are on every slice.

## Copy-the-neighbour Dice (how much of an organ the 2.5D neighbour slices show)

**File:** `figures/dataset_40/neighbour_dice_z/neighbour_dice_z.png` and the one-panel `neighbour_dice_z_single.png` (mean Dice against distance, below/above pooled, solid = middle of organ, dashed = near an end with neighbours past the end left out) (+ `.pdf`, table `neighbour_dice_z.csv`, run log `neighbour_dice_z_log.json`)
**Script:** `dataset_analysis/profile_figures/neighbour_dice_z.py --data-dir data/segthor_train_full/train --out-dir figures/dataset_40` (job: `jobsAndOutputs/dataset_40/jobs/neighbour_dice_z.job`)
**Shows:** For every slice containing an organ, the Dice between its mask and the mask on the slice 2.5–10 mm below or above, i.e. the Dice a model would get by copying the neighbour. Three panels: centre slices near the organ's top, in its middle, near its bottom; one line per organ; s1/s2/s4 mark what a 3-slice 2.5D input sees at stride 1/2/4.
**Unit and pooling:** One value per (patient, organ, centre slice, offset), labels resampled along z to 2.5 mm (nearest neighbour); median and 25–75% over all centre slices of all 40 patients. Top/bottom = within 4 slices of that organ's own end in that patient. Corrected labels.
**Useful for:** Judging how much new information 2.5D context carries per organ and stride: in the middle of the heart the neighbours are near-copies even at ±10 mm (Dice 0.93), while esophagus and trachea drop to about 0.65, and near organ ends every organ drops sharply.
