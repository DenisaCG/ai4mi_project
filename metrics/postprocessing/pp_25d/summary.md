# Probability-based post-processing: full_cv4_resenc_ds_unet_dice_ce_boundary_25d_c1_strong_geometric_50ep, 3 seeds, 4 folds, 40 patients

Re-predicted argmax vs the stored volumes: 511 voxels differ in total, at most 13 in one patient volume (GPU rounding; 0 expected otherwise).
Reference rows (baseline, lcc_heart_trachea, seed_vote*) come from results/seed_ensemble/rows.csv; baseline_seed0 is seed 0 alone.
avg*: seeds' probabilities averaged, one result per fold. CRF settings (sxy, compat; sxy, srgb, compat): a gauss (3, 3) bilateral (20, 10, 5); b gauss (3, 3) bilateral (50, 5, 5); c gauss (5, 5) bilateral (40, 10, 10); 5 iterations.

### fg Dice and per-organ Dice

| variant | fg | fg vs baseline | esophagus (vs baseline) | heart (vs baseline) | trachea (vs baseline) | aorta (vs baseline) | fold 0 fg | fold 1 fg | fold 2 fg | fold 3 fg |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.8929 | +0.0000 | 0.8059 (+0.0000) | 0.9449 (+0.0000) | 0.9003 (+0.0000) | 0.9206 (+0.0000) | 0.9067 | 0.8778 | 0.8953 | 0.8919 |
| lcc_heart_trachea | 0.8931 | +0.0001 | 0.8059 (+0.0000) | 0.9454 (+0.0005) | 0.9003 (+0.0000) | 0.9206 (+0.0000) | 0.9070 | 0.8754 | 0.8970 | 0.8929 |
| seed_vote | 0.8991 | +0.0062 | 0.8158 (+0.0099) | 0.9489 (+0.0040) | 0.9062 (+0.0059) | 0.9256 (+0.0050) | 0.9123 | 0.8853 | 0.9006 | 0.8983 |
| seed_vote_lcc_heart_trachea | 0.8993 | +0.0063 | 0.8158 (+0.0099) | 0.9493 (+0.0044) | 0.9063 (+0.0059) | 0.9256 (+0.0050) | 0.9124 | 0.8833 | 0.9023 | 0.8990 |
| baseline_seed0 | 0.8936 | +0.0007 | 0.8048 (-0.0011) | 0.9469 (+0.0020) | 0.9002 (-0.0001) | 0.9225 (+0.0019) | 0.9081 | 0.8772 | 0.8954 | 0.8937 |
| avg | 0.8995 | +0.0066 | 0.8165 (+0.0106) | 0.9492 (+0.0043) | 0.9063 (+0.0060) | 0.9260 (+0.0054) | 0.9126 | 0.8857 | 0.9011 | 0.8986 |
| avg_lcc_heart_trachea | 0.8996 | +0.0067 | 0.8165 (+0.0106) | 0.9496 (+0.0047) | 0.9064 (+0.0060) | 0.9260 (+0.0054) | 0.9127 | 0.8837 | 0.9028 | 0.8994 |

### fg HD95 and per-organ HD95 (mm)

| variant | fg | fg vs baseline | esophagus (vs baseline) | heart (vs baseline) | trachea (vs baseline) | aorta (vs baseline) | fold 0 fg | fold 1 fg | fold 2 fg | fold 3 fg |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 9.53 | +0.00 | 8.98 (+0.00) | 7.56 (+0.00) | 11.67 (+0.00) | 9.93 (+0.00) | 9.25 | 10.36 | 8.74 | 9.78 |
| lcc_heart_trachea | 8.58 | -0.96 | 8.98 (+0.00) | 6.76 (-0.79) | 8.63 (-3.04) | 9.93 (+0.00) | 6.07 | 10.15 | 8.41 | 9.66 |
| seed_vote | 7.57 | -1.97 | 8.03 (-0.95) | 6.29 (-1.27) | 7.18 (-4.49) | 8.77 (-1.16) | 5.35 | 8.35 | 7.95 | 8.61 |
| seed_vote_lcc_heart_trachea | 7.57 | -1.96 | 8.03 (-0.95) | 5.81 (-1.75) | 7.69 (-3.98) | 8.77 (-1.16) | 5.45 | 8.97 | 7.46 | 8.42 |
| baseline_seed0 | 8.30 | -1.24 | 8.83 (-0.15) | 6.65 (-0.91) | 8.05 (-3.62) | 9.66 (-0.27) | 5.90 | 9.83 | 8.89 | 8.56 |
| avg | 7.50 | -2.03 | 8.04 (-0.94) | 6.18 (-1.38) | 7.21 (-4.46) | 8.58 (-1.35) | 5.33 | 8.22 | 7.83 | 8.62 |
| avg_lcc_heart_trachea | 7.52 | -2.01 | 8.04 (-0.94) | 5.76 (-1.80) | 7.72 (-3.95) | 8.58 (-1.35) | 5.43 | 8.84 | 7.40 | 8.42 |

### fg ASSD and per-organ ASSD (mm)

| variant | fg | fg vs baseline | esophagus (vs baseline) | heart (vs baseline) | trachea (vs baseline) | aorta (vs baseline) | fold 0 fg | fold 1 fg | fold 2 fg | fold 3 fg |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1.677 | +0.000 | 1.732 (+0.000) | 2.138 (+0.000) | 1.330 (+0.000) | 1.508 (+0.000) | 1.512 | 1.845 | 1.592 | 1.760 |
| lcc_heart_trachea | 1.592 | -0.085 | 1.732 (+0.000) | 2.007 (-0.131) | 1.120 (-0.210) | 1.508 (+0.000) | 1.308 | 1.824 | 1.536 | 1.699 |
| seed_vote | 1.445 | -0.232 | 1.525 (-0.207) | 1.881 (-0.258) | 1.014 (-0.316) | 1.360 (-0.148) | 1.190 | 1.596 | 1.478 | 1.516 |
| seed_vote_lcc_heart_trachea | 1.434 | -0.243 | 1.525 (-0.207) | 1.834 (-0.304) | 1.017 (-0.313) | 1.360 (-0.148) | 1.194 | 1.635 | 1.427 | 1.481 |
| baseline_seed0 | 1.572 | -0.105 | 1.704 (-0.028) | 1.969 (-0.170) | 1.161 (-0.169) | 1.453 (-0.055) | 1.297 | 1.781 | 1.564 | 1.646 |
| avg | 1.442 | -0.235 | 1.535 (-0.198) | 1.872 (-0.267) | 1.018 (-0.312) | 1.345 (-0.163) | 1.187 | 1.600 | 1.466 | 1.516 |
| avg_lcc_heart_trachea | 1.432 | -0.246 | 1.535 (-0.198) | 1.829 (-0.310) | 1.018 (-0.312) | 1.345 (-0.163) | 1.192 | 1.636 | 1.419 | 1.479 |

### Rows with HD95 above 20 mm, per 40 patient x fold rows

| variant | esophagus | heart | trachea | aorta | all |
|---|---|---|---|---|---|
| baseline | 3.3 | 0.7 | 2.7 | 5.0 | 11.7 |
| lcc_heart_trachea | 3.3 | 0.0 | 3.3 | 5.0 | 11.7 |
| seed_vote | 3.0 | 0.0 | 1.0 | 4.0 | 8.0 |
| seed_vote_lcc_heart_trachea | 3.0 | 0.0 | 1.0 | 4.0 | 8.0 |
| baseline_seed0 | 3.0 | 0.0 | 2.0 | 6.0 | 11.0 |
| avg | 3.0 | 0.0 | 2.0 | 4.0 | 9.0 |
| avg_lcc_heart_trachea | 3.0 | 0.0 | 2.0 | 4.0 | 9.0 |

### False-positive slices, per 40 patient x fold rows

| variant | esophagus | heart | trachea | aorta | all |
|---|---|---|---|---|---|
| baseline | 126.0 | 144.7 | 159.7 | 172.3 | 602.7 |
| lcc_heart_trachea | 126.0 | 104.7 | 128.0 | 172.3 | 531.0 |
| seed_vote | 96.0 | 102.0 | 136.0 | 147.0 | 481.0 |
| seed_vote_lcc_heart_trachea | 96.0 | 82.0 | 120.0 | 147.0 | 445.0 |
| baseline_seed0 | 126.0 | 139.0 | 141.0 | 178.0 | 584.0 |
| avg | 96.0 | 100.0 | 135.0 | 143.0 | 474.0 |
| avg_lcc_heart_trachea | 96.0 | 80.0 | 120.0 | 143.0 | 439.0 |

### Rows with an empty prediction (undefined HD95), per 40 patient x fold rows

| variant | esophagus | heart | trachea | aorta | all |
|---|---|---|---|---|---|
| baseline | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| lcc_heart_trachea | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| seed_vote | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| seed_vote_lcc_heart_trachea | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| baseline_seed0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| avg | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| avg_lcc_heart_trachea | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
