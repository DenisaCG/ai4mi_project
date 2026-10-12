# Probability-based post-processing: full_cv4_resenc_ds_unet_dice_ce_strong_geometric_50ep, 3 seeds, 4 folds, 40 patients

Re-predicted argmax vs the stored volumes: 504 voxels differ in total, at most 11 in one patient volume (GPU rounding; 0 expected otherwise).
Reference rows (baseline, lcc_heart_trachea, seed_vote*) come from results/seed_ensemble/rows.csv; baseline_seed0 is seed 0 alone.
avg*: seeds' probabilities averaged, one result per fold. CRF settings (sxy, compat; sxy, srgb, compat): a gauss (3, 3) bilateral (20, 10, 5); b gauss (3, 3) bilateral (50, 5, 5); c gauss (5, 5) bilateral (40, 10, 10); 5 iterations.

### fg Dice and per-organ Dice

| variant | fg | fg vs baseline | esophagus (vs baseline) | heart (vs baseline) | trachea (vs baseline) | aorta (vs baseline) | fold 0 fg | fold 1 fg | fold 2 fg | fold 3 fg |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.8926 | +0.0000 | 0.8033 (+0.0000) | 0.9413 (+0.0000) | 0.9023 (+0.0000) | 0.9235 (+0.0000) | 0.9069 | 0.8762 | 0.8948 | 0.8926 |
| lcc_heart_trachea | 0.8935 | +0.0009 | 0.8033 (+0.0000) | 0.9417 (+0.0004) | 0.9054 (+0.0031) | 0.9235 (+0.0000) | 0.9075 | 0.8770 | 0.8966 | 0.8928 |
| seed_vote | 0.8988 | +0.0062 | 0.8141 (+0.0108) | 0.9458 (+0.0044) | 0.9064 (+0.0041) | 0.9289 (+0.0053) | 0.9135 | 0.8843 | 0.8989 | 0.8984 |
| seed_vote_lcc_heart_trachea | 0.8995 | +0.0069 | 0.8141 (+0.0108) | 0.9458 (+0.0045) | 0.9094 (+0.0070) | 0.9289 (+0.0053) | 0.9137 | 0.8853 | 0.9006 | 0.8985 |
| baseline_seed0 | 0.8919 | -0.0007 | 0.7999 (-0.0034) | 0.9438 (+0.0024) | 0.8991 (-0.0032) | 0.9248 (+0.0012) | 0.9042 | 0.8770 | 0.8928 | 0.8936 |
| avg | 0.8994 | +0.0068 | 0.8152 (+0.0119) | 0.9461 (+0.0047) | 0.9069 (+0.0045) | 0.9294 (+0.0059) | 0.9143 | 0.8850 | 0.8994 | 0.8989 |
| avg_lcc_heart_trachea | 0.9001 | +0.0075 | 0.8152 (+0.0119) | 0.9461 (+0.0048) | 0.9097 (+0.0074) | 0.9294 (+0.0059) | 0.9144 | 0.8859 | 0.9012 | 0.8989 |

### fg HD95 and per-organ HD95 (mm)

| variant | fg | fg vs baseline | esophagus (vs baseline) | heart (vs baseline) | trachea (vs baseline) | aorta (vs baseline) | fold 0 fg | fold 1 fg | fold 2 fg | fold 3 fg |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 9.76 | +0.00 | 9.04 (+0.00) | 9.93 (+0.00) | 9.55 (+0.00) | 10.52 (+0.00) | 9.01 | 11.97 | 8.09 | 9.97 |
| lcc_heart_trachea | 8.79 | -0.97 | 9.04 (+0.00) | 7.42 (-2.52) | 8.19 (-1.37) | 10.52 (+0.00) | 6.69 | 10.57 | 8.00 | 9.91 |
| seed_vote | 7.78 | -1.98 | 7.68 (-1.36) | 6.77 (-3.16) | 8.60 (-0.95) | 8.08 (-2.44) | 5.55 | 9.78 | 7.29 | 8.52 |
| seed_vote_lcc_heart_trachea | 7.56 | -2.20 | 7.68 (-1.36) | 6.68 (-3.25) | 7.81 (-1.75) | 8.08 (-2.44) | 5.55 | 8.83 | 7.38 | 8.49 |
| baseline_seed0 | 9.96 | +0.20 | 8.45 (-0.59) | 11.64 (+1.70) | 8.88 (-0.67) | 10.88 (+0.36) | 11.97 | 11.78 | 8.08 | 8.02 |
| avg | 7.76 | -2.00 | 7.65 (-1.40) | 6.76 (-3.17) | 8.57 (-0.99) | 8.05 (-2.46) | 5.44 | 9.87 | 7.26 | 8.46 |
| avg_lcc_heart_trachea | 7.53 | -2.23 | 7.65 (-1.40) | 6.66 (-3.28) | 7.78 (-1.78) | 8.05 (-2.46) | 5.43 | 8.91 | 7.35 | 8.45 |

### fg ASSD and per-organ ASSD (mm)

| variant | fg | fg vs baseline | esophagus (vs baseline) | heart (vs baseline) | trachea (vs baseline) | aorta (vs baseline) | fold 0 fg | fold 1 fg | fold 2 fg | fold 3 fg |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1.755 | +0.000 | 1.733 (+0.000) | 2.460 (+0.000) | 1.289 (+0.000) | 1.536 (+0.000) | 1.700 | 2.035 | 1.565 | 1.719 |
| lcc_heart_trachea | 1.614 | -0.141 | 1.733 (+0.000) | 2.116 (-0.343) | 1.068 (-0.221) | 1.536 (+0.000) | 1.326 | 1.906 | 1.514 | 1.709 |
| seed_vote | 1.492 | -0.262 | 1.546 (-0.187) | 1.978 (-0.482) | 1.139 (-0.150) | 1.306 (-0.230) | 1.202 | 1.782 | 1.463 | 1.522 |
| seed_vote_lcc_heart_trachea | 1.461 | -0.294 | 1.546 (-0.187) | 1.964 (-0.496) | 1.026 (-0.263) | 1.306 (-0.230) | 1.190 | 1.696 | 1.439 | 1.518 |
| baseline_seed0 | 1.815 | +0.060 | 1.674 (-0.060) | 2.720 (+0.260) | 1.251 (-0.038) | 1.616 (+0.079) | 2.102 | 2.011 | 1.605 | 1.543 |
| avg | 1.480 | -0.274 | 1.540 (-0.193) | 1.972 (-0.488) | 1.128 (-0.161) | 1.280 (-0.256) | 1.182 | 1.773 | 1.452 | 1.513 |
| avg_lcc_heart_trachea | 1.451 | -0.304 | 1.540 (-0.193) | 1.960 (-0.500) | 1.023 (-0.266) | 1.280 (-0.256) | 1.169 | 1.690 | 1.431 | 1.512 |

### Rows with HD95 above 20 mm, per 40 patient x fold rows

| variant | esophagus | heart | trachea | aorta | all |
|---|---|---|---|---|---|
| baseline | 3.0 | 1.7 | 4.3 | 4.7 | 13.7 |
| lcc_heart_trachea | 3.0 | 0.3 | 2.7 | 4.7 | 10.7 |
| seed_vote | 2.0 | 0.0 | 4.0 | 3.0 | 9.0 |
| seed_vote_lcc_heart_trachea | 2.0 | 0.0 | 2.0 | 3.0 | 7.0 |
| baseline_seed0 | 2.0 | 1.0 | 4.0 | 6.0 | 13.0 |
| avg | 2.0 | 0.0 | 4.0 | 3.0 | 9.0 |
| avg_lcc_heart_trachea | 2.0 | 0.0 | 2.0 | 3.0 | 7.0 |

### False-positive slices, per 40 patient x fold rows

| variant | esophagus | heart | trachea | aorta | all |
|---|---|---|---|---|---|
| baseline | 102.3 | 121.0 | 170.0 | 193.7 | 587.0 |
| lcc_heart_trachea | 102.3 | 94.3 | 138.3 | 193.7 | 528.7 |
| seed_vote | 90.0 | 90.0 | 153.0 | 165.0 | 498.0 |
| seed_vote_lcc_heart_trachea | 90.0 | 77.0 | 135.0 | 165.0 | 467.0 |
| baseline_seed0 | 99.0 | 88.0 | 172.0 | 192.0 | 551.0 |
| avg | 88.0 | 97.0 | 153.0 | 160.0 | 498.0 |
| avg_lcc_heart_trachea | 88.0 | 76.0 | 135.0 | 160.0 | 459.0 |

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
