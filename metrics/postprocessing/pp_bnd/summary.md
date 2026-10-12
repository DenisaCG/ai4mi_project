# Probability-based post-processing: full_cv4_resenc_ds_unet_dice_ce_strong_geometric_50ep_boundary, 3 seeds, 4 folds, 40 patients

Re-predicted argmax vs the stored volumes: 497 voxels differ in total, at most 15 in one patient volume (GPU rounding; 0 expected otherwise).
Reference rows (baseline, lcc_heart_trachea, seed_vote*) come from results/seed_ensemble/rows.csv; baseline_seed0 is seed 0 alone.
avg*: seeds' probabilities averaged, one result per fold. CRF settings (sxy, compat; sxy, srgb, compat): a gauss (3, 3) bilateral (20, 10, 5); b gauss (3, 3) bilateral (50, 5, 5); c gauss (5, 5) bilateral (40, 10, 10); 5 iterations.

### fg Dice and per-organ Dice

| variant | fg | fg vs baseline | esophagus (vs baseline) | heart (vs baseline) | trachea (vs baseline) | aorta (vs baseline) | fold 0 fg | fold 1 fg | fold 2 fg | fold 3 fg |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.8906 | +0.0000 | 0.8031 (+0.0000) | 0.9388 (+0.0000) | 0.9002 (+0.0000) | 0.9203 (+0.0000) | 0.9043 | 0.8773 | 0.8908 | 0.8899 |
| lcc_heart_trachea | 0.8923 | +0.0017 | 0.8031 (+0.0000) | 0.9406 (+0.0018) | 0.9051 (+0.0050) | 0.9203 (+0.0000) | 0.9056 | 0.8789 | 0.8929 | 0.8917 |
| seed_vote | 0.8973 | +0.0067 | 0.8135 (+0.0104) | 0.9450 (+0.0062) | 0.9063 (+0.0061) | 0.9245 (+0.0042) | 0.9111 | 0.8855 | 0.8960 | 0.8967 |
| seed_vote_lcc_heart_trachea | 0.8982 | +0.0076 | 0.8135 (+0.0104) | 0.9450 (+0.0062) | 0.9097 (+0.0095) | 0.9245 (+0.0042) | 0.9112 | 0.8870 | 0.8974 | 0.8972 |
| baseline_seed0 | 0.8897 | -0.0009 | 0.8040 (+0.0009) | 0.9360 (-0.0028) | 0.8997 (-0.0005) | 0.9190 (-0.0013) | 0.9037 | 0.8746 | 0.8902 | 0.8902 |
| avg | 0.8978 | +0.0072 | 0.8143 (+0.0112) | 0.9454 (+0.0066) | 0.9066 (+0.0064) | 0.9249 (+0.0046) | 0.9114 | 0.8861 | 0.8967 | 0.8969 |
| avg_lcc_heart_trachea | 0.8986 | +0.0080 | 0.8143 (+0.0112) | 0.9454 (+0.0066) | 0.9098 (+0.0097) | 0.9249 (+0.0046) | 0.9114 | 0.8875 | 0.8982 | 0.8973 |

### fg HD95 and per-organ HD95 (mm)

| variant | fg | fg vs baseline | esophagus (vs baseline) | heart (vs baseline) | trachea (vs baseline) | aorta (vs baseline) | fold 0 fg | fold 1 fg | fold 2 fg | fold 3 fg |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 13.05 | +0.00 | 9.10 (+0.00) | 19.52 (+0.00) | 13.13 (+0.00) | 10.46 (+0.00) | 14.18 | 10.06 | 9.90 | 18.05 |
| lcc_heart_trachea | 8.86 | -4.19 | 9.10 (+0.00) | 7.65 (-11.87) | 8.24 (-4.89) | 10.46 (+0.00) | 6.66 | 9.36 | 9.46 | 9.95 |
| seed_vote | 8.15 | -4.90 | 8.41 (-0.68) | 7.12 (-12.40) | 8.35 (-4.77) | 8.71 (-1.74) | 5.98 | 9.39 | 8.63 | 8.60 |
| seed_vote_lcc_heart_trachea | 8.00 | -5.05 | 8.41 (-0.68) | 7.05 (-12.46) | 7.83 (-5.30) | 8.71 (-1.74) | 5.99 | 8.50 | 8.69 | 8.82 |
| baseline_seed0 | 13.67 | +0.63 | 8.14 (-0.96) | 13.03 (-6.49) | 21.84 (+8.71) | 11.69 (+1.24) | 24.00 | 10.62 | 11.01 | 9.06 |
| avg | 8.09 | -4.96 | 8.33 (-0.77) | 7.05 (-12.47) | 8.38 (-4.75) | 8.61 (-1.85) | 6.01 | 9.49 | 8.28 | 8.58 |
| avg_lcc_heart_trachea | 7.95 | -5.10 | 8.33 (-0.77) | 7.03 (-12.49) | 7.85 (-5.27) | 8.61 (-1.85) | 6.02 | 8.61 | 8.37 | 8.81 |

### fg ASSD and per-organ ASSD (mm)

| variant | fg | fg vs baseline | esophagus (vs baseline) | heart (vs baseline) | trachea (vs baseline) | aorta (vs baseline) | fold 0 fg | fold 1 fg | fold 2 fg | fold 3 fg |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 2.041 | +0.000 | 1.741 (+0.000) | 3.428 (+0.000) | 1.492 (+0.000) | 1.505 (+0.000) | 2.098 | 1.908 | 1.728 | 2.432 |
| lcc_heart_trachea | 1.618 | -0.424 | 1.741 (+0.000) | 2.182 (-1.246) | 1.044 (-0.448) | 1.505 (+0.000) | 1.385 | 1.746 | 1.657 | 1.684 |
| seed_vote | 1.500 | -0.542 | 1.553 (-0.188) | 2.028 (-1.401) | 1.105 (-0.386) | 1.313 (-0.192) | 1.262 | 1.679 | 1.561 | 1.497 |
| seed_vote_lcc_heart_trachea | 1.466 | -0.575 | 1.553 (-0.188) | 2.016 (-1.412) | 0.982 (-0.510) | 1.313 (-0.192) | 1.256 | 1.575 | 1.544 | 1.490 |
| baseline_seed0 | 2.216 | +0.175 | 1.707 (-0.034) | 3.530 (+0.101) | 2.050 (+0.558) | 1.579 (+0.074) | 3.276 | 2.087 | 1.867 | 1.635 |
| avg | 1.489 | -0.553 | 1.546 (-0.196) | 2.016 (-1.412) | 1.093 (-0.399) | 1.300 (-0.204) | 1.261 | 1.669 | 1.530 | 1.496 |
| avg_lcc_heart_trachea | 1.460 | -0.582 | 1.546 (-0.196) | 2.009 (-1.419) | 0.983 (-0.508) | 1.300 (-0.204) | 1.257 | 1.576 | 1.515 | 1.490 |

### Rows with HD95 above 20 mm, per 40 patient x fold rows

| variant | esophagus | heart | trachea | aorta | all |
|---|---|---|---|---|---|
| baseline | 3.3 | 4.3 | 3.7 | 6.3 | 17.7 |
| lcc_heart_trachea | 3.3 | 0.0 | 1.3 | 6.3 | 11.0 |
| seed_vote | 3.0 | 0.0 | 3.0 | 4.0 | 10.0 |
| seed_vote_lcc_heart_trachea | 3.0 | 0.0 | 1.0 | 4.0 | 8.0 |
| baseline_seed0 | 3.0 | 2.0 | 6.0 | 8.0 | 19.0 |
| avg | 3.0 | 0.0 | 3.0 | 3.0 | 9.0 |
| avg_lcc_heart_trachea | 3.0 | 0.0 | 1.0 | 3.0 | 7.0 |

### False-positive slices, per 40 patient x fold rows

| variant | esophagus | heart | trachea | aorta | all |
|---|---|---|---|---|---|
| baseline | 130.0 | 297.3 | 173.3 | 187.7 | 788.3 |
| lcc_heart_trachea | 130.0 | 111.7 | 127.3 | 187.7 | 556.7 |
| seed_vote | 107.0 | 110.0 | 150.0 | 158.0 | 525.0 |
| seed_vote_lcc_heart_trachea | 107.0 | 98.0 | 120.0 | 158.0 | 483.0 |
| baseline_seed0 | 128.0 | 134.0 | 213.0 | 173.0 | 648.0 |
| avg | 105.0 | 100.0 | 146.0 | 155.0 | 506.0 |
| avg_lcc_heart_trachea | 105.0 | 91.0 | 120.0 | 155.0 | 471.0 |

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
