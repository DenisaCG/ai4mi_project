# Probability-based post-processing: full_cv4_resenc_ds_unet_dice_ce_strong_geometric_50ep_boundary_w008, 3 seeds, 4 folds, 40 patients

Re-predicted argmax vs the stored volumes: 499 voxels differ in total, at most 16 in one patient volume (GPU rounding; 0 expected otherwise).
Reference rows (baseline, lcc_heart_trachea, seed_vote*) come from results/seed_ensemble/rows.csv; baseline_seed0 is seed 0 alone.
avg*: seeds' probabilities averaged, one result per fold. CRF settings (sxy, compat; sxy, srgb, compat): a gauss (3, 3) bilateral (20, 10, 5); b gauss (3, 3) bilateral (50, 5, 5); c gauss (5, 5) bilateral (40, 10, 10); 5 iterations.

### fg Dice and per-organ Dice

| variant | fg | fg vs baseline | esophagus (vs baseline) | heart (vs baseline) | trachea (vs baseline) | aorta (vs baseline) | fold 0 fg | fold 1 fg | fold 2 fg | fold 3 fg |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.8915 | +0.0000 | 0.8023 (+0.0000) | 0.9415 (+0.0000) | 0.8995 (+0.0000) | 0.9225 (+0.0000) | 0.9052 | 0.8776 | 0.8911 | 0.8920 |
| lcc_heart_trachea | 0.8927 | +0.0012 | 0.8023 (+0.0000) | 0.9417 (+0.0002) | 0.9041 (+0.0046) | 0.9225 (+0.0000) | 0.9057 | 0.8790 | 0.8931 | 0.8928 |
| seed_vote | 0.8978 | +0.0064 | 0.8133 (+0.0109) | 0.9451 (+0.0035) | 0.9056 (+0.0060) | 0.9274 (+0.0049) | 0.9112 | 0.8867 | 0.8951 | 0.8983 |
| seed_vote_lcc_heart_trachea | 0.8984 | +0.0070 | 0.8133 (+0.0109) | 0.9451 (+0.0035) | 0.9079 (+0.0084) | 0.9274 (+0.0049) | 0.9113 | 0.8874 | 0.8965 | 0.8984 |
| baseline_seed0 | 0.8914 | -0.0001 | 0.8046 (+0.0022) | 0.9407 (-0.0009) | 0.8976 (-0.0020) | 0.9227 (+0.0002) | 0.9057 | 0.8775 | 0.8887 | 0.8936 |
| avg | 0.8982 | +0.0067 | 0.8135 (+0.0112) | 0.9453 (+0.0038) | 0.9060 (+0.0065) | 0.9278 (+0.0053) | 0.9117 | 0.8865 | 0.8956 | 0.8989 |
| avg_lcc_heart_trachea | 0.8987 | +0.0072 | 0.8135 (+0.0112) | 0.9453 (+0.0038) | 0.9083 (+0.0087) | 0.9278 (+0.0053) | 0.9118 | 0.8872 | 0.8970 | 0.8989 |

### fg HD95 and per-organ HD95 (mm)

| variant | fg | fg vs baseline | esophagus (vs baseline) | heart (vs baseline) | trachea (vs baseline) | aorta (vs baseline) | fold 0 fg | fold 1 fg | fold 2 fg | fold 3 fg |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 9.19 | +0.00 | 8.95 (+0.00) | 8.07 (+0.00) | 9.49 (+0.00) | 10.24 (+0.00) | 7.14 | 10.66 | 8.56 | 10.39 |
| lcc_heart_trachea | 8.82 | -0.36 | 8.95 (+0.00) | 7.89 (-0.18) | 8.21 (-1.28) | 10.24 (+0.00) | 6.73 | 9.94 | 8.43 | 10.19 |
| seed_vote | 8.15 | -1.04 | 8.15 (-0.79) | 7.30 (-0.77) | 8.70 (-0.79) | 8.42 (-1.82) | 6.00 | 8.98 | 8.37 | 9.24 |
| seed_vote_lcc_heart_trachea | 7.91 | -1.28 | 8.15 (-0.79) | 7.29 (-0.78) | 7.77 (-1.72) | 8.42 (-1.82) | 5.99 | 8.06 | 8.40 | 9.17 |
| baseline_seed0 | 8.93 | -0.26 | 9.23 (+0.29) | 7.95 (-0.12) | 9.05 (-0.44) | 9.49 (-0.75) | 6.53 | 9.93 | 8.79 | 10.47 |
| avg | 8.12 | -1.07 | 8.09 (-0.85) | 7.29 (-0.77) | 8.65 (-0.84) | 8.45 (-1.80) | 5.97 | 8.95 | 8.36 | 9.20 |
| avg_lcc_heart_trachea | 7.90 | -1.29 | 8.09 (-0.85) | 7.30 (-0.77) | 7.75 (-1.74) | 8.45 (-1.80) | 5.97 | 8.05 | 8.40 | 9.16 |

### fg ASSD and per-organ ASSD (mm)

| variant | fg | fg vs baseline | esophagus (vs baseline) | heart (vs baseline) | trachea (vs baseline) | aorta (vs baseline) | fold 0 fg | fold 1 fg | fold 2 fg | fold 3 fg |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1.688 | +0.000 | 1.726 (+0.000) | 2.225 (+0.000) | 1.315 (+0.000) | 1.486 (+0.000) | 1.509 | 1.873 | 1.616 | 1.754 |
| lcc_heart_trachea | 1.614 | -0.074 | 1.726 (+0.000) | 2.173 (-0.051) | 1.070 (-0.244) | 1.486 (+0.000) | 1.372 | 1.788 | 1.568 | 1.728 |
| seed_vote | 1.497 | -0.190 | 1.546 (-0.180) | 2.054 (-0.171) | 1.095 (-0.220) | 1.295 (-0.191) | 1.261 | 1.610 | 1.538 | 1.581 |
| seed_vote_lcc_heart_trachea | 1.479 | -0.209 | 1.546 (-0.180) | 2.053 (-0.172) | 1.021 (-0.294) | 1.295 (-0.191) | 1.250 | 1.564 | 1.527 | 1.575 |
| baseline_seed0 | 1.666 | -0.022 | 1.732 (+0.006) | 2.220 (-0.005) | 1.234 (-0.081) | 1.479 (-0.007) | 1.373 | 1.834 | 1.638 | 1.820 |
| avg | 1.491 | -0.197 | 1.553 (-0.173) | 2.050 (-0.175) | 1.087 (-0.228) | 1.275 (-0.211) | 1.255 | 1.616 | 1.532 | 1.562 |
| avg_lcc_heart_trachea | 1.474 | -0.214 | 1.553 (-0.173) | 2.049 (-0.176) | 1.019 (-0.296) | 1.275 (-0.211) | 1.246 | 1.570 | 1.522 | 1.557 |

### Rows with HD95 above 20 mm, per 40 patient x fold rows

| variant | esophagus | heart | trachea | aorta | all |
|---|---|---|---|---|---|
| baseline | 3.0 | 1.0 | 3.7 | 4.3 | 12.0 |
| lcc_heart_trachea | 3.0 | 1.0 | 1.3 | 4.3 | 9.7 |
| seed_vote | 3.0 | 1.0 | 3.0 | 4.0 | 11.0 |
| seed_vote_lcc_heart_trachea | 3.0 | 1.0 | 2.0 | 4.0 | 10.0 |
| baseline_seed0 | 3.0 | 1.0 | 5.0 | 4.0 | 13.0 |
| avg | 3.0 | 1.0 | 3.0 | 4.0 | 11.0 |
| avg_lcc_heart_trachea | 3.0 | 1.0 | 2.0 | 4.0 | 10.0 |

### False-positive slices, per 40 patient x fold rows

| variant | esophagus | heart | trachea | aorta | all |
|---|---|---|---|---|---|
| baseline | 123.0 | 127.7 | 169.3 | 175.0 | 595.0 |
| lcc_heart_trachea | 123.0 | 102.7 | 132.0 | 175.0 | 532.7 |
| seed_vote | 99.0 | 104.0 | 159.0 | 156.0 | 518.0 |
| seed_vote_lcc_heart_trachea | 99.0 | 95.0 | 134.0 | 156.0 | 484.0 |
| baseline_seed0 | 118.0 | 117.0 | 146.0 | 172.0 | 553.0 |
| avg | 96.0 | 93.0 | 155.0 | 149.0 | 493.0 |
| avg_lcc_heart_trachea | 96.0 | 90.0 | 132.0 | 149.0 | 467.0 |

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
