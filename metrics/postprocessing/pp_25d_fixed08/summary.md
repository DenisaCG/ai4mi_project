# Probability-based post-processing: full_cv4_resenc_ds_unet_dice_ce_25d_c1_strong_geometric_50ep_boundary_w008, 3 seeds, 4 folds, 40 patients

Re-predicted argmax vs the stored volumes: 544 voxels differ in total, at most 13 in one patient volume (GPU rounding; 0 expected otherwise).
Reference rows (baseline, lcc_heart_trachea, seed_vote*) come from results/seed_ensemble/rows.csv; baseline_seed0 is seed 0 alone.
avg*: seeds' probabilities averaged, one result per fold. CRF settings (sxy, compat; sxy, srgb, compat): a gauss (3, 3) bilateral (20, 10, 5); b gauss (3, 3) bilateral (50, 5, 5); c gauss (5, 5) bilateral (40, 10, 10); 5 iterations.

### fg Dice and per-organ Dice

| variant | fg | fg vs baseline | esophagus (vs baseline) | heart (vs baseline) | trachea (vs baseline) | aorta (vs baseline) | fold 0 fg | fold 1 fg | fold 2 fg | fold 3 fg |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.8930 | +0.0000 | 0.8041 (+0.0000) | 0.9441 (+0.0000) | 0.9035 (+0.0000) | 0.9201 (+0.0000) | 0.9080 | 0.8759 | 0.8938 | 0.8942 |
| lcc_heart_trachea | 0.8933 | +0.0004 | 0.8041 (+0.0000) | 0.9449 (+0.0008) | 0.9042 (+0.0007) | 0.9201 (+0.0000) | 0.9087 | 0.8746 | 0.8955 | 0.8946 |
| seed_vote | 0.8991 | +0.0061 | 0.8144 (+0.0104) | 0.9483 (+0.0042) | 0.9081 (+0.0046) | 0.9254 (+0.0054) | 0.9136 | 0.8839 | 0.8986 | 0.9002 |
| seed_vote_lcc_heart_trachea | 0.8995 | +0.0066 | 0.8144 (+0.0104) | 0.9487 (+0.0045) | 0.9095 (+0.0060) | 0.9254 (+0.0054) | 0.9137 | 0.8841 | 0.9000 | 0.9003 |
| baseline_seed0 | 0.8944 | +0.0015 | 0.8045 (+0.0004) | 0.9463 (+0.0021) | 0.9023 (-0.0012) | 0.9246 (+0.0045) | 0.9104 | 0.8796 | 0.8944 | 0.8934 |
| avg | 0.8995 | +0.0065 | 0.8149 (+0.0108) | 0.9487 (+0.0046) | 0.9086 (+0.0050) | 0.9257 (+0.0056) | 0.9140 | 0.8842 | 0.8990 | 0.9006 |
| avg_lcc_heart_trachea | 0.8999 | +0.0069 | 0.8149 (+0.0108) | 0.9490 (+0.0048) | 0.9099 (+0.0064) | 0.9257 (+0.0056) | 0.9141 | 0.8842 | 0.9003 | 0.9008 |

### fg HD95 and per-organ HD95 (mm)

| variant | fg | fg vs baseline | esophagus (vs baseline) | heart (vs baseline) | trachea (vs baseline) | aorta (vs baseline) | fold 0 fg | fold 1 fg | fold 2 fg | fold 3 fg |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 9.46 | +0.00 | 8.44 (+0.00) | 11.98 (+0.00) | 8.06 (+0.00) | 9.36 (+0.00) | 9.60 | 9.84 | 8.62 | 9.77 |
| lcc_heart_trachea | 8.23 | -1.23 | 8.44 (+0.00) | 7.10 (-4.87) | 8.03 (-0.04) | 9.36 (+0.00) | 6.13 | 9.88 | 8.26 | 8.66 |
| seed_vote | 7.50 | -1.96 | 7.36 (-1.07) | 7.90 (-4.08) | 6.91 (-1.15) | 7.83 (-1.53) | 5.59 | 9.05 | 7.69 | 7.68 |
| seed_vote_lcc_heart_trachea | 7.20 | -2.26 | 7.36 (-1.07) | 6.57 (-5.41) | 7.03 (-1.03) | 7.83 (-1.53) | 5.42 | 8.15 | 7.54 | 7.68 |
| baseline_seed0 | 7.88 | -1.58 | 8.43 (-0.01) | 8.18 (-3.79) | 7.99 (-0.07) | 6.92 (-2.43) | 6.37 | 9.24 | 8.08 | 7.84 |
| avg | 7.48 | -1.97 | 7.60 (-0.84) | 6.93 (-5.05) | 6.90 (-1.16) | 8.51 (-0.85) | 5.53 | 8.45 | 8.29 | 7.67 |
| avg_lcc_heart_trachea | 7.42 | -2.04 | 7.60 (-0.84) | 6.54 (-5.44) | 7.03 (-1.03) | 8.51 (-0.85) | 5.42 | 8.43 | 8.15 | 7.67 |

### fg ASSD and per-organ ASSD (mm)

| variant | fg | fg vs baseline | esophagus (vs baseline) | heart (vs baseline) | trachea (vs baseline) | aorta (vs baseline) | fold 0 fg | fold 1 fg | fold 2 fg | fold 3 fg |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1.706 | +0.000 | 1.685 (+0.000) | 2.475 (+0.000) | 1.182 (+0.000) | 1.482 (+0.000) | 1.720 | 1.812 | 1.655 | 1.638 |
| lcc_heart_trachea | 1.568 | -0.139 | 1.685 (+0.000) | 2.028 (-0.448) | 1.075 (-0.106) | 1.482 (+0.000) | 1.316 | 1.814 | 1.583 | 1.558 |
| seed_vote | 1.445 | -0.261 | 1.467 (-0.218) | 2.048 (-0.427) | 0.994 (-0.188) | 1.269 (-0.213) | 1.288 | 1.599 | 1.505 | 1.386 |
| seed_vote_lcc_heart_trachea | 1.397 | -0.309 | 1.467 (-0.218) | 1.883 (-0.593) | 0.969 (-0.213) | 1.269 (-0.213) | 1.180 | 1.562 | 1.467 | 1.379 |
| baseline_seed0 | 1.546 | -0.160 | 1.686 (+0.000) | 2.088 (-0.387) | 1.164 (-0.017) | 1.247 (-0.235) | 1.351 | 1.658 | 1.635 | 1.542 |
| avg | 1.440 | -0.266 | 1.497 (-0.188) | 1.993 (-0.482) | 0.987 (-0.195) | 1.282 (-0.200) | 1.270 | 1.592 | 1.518 | 1.379 |
| avg_lcc_heart_trachea | 1.406 | -0.300 | 1.497 (-0.188) | 1.879 (-0.596) | 0.967 (-0.215) | 1.282 (-0.200) | 1.178 | 1.591 | 1.484 | 1.373 |

### Rows with HD95 above 20 mm, per 40 patient x fold rows

| variant | esophagus | heart | trachea | aorta | all |
|---|---|---|---|---|---|
| baseline | 3.3 | 2.3 | 2.7 | 3.3 | 11.7 |
| lcc_heart_trachea | 3.3 | 0.3 | 2.0 | 3.3 | 9.0 |
| seed_vote | 2.0 | 1.0 | 0.0 | 4.0 | 7.0 |
| seed_vote_lcc_heart_trachea | 2.0 | 0.0 | 0.0 | 4.0 | 6.0 |
| baseline_seed0 | 4.0 | 1.0 | 4.0 | 0.0 | 9.0 |
| avg | 3.0 | 0.0 | 0.0 | 5.0 | 8.0 |
| avg_lcc_heart_trachea | 3.0 | 0.0 | 0.0 | 5.0 | 8.0 |

### False-positive slices, per 40 patient x fold rows

| variant | esophagus | heart | trachea | aorta | all |
|---|---|---|---|---|---|
| baseline | 115.3 | 147.3 | 156.0 | 172.0 | 590.7 |
| lcc_heart_trachea | 115.3 | 93.0 | 129.3 | 172.0 | 509.7 |
| seed_vote | 87.0 | 122.0 | 129.0 | 145.0 | 483.0 |
| seed_vote_lcc_heart_trachea | 87.0 | 87.0 | 120.0 | 145.0 | 439.0 |
| baseline_seed0 | 113.0 | 130.0 | 151.0 | 153.0 | 547.0 |
| avg | 84.0 | 97.0 | 128.0 | 142.0 | 451.0 |
| avg_lcc_heart_trachea | 84.0 | 81.0 | 119.0 | 142.0 | 426.0 |

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
