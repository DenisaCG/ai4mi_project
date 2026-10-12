# Final model comparison: validation results

Validation only (the 40 labelled patients, 4-fold cross-validation × 3 seeds). Test patients (41–60) are not scored here: they have no ground truth.

Each value is the mean over the 12 runs (4 folds × 3 seeds), best checkpoint, unless marked otherwise.

All five columns are the ResEnc U-Net with deep supervision, strong geometric augmentation and 50 epochs (`full_cv4_resenc_ds_unet_dice_ce_strong_geometric_50ep`); the four others add a boundary loss and/or 2.5D input to it.

- **Bold** is the best value in each row. The number in parentheses is its margin over the next-best value.
- <u>Underline</u> is the second-best value.
- For Dice higher is better. For HD95 and ASSD lower is better.
- Post-processing: 3-seed softmax average, argmax, then largest connected component for heart and trachea only. With TTA, the softmax averaged over the seeds is the TTA-averaged softmax.
- Raw TTA: the `all` variant of `src/eval_tta.py` (rotation, scale and shift), averaged over the runs.

| Metric | ResEnc DS U-Net, 2D Dice+CE (final model) | 2D + boundary, ramped | 2D + boundary, fixed 0.08 | 2.5D + boundary, ramped | 2.5D + boundary, fixed 0.08 |
|---|---|---|---|---|---|
| **Raw, no TTA** | | | | | |
| Foreground Dice | 0.8926 | 0.8906 | 0.8915 | <u>0.8929</u> | **0.8930 (0.0001)** |
| Esophagus Dice | 0.8033 | 0.8031 | 0.8023 | **0.8059 (0.0018)** | <u>0.8041</u> |
| Heart Dice | 0.9413 | 0.9388 | 0.9415 | **0.9449 (0.0008)** | <u>0.9441</u> |
| Trachea Dice | 0.9023 | 0.9002 | 0.8995 | 0.9003 | **0.9035 (0.0012)** |
| Aorta Dice | **0.9235 (0.0010)** | 0.9203 | <u>0.9225</u> | 0.9206 | 0.9201 |
| Foreground HD95 (mm) | 9.76 | 13.05 | **9.19 (0.27)** | 9.53 | <u>9.46</u> |
| Foreground ASSD (mm) | 1.755 | 2.041 | <u>1.688</u> | **1.677 (0.011)** | 1.706 |
| **Post-processed, no TTA** | | | | | |
| Foreground Dice | **0.9001 (0.0002)** | 0.8986 | 0.8987 | 0.8996 | <u>0.8999</u> |
| Esophagus Dice | <u>0.8152</u> | 0.8143 | 0.8135 | **0.8165 (0.0013)** | 0.8149 |
| Heart Dice | 0.9461 | 0.9454 | 0.9453 | **0.9496 (0.0006)** | <u>0.9490</u> |
| Trachea Dice | 0.9097 | <u>0.9098</u> | 0.9083 | 0.9064 | **0.9099 (0.0001)** |
| Aorta Dice | **0.9294 (0.0016)** | 0.9249 | <u>0.9278</u> | 0.9260 | 0.9257 |
| Foreground HD95 (mm) | 7.53 | 7.95 | 7.90 | <u>7.52</u> | **7.42 (0.10)** |
| Foreground ASSD (mm) | 1.451 | 1.460 | 1.474 | <u>1.432</u> | **1.406 (0.026)** |
| **Raw, TTA** | | | | | |
| Foreground Dice | 0.8936 | 0.8920 | 0.8926 | **0.8939 (0.0000)** | **0.8939 (0.0000)** |
| Esophagus Dice | 0.8052 | 0.8056 | 0.8050 | **0.8083 (0.0022)** | <u>0.8061</u> |
| Heart Dice | 0.9419 | 0.9395 | 0.9420 | **0.9448 (0.0001)** | <u>0.9447</u> |
| Trachea Dice | <u>0.9029</u> | 0.9008 | 0.9001 | 0.9011 | **0.9038 (0.0009)** |
| Aorta Dice | **0.9245 (0.0011)** | 0.9219 | <u>0.9234</u> | 0.9215 | 0.9208 |
| Foreground HD95 (mm) | 9.30 | 12.10 | <u>9.01</u> | **8.64 (0.37)** | 9.25 |
| Foreground ASSD (mm) | 1.674 | 1.945 | <u>1.644</u> | **1.622 (0.022)** | 1.646 |
| **Post-processed, TTA** | | | | | |
| Foreground Dice | **0.9011 (0.0005)** | 0.8997 | 0.8997 | 0.9004 | <u>0.9006</u> |
| Esophagus Dice | <u>0.8170</u> | 0.8166 | 0.8164 | **0.8187 (0.0017)** | 0.8160 |
| Heart Dice | 0.9466 | 0.9458 | 0.9455 | **0.9495 (0.0000)** | **0.9495 (0.0000)** |
| Trachea Dice | **0.9106 (0.0000)** | 0.9098 | 0.9089 | 0.9068 | **0.9106 (0.0000)** |
| Aorta Dice | **0.9300 (0.0019)** | 0.9267 | <u>0.9281</u> | 0.9267 | 0.9262 |
| Foreground HD95 (mm) | <u>7.45</u> | 7.77 | 7.87 | 7.50 | **7.37 (0.08)** |
| Foreground ASSD (mm) | 1.428 | 1.444 | 1.461 | <u>1.425</u> | **1.396 (0.029)** |

A margin of (0.0000) means the two best values are equal at the printed precision; both are bold. Margins of 0.0001 are also at rounding level, and differences this small are likely within seed and fold variation (no significance test was run).

## Sources

- Raw, no TTA: `metrics/<experiment>_fold*/seed*/metrics_3d.csv` (the 2.5D, fixed 0.08 arm is in `metrics/augmentation_metrics/joint/`), pooled in the same way as `metrics/postprocessing/pp_*/summary.md`.
- Post-processed, no TTA: `metrics/postprocessing/pp_ctrl`, `pp_bnd`, `pp_w008`, `pp_25d`, `pp_25d_fixed08` (`summary.md`, row `avg_lcc_heart_trachea`).
- Raw TTA: `runs/<experiment>_fold*/seed*/eval/metrics_3d_tta.csv` (written by `src/eval_tta.py`), variant `all`, pooled per run (organ mean over patients, foreground = mean of the four organs), then averaged over the 12 runs. The `none` variant of the same files reproduces the Raw, no TTA column.
- Post-processed + TTA: `results/pp_tta_ctrl`, `results/pp_tta_bnd`, `results/pp_tta_w008`, `results/pp_tta_25d`, `results/pp_tta_25d_fixed08` (`rows.csv`, variant `avg_lcc_heart_trachea`), pooled with `tools.postprocess_eval.aggregate` like the `metrics/postprocessing/pp_*` tables. The `--save-probs` option of `src/eval_tta.py` (not yet on this branch) stores the TTA-averaged softmax and `tools.prob_postprocess score` post-processes it. As a check, running the same pipeline on the plain (`none`) predictions of the 2D Dice+CE arm reproduces `results/pp_ctrl` exactly (0.9001 / 7.53 / 1.451).

## Updating this table

1. When new numbers come in, replace them in the table, then recompute the bold, underline and margin in each row.
2. Keep the validation-only rule: do not add test-set numbers here.
