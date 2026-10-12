# Final model comparison: validation results

Validation only (the 40 labelled patients, 4-fold cross-validation × 3 seeds). Test patients (41–60) are not scored here: they have no ground truth.

Each value is the mean over the 12 runs (4 folds × 3 seeds), best checkpoint, unless marked otherwise.

- **Bold** is the best value in each row. The number in parentheses is its margin over the next-best value.
- <u>Underline</u> is the second-best value.
- For Dice higher is better. For HD95 and ASSD lower is better.
- Post-processing: 3-seed softmax average, argmax, then largest connected component for heart and trachea only.
- Raw TTA: the `all` variant of `src/eval_tta.py` (rotation, scale and shift), averaged over the runs.

Status key: **missing** = not started, **running** = job in progress.

| Metric | 2D Dice+CE (final model) | 2D + boundary, ramped | 2D + boundary, fixed 0.08 | 2.5D + boundary, ramped | 2.5D + boundary, fixed 0.08 |
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
| Foreground Dice | **0.8936** | missing | missing | missing | running |
| Esophagus Dice | **0.8050** | missing | missing | missing | running |
| Heart Dice | **0.9420** | missing | missing | missing | running |
| Trachea Dice | **0.9030** | missing | missing | missing | running |
| Aorta Dice | **0.9250** | missing | missing | missing | running |
| Foreground HD95 (mm) | **9.30** | missing | missing | missing | running |
| Foreground ASSD (mm) | **1.674** | missing | missing | missing | running |
| **Post-processed, TTA** | | | | | |
| Foreground Dice | missing | missing | missing | missing | missing |
| Esophagus Dice | missing | missing | missing | missing | missing |
| Heart Dice | missing | missing | missing | missing | missing |
| Trachea Dice | missing | missing | missing | missing | missing |
| Aorta Dice | missing | missing | missing | missing | missing |
| Foreground HD95 (mm) | missing | missing | missing | missing | missing |
| Foreground ASSD (mm) | missing | missing | missing | missing | missing |

Raw TTA has only the 2D Dice+CE column so far, so it has no comparison yet. Bold there marks the only value available.

## Sources

- Raw, no TTA: `runs/<experiment>_fold*/seed*/eval/metrics_3d.csv`, pooled in the same way as `results/pp_*/summary.md`.
- Post-processed, no TTA: `results/pp_ctrl`, `results/pp_bnd`, `results/pp_w008`, `results/pp_25d`, `results/pp_25d_fixed08` (`summary.md`, row `avg_lcc_heart_trachea`).
- Raw TTA, 2D Dice+CE: `python -m src.aggregate_tta --experiment full_cv4_resenc_ds_unet_dice_ce_strong_geometric_50ep` (row `all`).
- Raw TTA for the other arms: `jobsAndOutputs/pipeline/jobs/eval_tta.job`, one `sbatch` per experiment. The 2.5D fixed-weight run is job 27929100.
- Post-processed + TTA: not implemented yet. `src/eval_tta.py` needs to save the averaged probabilities that `tools/prob_postprocess.py` reads.

## Updating this table

1. When a TTA job finishes, run `python -m src.aggregate_tta --experiment <experiment>` and copy the `all` row into the Raw TTA cells.
2. Replace `running` and `missing` with the numbers, then recompute the bold, underline and margin in each row.
3. Keep the validation-only rule: do not add test-set numbers here.
