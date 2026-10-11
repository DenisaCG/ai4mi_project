# Final training plan

Two models, each trained once on **all 40 labelled patients** (train + val of the CV work), then used to predict the **20 test patients** (Patient_41–60, CT only, no GT).

Test patients have no GT, so every model gets **two kinds of runs**:

| Model | All-data run (submission) | 4-fold CV run (held-out estimate, validates the all-data run) |
|---|---|---|
| 1. Midterm improved ENet (Dice(fg)+CE, no aug), 50 ep | `configs/final_enet_dice_ce.yaml` | `configs/full_cv4_enet_dice_ce_50ep.yaml` |
| 2. Best model: ResEnc DS U-Net + strong geometric, 50 ep | `configs/final_best_model.yaml` | `configs/full_cv4_resenc_ds_unet_dice_ce_strong_geometric_50ep.yaml` |

Seeds 0-2 for all four. The CV runs also give the folds x seeds data to re-check the seed-ensemble post-processing on these exact models.

**Why model 2 is ResEnc DS + strong geometric** (4 folds x 3 seeds, 25 epochs, fg Dice of the 2D val; "last" = final epoch, the number that matches an all-data run, "best" is picked on val and optimistic):

| Run | best | last |
|---|---|---|
| ENet, no aug (model 1 recipe) | 0.8353 | n/a |
| ENet strong geometric, 25 ep | 0.8617 | 0.8547 |
| ENet strong geometric, 50 ep | 0.8707 | 0.8650 |
| ENet extreme geometric, 25 ep (p0.9, rot 20, scale 0.7-1.3) | 0.8622 | 0.8561 |
| ResEnc DS augmented (rss p0.5), 25 ep | 0.8795 | 0.8663 |
| **ResEnc DS strong geometric, 25 ep** | **0.8862** | **0.8777** |

Takeaways: doubling ENet to 50 epochs gave +0.009-0.010; extreme geometric is no better than strong; ResEnc DS beats ENet at equal augmentation and epochs. The last epoch is 0.006-0.009 below the val-picked best, so quote last-epoch numbers for the all-data runs. Plain U-Net with cosine decay was worse than constant LR at 25 epochs (0.8435 vs 0.8515); 50 epochs for ResEnc is not yet tested, that is what the 50 ep CV run measures.

## Compute (A100 = 128 SBU/hr, billed on requested time; estimates from CV epoch times, 40 vs 30 training patients is x1.33)
| Block | Runs | Walltime each | SBU |
|---|---|---|---|
| ENet CV, 50 ep | 12 | 1 h | 1536 |
| ResEnc CV, 50 ep | 12 | 1.25 h | 1920 |
| ENet all-data, 50 ep | 3 | 1 h | 384 |
| ResEnc all-data, 50 ep | 3 | 1.5 h | 576 |
| Smoke tests, slicing, prediction/post-processing | | | ~150 |
| **Total** | 30 | | **~4600** |

Order: CV first (needs no code changes, can be submitted now) while the no-validation code is written; the all-data runs follow.

## Data
- Test CTs: `data/segthor_train_full/test/Patient_41..60.nii.gz` (from `test.zip`, kept next to them). Same layout `slice_segthor.py` already expects (`<source>/test/<id>.nii.gz`), original/raw, read-only.
- Labelled: `data/segthor_train_full/train/Patient_01..40` (corrected GT, labels 0–4).

## Decisions to make
1. **Which is the best model?** Decided: ResEnc DS + strong geometric (table above). The 50 ep CV run confirms the epoch count.
2. **Epochs / checkpoint.** 50 epochs, last epoch (no val set, so no `best.pt`). Constant LR makes the last epoch noisy (0.006-0.009 below the best); if the 50 ep CV shows a large last-vs-best gap, try averaging/EMA of the last epochs or a late LR decay, tested in CV first.
3. **Seeds: 3 per model (0, 1, 2), 6 runs total**, same as the CV work. Runs land in `runs/<experiment>/seed<k>/`, one config per model, seed passed with `--set seed=k`.
4. **Post-processing (decided from `~/pp_inventory_igardner1`, 4 folds, base ResEnc DS U-Net dice_ce without augmentation, fg Dice of the 4 organs; ensembles vs the mean of the 3 single seeds):**

   | Method | Dice | HD95 mm |
   |---|---|---|
   | 3-seed softmax average (`avg`) | +0.0093 | -2.5 |
   | **`avg` + largest connected component for heart and trachea only (`avg_lcc_heart_trachea`)** | **+0.0095** | **-3.5** |
   | 3-seed per-organ vote + same LCC | +0.0103 | -2.8 |
   | LCC heart+trachea on a single model | +0.0003 | -1.8 |
   | LCC on all organs / esophagus | -0.012 / -0.010 | +4.5 / +5.4 |
   | CRF (best variant) | -0.005 | -1.3 |
   | slice gating with GT (`gate`) | +0.015 | -5.2, **uses the ground truth, not usable on test** |

   Use: **average the softmax of the 3 seeds, argmax, then largest connected component for heart and trachea only.** The gain is almost all from the seed ensemble; the LCC adds ~0.000-0.001 Dice but cuts HD95 by 1 mm. Never LCC the esophagus (it is often legitimately split) or aorta, no CRF, no size/z filters (all ~0 or negative). The vote variant is 0.0008 Dice better but within noise, so take the simpler average. Caveat: measured on ResEnc DS without augmentation; re-check `avg_lcc_heart_trachea` on the chosen best model's CV runs (ENet and the augmented models) before the final submission. Tools: `~/ai4mi_postprocess/tools/` (`prob_postprocess.py`, `seed_ensemble.py`), branch `postprocess-oracle`; they need saved softmax files per seed, so the final predict step must save probabilities, not just labels. Apply the same rule to both models.

## Code changes needed (not done yet)
1. `slice_segthor.py` / `src/splits.py` / `src/folds.py`: accept `retains: 0` = all patients in `train/`, empty (or no) `val/`; also slice the test split (`get_splits` already lists test ids, but `main` only loops `train`/`val`; `slice_patient(test_mode=True)` exists). Normalization stats, target spacing and ROI size T come from all 40 patients; every test patient must be inside the ROI window (`roi_centre` uses image only), report retention warnings are impossible without GT so check window position visually for a few.
2. `src/engine.py`: skip the val loop/checkpoint selection when there is no val split; always save `last.pt` (+ whole-net pickle) at the final epoch. `select_metric` has no val to read.
3. `src/evaluate.py`: forward-pass timing on the test set (section above) and a test-only path (already predicts and stitches `test` when `test_source_pattern` is set); skip val scoring.
4. Config validation: allow `source_pattern: null` with no val.
5. Tests for the above, plus a smoke run on `gpu_mig` (`--smoke`) before the real jobs.

## Inference time (test set only)
Time only the inference itself: the network forward pass over each test patient's slices, for the 20 test CTs. Not preprocessing, stitching or post-processing.
- Per patient: `time.perf_counter()` around the forward pass with `torch.cuda.synchronize()` before each read, one untimed warm-up patient, on a `gpu_a100` job. Write seconds per patient to `eval/inference_time.csv`; report the mean over the 20 patients.
- Compare the two final models on the same 20 test CTs, same GPU, same job: upgraded ENet (`final_enet_dice_ce`) vs the final U-Net (`final_best_model`, ResEnc DS). Report mean seconds per patient for each side by side, plus parameter count. Single seed (seed 0) per model. `src/evaluate.py` has no timing today, so this is added with the no-validation changes.

## Run order
1. Implement and smoke-test the no-val path (`gpu_mig`).
2. Submit model 1, seeds 0-2 as an array (`--array=0-2`, `gpu_a100`, each ~4/3 of one CV run's cost).
3. When the best model is decided, finalise `final_best_model.yaml`, smoke-test, submit seeds 0-2 the same way.
4. Stitch test predictions to NIfTI on the original grid (`src.evaluate`) and check shape/orientation/labels {0..4} against a test CT before submitting.

## Sanity checks before submitting
- Predictions for all 20 test ids, label set within {0..4}, header/affine equal to the input CT.
- Visual overlay for a handful of test patients (no GT to score against).
- Log which SEGTHOR version, config, commit hash and seed with each model.
