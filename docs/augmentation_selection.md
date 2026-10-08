# How the final augmentation setup was chosen

Dataset: Full40 corrected SegTHOR, 4 patient-level folds (30 train / 10 val). All metrics are
**3D** (Dice, HD95, ASSD on the original CT grid, from `summary.json` `eval`), foreground mean
over esophagus, heart, trachea, aorta.

## 1. Screening: one transform at a time

Each transform was added alone to the improved ENet (`full_cv4_enet_dice_ce`: Dice+CE,
median spacing, HU window + z-score, ROI crop), fold 0-3, **seed 0 only** (4 runs per arm).
Single-transform configs are `configs/full_cv4_enet_dice_ce_<transform>.yaml`. The control is
the same four seed-0 runs without augmentation (mean Dice 0.8376); the 12-run control mean
(0.835) is not comparable to a seed-0-only arm.

| Transform | Settings | Dice (fg) | Delta vs seed-0 control | HD95 mm | ASSD mm | Folds improved |
|---|---|---:|---:|---:|---:|---:|
| Rotation | p=0.5, -10 to +10 deg | 0.8501 | +0.0125 | 18.0 | 2.60 | 4/4 |
| Scaling | p=0.5, 0.9-1.1 | 0.8469 | +0.0094 | 12.9 | 2.36 | 4/4 |
| Shift | p=0.5, up to 10% | 0.8450 | +0.0074 | 15.6 | 2.45 | 4/4 |
| Shear | p=0.5, -5 to +5 deg | 0.8407 | +0.0031 | 13.7 | 2.47 | 3/4 |
| Gaussian noise | p=0.5, sigma 0.05 | 0.8389 | +0.0013 | 12.6 | 2.42 | 2/4 |
| Gaussian blur | p=0.2, sigma 0.5-1.0 | 0.8382 | +0.0006 | 13.8 | 2.46 | 2/4 |
| Brightness/contrast | p=0.15, 0.75-1.25 | 0.8345 | -0.0031 | 14.0 | 2.68 | 2/4 |
| Gamma | p=0.2, 0.7-1.5 | 0.8324 | -0.0051 | 13.2 | 2.65 | 0/4 |

## 2. Decision

Joint recipe: **rotation + scaling + shift**, applied together, each `p=0.5`.

- **Kept:** rotation, scaling and shift improved Dice on every fold (gain +0.007 to +0.013) and
  are physically plausible for CT (patient orientation, body size and table/field-of-view
  position vary between scans).
- **Dropped:** Gaussian noise (+0.001, 2/4 folds; no effect at the tested sigma), blur (+0.001),
  shear (small and inconsistent; CT geometry is rigid, so shear is not a realistic variation),
  gamma and brightness/contrast (negative or neutral; HU intensities are calibrated, so
  nonlinear remapping is not a natural CT variation).
- HD95 and ASSD do not follow Dice closely (rotation has the best Dice and the worst HD95); with
  4 runs and outlier-prone HD95 they were not used for selection.

Caveats: seed 0 only, and effect sizes are comparable to seed noise (earlier 12-run SDs were
about 0.02 Dice), so the ranking is indicative, not proven. The transforms were chosen on the
same validation folds that are reported later, which slightly inflates the joint result.

## 3. Final experiment: joint augmentation, 3 models x 4 folds x 3 seeds = 36 runs

Rotation, scaling and shift together (see [`data_augmentation.md`](data_augmentation.md)).
Each model is compared with its own no-augmentation control, paired by fold and seed.

| Model | Config | Control | Control Dice (fg) | Augmented Dice (fg) | Delta (paired) | Folds/seeds improved |
|---|---|---|---:|---:|---:|---:|
| BASE ENet | `full_cv4_enet_ce_augmented` | `full_cv4_enet_ce` | 0.707 | 0.741 | +0.033 | 12/12 |
| Improved ENet | `full_cv4_enet_dice_ce_augmented` | `full_cv4_enet_dice_ce` | 0.835 | 0.852 | +0.016 | 12/12 |
| ResEnc + DS U-Net | `full_cv4_resenc_ds_unet_dice_ce_augmented` | `full_cv4_resenc_ds_unet_dice_ce` | 0.867 | 0.880 | +0.012 | 12/12 |

Full results, mean +/- std over 12 runs (4 folds x 3 seeds), 3D metrics:

| Model | Dice fg | Esophagus | Heart | Trachea | Aorta | HD95 fg (mm) | ASSD fg (mm) | Runs done |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE ENet + joint aug | 0.741 +/- 0.026 | 0.516 | 0.895 | 0.758 | 0.794 | 22.2 +/- 6.7 | 4.11 +/- 0.91 | 12/12 |
| Improved ENet + joint aug | 0.852 +/- 0.018 | 0.713 | 0.929 | 0.876 | 0.888 | 13.1 +/- 3.2 | 2.33 +/- 0.37 | 12/12 |
| ResEnc + DS U-Net + joint aug | 0.880 +/- 0.013 | 0.784 | 0.933 | 0.890 | 0.911 | 11.8 +/- 2.0 | 2.13 +/- 0.36 | 12/12 |

Controls for the same table (Dice, per organ order as above, HD95, ASSD): BASE ENet 0.707 (0.469, 0.877, 0.743, 0.741), 25.8 mm, 4.91 mm; Improved ENet 0.835 (0.678, 0.922, 0.864, 0.877), 14.9 mm, 2.65 mm; ResEnc + DS U-Net 0.867 (0.763, 0.920, 0.886, 0.901), 13.8 mm, 2.42 mm.

Reading: Dice improves in all 12 fold/seed pairs for every model. HD95 improves in 8-10 of 12 pairs and ASSD in 10 of 12.
The esophagus, the hardest organ, gains the most (+0.021 to +0.047). The gain is largest for BASE ENet and
shrinks as the baseline improves (+0.033, +0.016, +0.012). The effect is comparable to the run-to-run std,
so the paired comparison, not the means alone, is the evidence. Transforms were selected on the same folds
(see Caveats), so these gains are slightly optimistic.

Source of the numbers: `metrics/<experiment>_fold*/seed*/summary.json` (`eval` block); aggregate with
`python -m src.aggregate --filter augmented`.

## 4. Does combining beat the best single transform?

Seed 0 only (4 folds), improved ENet, matched folds, 3D Dice (fg). Control: 0.8376.

| Recipe | Dice | Delta vs control | HD95 mm | ASSD mm | Per-fold delta |
|---|---:|---:|---:|---:|---|
| Rotation | 0.8501 | +0.0125 | 18.0 | 2.60 | +0.005 / +0.026 / +0.003 / +0.015 |
| Scaling | 0.8469 | +0.0094 | 12.9 | 2.36 | +0.012 / +0.012 / 0.000 / +0.013 |
| Shift | 0.8450 | +0.0074 | 15.6 | 2.45 | +0.010 / +0.010 / 0.000 / +0.009 |
| **Joint: rotation + scaling + shift** | **0.8541** | **+0.0165** | 13.3 | 2.29 | +0.015 / +0.029 / +0.005 / +0.017 |

The joint recipe is better than rotation alone on all four folds (by +0.010, +0.003, +0.001, +0.001; mean +0.004),
and has lower HD95 and ASSD. The gain over rotation is small (about 0.004 Dice) and rests on four seed-0 runs, so
treat "joint > best single transform" as suggestive. A 12-run rotation-only arm (seeds 1 and 2) would test it.
