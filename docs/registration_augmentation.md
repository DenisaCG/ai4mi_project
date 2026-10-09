# Full40 offline registration augmentation

Offline 3D deformable registration uses corrected Full40 CT/GT NIfTIs. It is separate
from the completed online rotation + scaling + shift experiment. Each of the four
existing CV folds keeps its 30 real training and 10 real validation patients.
Registration uses only the 30 training CTs
in that fold; the target GT is never read during transform estimation.

The final registration experiment compares [improved ENet](../configs/full_cv4_enet_dice_ce.yaml)
with [improved ENet + registration](../configs/full_cv4_enet_dice_ce_registration.yaml),
and [ResEnc + DS U-Net](../configs/full_cv4_resenc_ds_unet_dice_ce.yaml)
with [ResEnc + DS U-Net + registration](../configs/full_cv4_resenc_ds_unet_dice_ce_registration.yaml).
Each registration arm has four folds × three training seeds: 12 runs per model,
24 training/evaluation runs total. The historical BASE registration config is not an active arm.

## Current status

The focused registration suite passes 13/13 tests. One isolated fold-0 pair,
Patient_26 → Patient_19, has completed registration and CT/GT quality checks;
its volumes, transform, figures, and reports are under
[`metrics/augmentation_metrics/registration/smoketest/`](../metrics/augmentation_metrics/registration/smoketest/README.md).
The original and repeat results differ, and anatomical QC raises concern about
alignment to the target patient. The measurements, figure interpretations, and
remaining uncertainties are in the
[scientific QC report](../metrics/augmentation_metrics/registration/smoketest/scientific_qc.md).
This isolated pair is not part of the production registration dataset. Bulk
generation and registration-augmented training have not started; the pilot
requires further review before either proceeds.

## Pairing and outputs

For each fold, the 30 real training IDs are sorted, shuffled with
`random.Random(42)`, and paired cyclically. Every real training patient is used
once as source and once as target. Pairing does not depend on training seed 0/1/2.
Fold 0 begins with Patient_26 → Patient_19.

`src.registration` writes 30 synthetic CT and GT volumes per fold under
`data/registration_full40/fold<F>/train/Synthetic_f<F>_<NN>/`. The
`data/registration_full40` path is a symlink to
`/scratch-shared/$USER/ai4mi_project/registration_full40`, created by the
builder if needed. The original `data/segthor_train_full` tree is never changed.
The fold's `manifest.json` records the real split, all 30 planned pairs, the
parameters, SimpleITK version, output paths, status and runtime. Completed pairs
are reused when the builder is restarted.

Both models consume exactly those NIfTIs and, because they have identical improved
preprocessing, the same synthetic PNG slices under
`fold<F>/processed/<improved sliced-cache name>/train/`. Slicing uses the existing
real fold's median spacing, HU window/z-score and ROI size. The builder checks
that the statistics and ROI metadata list exactly the 30 real training patients.

The two registration configs retain the corresponding no-augmentation reference
`data.preprocess`, model, loss, optimizer, scheduler, training and evaluation
settings. Their real `data.root` caches and validation source GT stay unchanged.
The data loader appends synthetic PNGs **only to train**, and refuses to train until
all 30 pairs and that arm's synthetic slices are complete.

Both arms retain the original 25-epoch training protocol. Adding synthetic slices
increases training samples and optimizer updates per epoch relative to each
no-augmentation control. This is a limitation when attributing any difference
solely to the deformation, and the training schedule is not silently adjusted.

## Build

SimpleITK 2.5.6 is required in `ai4mi`. The registration parameters are centralized
in `src/registration.py:DEFAULT_PARAMETERS`; either arm can override them with
`data.registration.params`, but both configs must request identical parameters
to share a NIfTI source. The default is a physical-space, CT-only rigid plus
coarse cubic B-spline registration. The final transform maps target-grid points
to source coordinates for resampling. Source CT uses linear interpolation and
−1000 HU fill; source GT uses that same transform with nearest-neighbor
interpolation and background label 0.

From the repository root, submit **only the first fold-0 pair**:

```bash
sbatch --export=ALL,FOLD=0,LIMIT=1,VISUALIZE=1 jobsAndOutputs/augmentation/jobs/registration/registration.job
```

`registration.job` writes Slurm logs under `jobsAndOutputs/augmentation/outputs/registration/`
and reads `FOLD` (0–3), optional `LIMIT`, and optional
`VISUALIZE=1`; its config list contains the improved ENet and ResEnc + DS U-Net
registration arms. The command above uses the general builder and writes
`data/registration_full40/fold0/qc/Synthetic_f0_00.png` for visual inspection.
The fold remains incomplete and cannot be used for training.

If the pilot is accepted after further review, omit `LIMIT` to resume the
remaining pairs in fold 0, then submit the same job with `FOLD=1`, `2` and `3`.
No training or CV job is submitted by the registration builder. Once a fold is
complete,
`jobsAndOutputs/augmentation/jobs/registration/cv.job` runs the shared CV
training/evaluation mechanism for the two registration configs, with array
tasks 0–11 for each model (folds 0–3 × seeds 0–2). Submit it from the repository
root with `--export=ALL,CONFIG=configs/<registration-arm>.yaml`; its Slurm logs
go under `jobsAndOutputs/augmentation/outputs/registration/`. Small metrics
copies go under `metrics/augmentation_metrics/registration/`.

For example, after all four folds are complete:

```bash
sbatch --job-name=full_cv4_enet_dice_ce_registration \
  --export=ALL,CONFIG=configs/full_cv4_enet_dice_ce_registration.yaml \
  jobsAndOutputs/augmentation/jobs/registration/cv.job
```

Use the ResEnc registration YAML with the same wrapper for its 12 tasks.

The minimum per-pair automatic checks are finite CT values, matching CT/GT
geometry, valid GT labels and presence of all four organs. Visual inspection
must still assess plausible anatomy and label alignment; these checks do not
establish registration quality.
