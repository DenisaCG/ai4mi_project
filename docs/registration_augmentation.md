# Full40 offline registration augmentation

This arm uses corrected Full40 CT/GT NIfTIs. It is separate from the online rotation,
scaling and Gaussian noise arms. Each of the four existing CV folds keeps its 30 real
training and 10 real validation patients. Registration uses only the 30 training CTs
in that fold; the target GT is never read during transform estimation.

## Pairing and outputs

For each fold, the 30 real training IDs are sorted, shuffled with
`random.Random(42)`, and paired cyclically. Every real training patient is used
once as source and once as target. Pairing does not depend on training seed 0/1/2.
Fold 0 begins with Patient_26 → Patient_19.

`src.registration` writes synthetic CT and GT NIfTIs under
`data/registration_full40/fold<F>/train/Synthetic_f<F>_<NN>/`. The
`data/registration_full40` path is a symlink to
`/scratch-shared/$USER/ai4mi_project/registration_full40`, created by the
builder if needed. The original `data/segthor_train_full` tree is never changed.
The fold's `manifest.json` records the real split, all 30 planned pairs, the
parameters, SimpleITK version, output paths, status and runtime. Completed pairs
are reused when the builder is restarted.

Both arms consume exactly those NIfTIs:

- BASE: synthetic slices under `fold<F>/processed/<BASE sliced-cache name>/train/`,
  using the reference BASE per-volume min–max and resize.
- Improved: synthetic slices under
  `fold<F>/processed/<improved sliced-cache name>/train/`, using the existing
  real fold's median spacing, HU window/z-score and ROI size. The builder checks
  that the statistics and ROI metadata list exactly the 30 real training patients.

The two registration configs retain the corresponding no-augmentation reference
`data.preprocess`, model, loss, optimizer, scheduler, training and evaluation
settings. Their real `data.root` caches and validation source GT stay unchanged.
The data loader appends synthetic PNGs **only to train**, and refuses to train until
all 30 pairs and that arm's synthetic slices are complete.

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
sbatch --export=ALL,FOLD=0,LIMIT=1,VISUALIZE=1 jobsAndOutputs/pipeline/jobs/registration.job
```

This uses the general builder and writes
`data/registration_full40/fold0/qc/Synthetic_f0_00.png` for visual inspection.
The fold remains incomplete and cannot be used for training.

After reviewing the pilot, omit `LIMIT` to resume the remaining pairs in fold 0,
then submit the same job with `FOLD=1`, `2` and `3`. No training or CV job is
submitted by the registration builder. The generic `cv.job` is the eventual
training mechanism for the two completed registration configs.

The minimum per-pair automatic checks are finite CT values, matching CT/GT
geometry, valid GT labels and presence of all four organs. Visual inspection
must still assess plausible anatomy and label alignment; these checks do not
establish registration quality.
