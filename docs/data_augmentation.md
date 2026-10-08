# Full40 SegTHOR data augmentation

## Goal and design

We test whether simple, scientifically motivated augmentation improves thoracic organ
segmentation on the Full40 **corrected** SegTHOR dataset. Each augmentation is tested
independently against two completed ENet references: [BASE](../configs/full_cv4_enet_ce.yaml)
(`full_cv4_enet_ce`) and [improved](../configs/full_cv4_enet_dice_ce.yaml)
(`full_cv4_enet_dice_ce`). This tests whether augmentation helps the original BASE
pipeline and whether it adds benefit after the preprocessing and loss improvements.

Each arm uses the reference's four patient folds (30 train / 10 validation patients
per fold) and three training seeds: **12 runs per arm**, or 72 intended augmentation
runs across six arms. Folds and seeds match the corresponding no-augmentation
reference. The single-transform arms are not combined; the combined (joint) recipe is described below. Each transform has `p: 0.5`,
an experimental design choice rather than a literature-derived optimum: approximately
half of training-slice presentations are augmented, with frequency held constant
across the three ablations. Validation and test data are not augmented.

## Augmentations

| Augmentation | BASE | Improved | Motivation |
|---|---|---|---|
| Rotation | `p=0.5`, −10° to +10°, `fill=0.0` | `p=0.5`, −10° to +10°, `fill=ct_window_low` | Mild orientation variation |
| Scaling | `p=0.5`, 0.9–1.1, `fill=0.0` | `p=0.5`, 0.9–1.1, `fill=ct_window_low` | Mild anatomical/body-scale variation |
| Gaussian noise | `p=0.5`, `sigma=0.00164` | `p=0.5`, `sigma=0.05` | Small intensity perturbations |

Geometric transforms move the CT and ground truth (GT) together, using bilinear
interpolation for CT and nearest-neighbor interpolation for GT. Areas outside the
source image become GT background. Gaussian noise changes only the CT, across the
whole image, and leaves the GT unchanged.

BASE normalizes each volume's own HU minimum and maximum to `[0, 1]`; geometric
fill is therefore `0.0`. Improved uses a CT-window z-score. Its `ct_window_low`
fill is resolved from each fold's `ct_norm_stats.json` as the normalized lower
clipping endpoint `(lo - mean) / std`. Improved `sigma=0.05` is a conservative
normalized perturbation corresponding to approximately 8.8–9.0 HU given the
Full40 fold statistics. BASE `sigma=0.00164` is approximately `8.9 / 5436`, where
5,436 HU is the median raw Full40 volume range in the existing build information.
Because BASE's per-volume ranges vary substantially, this fixed sigma is **not**
exactly 8.9 HU for every patient; it approximately matches the improved
perturbation at the median range. Neither sigma is a measurement of scanner noise.

## Scientific context

- [He et al. (2019)](https://ceur-ws.org/Vol-2349/SegTHOR2019_paper_2.pdf) used flipping and scaling from 0.6 to 1 in a SegTHOR multi-task method. This supports studying geometric augmentation; our 0.9–1.1 scale range is deliberately milder.
- [Vesal et al. (2019)](https://ceur-ws.org/Vol-2349/SegTHOR2019_paper_13.pdf) reported better SegTHOR performance for an augmented 2D model. Their paper does not establish our transform probabilities or parameter ranges.
- [Wang et al. (2019), EMSN](https://ceur-ws.org/Vol-2349/SegTHOR2019_paper_8.pdf) used registration-based augmentation on SegTHOR. It motivates a separately considered deformation direction, but does not justify our `p=0.5`, ±10° rotation, or scaling range. Registration-based deformation is not part of these six arms and has not been implemented for this experiment.

## Implementation and configurations

The existing path is **config → dataset → training-only `Augmented` wrapper →
registered augmentation factory** in [`src/augment.py`](../src/augment.py).
Augmentation is online: dataset length and optimizer steps per epoch do not
increase, and the same slice can be augmented on one presentation and unchanged
on another. The six configs are:

| BASE (`full_cv4_enet_ce`) | Improved (`full_cv4_enet_dice_ce`) |
|---|---|
| [`configs/full_cv4_enet_ce_rotation.yaml`](../configs/full_cv4_enet_ce_rotation.yaml) | [`configs/full_cv4_enet_dice_ce_rotation.yaml`](../configs/full_cv4_enet_dice_ce_rotation.yaml) |
| [`configs/full_cv4_enet_ce_scaling.yaml`](../configs/full_cv4_enet_ce_scaling.yaml) | [`configs/full_cv4_enet_dice_ce_scaling.yaml`](../configs/full_cv4_enet_dice_ce_scaling.yaml) |
| [`configs/full_cv4_enet_ce_gaussian_noise.yaml`](../configs/full_cv4_enet_ce_gaussian_noise.yaml) | [`configs/full_cv4_enet_dice_ce_gaussian_noise.yaml`](../configs/full_cv4_enet_dice_ce_gaussian_noise.yaml) |

## No-augmentation references

The completed references each contain 12 evaluated runs. Values below are
foreground 3D metric means ± standard deviations across runs from
[`metrics/cv_summary.md`](../metrics/cv_summary.md).

| Reference | Dice | HD95 | ASSD |
|---|---:|---:|---:|
| Full40 BASE | 0.707 ± 0.029 | 25.8 ± 5.4 mm | 4.91 ± 0.76 mm |
| Full40 improved | 0.835 ± 0.022 | 14.9 ± 2.2 mm | 2.65 ± 0.31 mm |

Improved is the no-augmentation control for the improved arms; BASE is the
control for the BASE arms. Augmentation results are in [`augmentation_selection.md`](augmentation_selection.md).

## Validation and submission status

Recorded prelaunch checks: automated suite **185 passed, 0 failed, 0 skipped**;
a separate CPU audit resolved six configs × four folds and constructed all 24
training augmentation datasets; all required BASE and improved fold caches were
accessible before launch. An earlier submission failed at `cv.job`'s cache
preflight, before training or augmentation, when cache paths were unavailable.

Status (2026-10-08). The single-transform screening was run on the improved ENet only,
seed 0 (4 runs per arm): rotation, scaling, Gaussian noise, shift, shear, gamma,
brightness/contrast and blur (the last five added in later configs). BASE
single-transform arms and seeds 1-2 of the single-transform arms were not run. An
earlier submission of all six arms (72 runs) was budget-cancelled except Improved +
Rotation fold 0 / seed 0. The final recipe combines rotation, scaling and shift and was
run on BASE ENet, improved ENet and the ResEnc + DS U-Net (12 runs each, all complete);
see [`augmentation_selection.md`](augmentation_selection.md) for the selection and results.
No-augmentation references are complete.

## Joint augmentation (rotation + scaling + shift together)

The final recipe applies **rotation, scaling and shift jointly** in one training run
and is chosen from the single-transform screening; see
[`augmentation_selection.md`](augmentation_selection.md) for how and why. Transforms run in
that order, each with `p: 0.5`. Rotation (-10 to +10 degrees), scaling (0.9 to 1.1) and shift
(up to 10% of image size) use the same parameters as the single-transform arms (BASE
`fill=0.0`; others `fill=ct_window_low`). Rotation, scaling and shift are separate
resampling steps, so a slice hit by several is interpolated more than once. Same folds and
seeds as the controls: 12 runs per model.

| Model | Control | Joint-augmentation config |
|---|---|---|
| BASE ENet | `full_cv4_enet_ce` | [`configs/full_cv4_enet_ce_augmented.yaml`](../configs/full_cv4_enet_ce_augmented.yaml) |
| Improved ENet | `full_cv4_enet_dice_ce` | [`configs/full_cv4_enet_dice_ce_augmented.yaml`](../configs/full_cv4_enet_dice_ce_augmented.yaml) |
| ResEnc + DS U-Net | `full_cv4_resenc_ds_unet_dice_ce` | [`configs/full_cv4_resenc_ds_unet_dice_ce_augmented.yaml`](../configs/full_cv4_resenc_ds_unet_dice_ce_augmented.yaml) |
