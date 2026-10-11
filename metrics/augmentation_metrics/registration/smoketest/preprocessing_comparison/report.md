# Native versus spacing-first registration: one-pair diagnostic

**Scope:** Corrected Full40 SegTHOR, fold-0 training pair Patient_26 -> Patient_19,
pairing seed 42. Slurm CPU job **27814090** completed all four runs on `tcn545`
with exit code 0. Each run used one SimpleITK/ITK/OMP thread and metric-sampling
seed 4242. Only registration estimation inputs differed: native HU CTs versus
the existing fold-0 median-spacing, pre-crop 3D HU CTs. The same physical
transform algorithm and settings were used in both conditions. Each initial
(geometric-center), fitted rigid and final rigid-plus-B-spline transform was
applied to the **same native source CT/GT** on the **same native target grid**.
Target GT was used for QC only, never to estimate transforms.

The exact input paths, SHA-256 hashes, physical geometry, optimizer settings,
termination conditions and per-run measurements are in each
`<condition>_<repeat>/reports/qc.json`. All source and cache hashes matched
before and after each run. The fold's 30 training IDs and existing normalization
and ROI metadata were checked before registration. The existing preprocessing
slicer was called once on each final CT/GT pair in a temporary directory
**inside this diagnostic directory**, and that temporary output was removed.
No production cache, dataset or registration manifest was written.

## Alignment

Each row below is identical across that condition's two repeats. The initial
stage is the SimpleITK geometric-center initializer, not an identity transform.
Body Dice uses a simple `CT > -500 HU` proxy on the fixed target grid. CT MAE is
computed on the **same target-body mask** in every row after clipping both CTs
to [-1000, 400] HU. Interpatient organ Dice is a QC indicator, not an
acceptance threshold.

| Estimation input | Stage | Body Dice | Target-body CT MAE (HU) | Heart Dice vs target GT | Heart centroid distance (mm) |
|---|---|---:|---:|---:|---:|
| Native | Initial | 0.484 | 340.2 | 0.156 | 66.0 |
| Native | Rigid | 0.452 | 385.9 | 0.144 | 66.6 |
| Native | Final | 0.446 | 398.7 | 0.178 | 63.1 |
| Resampled | Initial | 0.484 | 340.2 | 0.156 | 66.0 |
| Resampled | Rigid | 0.439 | 402.3 | 0.121 | 69.6 |
| Resampled | Final | 0.431 | 419.7 | 0.123 | 68.8 |

The spacing-first rigid stage is worse than native rigid on body overlap, CT
MAE and heart alignment. Both final stages have worse body overlap and CT MAE
than their own rigid stage. B-spline refinement partly improves the heart
Dice/centroid compared with rigid in the native condition, but does not solve
the large interpatient mismatch. The spacing-first final stage remains worse
than native final on all listed measures. Figures show persistent mismatch
between source and target anatomy; all GT contours are drawn on their own CT
in the same target physical slice. Inspect
[`native_1/figures/initial_rigid_final_organs.png`](native_1/figures/initial_rigid_final_organs.png)
and [`resampled_1/figures/initial_rigid_final_organs.png`](resampled_1/figures/initial_rigid_final_organs.png).

| Final-stage organ | Native Dice / centroid mm | Resampled Dice / centroid mm | Native volume change vs source | Resampled volume change vs source |
|---|---:|---:|---:|---:|
| Esophagus | 0 / 48.4 | 0 / 51.5 | +0.1% | -5.4% |
| Heart | 0.178 / 63.1 | 0.123 / 68.8 | +8.5% | +6.0% |
| Trachea | 0 / 36.6 | 0 / 47.1 | -1.9% | -11.9% |
| Aorta | 0.011 / 51.7 | 0 / 55.2 | +1.8% | -6.2% |

All four organs were present after warping, and the unchanged fold-0 ROI
retained **100% of resampled label voxels** in every run. Full stage-specific
CT/GT NIfTIs and verified transforms remain in each run directory.

## Repeatability and deformation

Within each condition, both runs produced identical initial, rigid and final
physical-point mappings at a 9 x 9 x 9 target-FOV sample; maximum sampled
distance was **0 mm**. Stage CT volumes have **0 HU mean absolute difference**
between repeats. Every stage has per-organ repeat Dice **1.0** and centroid
shift **0 mm**. These observations establish repeatability for this one pair,
one-thread setting and node; they do not establish that multithreading caused
the earlier nonreproducibility, nor generalize to other patients or nodes.

On the coarse 4 x 4 x 5 mm target grid, median / maximum B-spline displacement
was **4.36 / 13.63 mm** for native and **4.14 / 13.47 mm** for spacing-first.
The sampled full-transform Jacobian ranged **0.817–1.259** for native and
**0.786–1.287** for spacing-first, with no nonpositive sampled voxels. This
does **not** rule out folding between sample points. Full displacement is
roughly 365–447 mm across the conditions, dominated by the scanner-coordinate
offset and not interpretable as local anatomical deformation. Per-run
[`deformation.png`](native_1/figures/deformation.png) figures show the sampled
displacement and Jacobian fields.

## Runtime and resources

| Condition | Rigid stage (s), repeats 1 / 2 | B-spline stage (s), repeats 1 / 2 | End-to-end run wall time (s), repeats 1 / 2 |
|---|---:|---:|---:|
| Native | 23.20 / 23.15 | 366.93 / 364.72 | 570.42 / 564.18 |
| Resampled | 23.39 / 23.63 | 267.45 / 266.58 | 472.87 / 466.88 |

Per-process peak resident memory was **3.26–3.28 GiB**. The full sequential
Slurm job, including summary generation, elapsed **35 min 44 s**. The job
requested 2 CPUs and 12 GiB; Slurm accounted 24 CPUs due to the memory
request. Stdout/stderr are under `jobsAndOutputs/augmentation/outputs/registration/`.

## Recommendation

**Do not adopt spacing-first estimation** based on this pair. It was faster
but worsened measured rigid and final anatomical/CT alignment, and did not
improve repeatability because both conditions were already exactly repeatable
under one thread. **Do not start bulk generation with native registration
either:** even its better result has poor target-organ overlap and visible
misalignment. Keep native estimation as the reference for a separate,
CT-derived anatomical-initialization pilot, with rigid-stage review before
B-spline refinement. This experiment did not tune or test such an initializer.
