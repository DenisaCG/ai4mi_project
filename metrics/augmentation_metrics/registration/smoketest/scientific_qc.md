# One-pair registration: reproducibility and anatomical QC

**Scope.** Corrected Full40 SegTHOR, fold 0, Patient_26 (moving/source) to
Patient_19 (fixed/target), pairing seed 42. This report inspects only the CT/GT
volumes from Slurm jobs 27793262 and 27793349, the latter job's serialized
transform, the logs, and the current code/config. No registration was rerun.
The numerical measurements are in [`reports/reproducibility_qc.json`](reports/reproducibility_qc.json).

## Reproducibility

The fold split uses a local `random.Random(0)`; cyclic pairing uses a separate
`random.Random(42)`. These select patients and do not generate a registration
transform. Both rigid and B-spline registration construct independent
`ImageRegistrationMethod` objects with random 5% metric sampling and the explicit
SimpleITK seed **4242**. The geometric rigid initializer and B-spline initializer
have no RNG in this code. Regular-step gradient descent and LBFGSB have no
stochastic optimizer setting here. There is no NumPy RNG call in this path.
SimpleITK [documents](https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1ImageRegistrationMethod.html)
that a nonzero metric seed fixes the sampled points; it does not establish
bitwise reproducibility of a multithreaded metric/optimizer trajectory.
[ITK discussion](https://discourse.itk.org/t/difference-between-random-regular-and-none-sampling/5365)
specifically notes that mutual information with multiple threads may remain
non-reproducible. Four SimpleITK threads and `OMP_NUM_THREADS=4` were requested
in both jobs. Multithreaded metric accumulation is a plausible contributor,
**not a proven cause** of this pair's difference.

Both logs record the same Git commit `5a3b451`, Python executable, first pair,
source/target physical geometries, 6×6×6 B-spline mesh, and the same CPU job
script/resource request. They ran on different Genoa nodes (`tcn573` and
`tcn624`). The config contains no parameter overrides; the current default
registration parameters, recorded in the repeat's `reports/qc.json`, specify
the same deterministic geometric initialization and metric seed for both
stages. Between the two runs, the export and QC path changed; it executes
after optimization and does not alter the registration method. The original
log does **not** contain a full resolved parameter dump, input hashes,
SimpleITK binary hash, or optimizer trajectory, so exact runtime identity and
the mechanism of divergence cannot be proven retrospectively.

The original rigid metric was −0.071212, versus −0.071810 on repeat; both
reached the configured 150-iteration limit. The original B-spline stage met
function tolerance after 13 iterations (metric −0.077280), versus 26
iterations (metric −0.080712) on repeat. Divergence is already visible at
the rigid stage and cannot be attributed to HDF5 export.

The CTs share the same 512×512×190 target grid. Of 49,807,360 voxels,
35,263,607 (70.8%) have different integer HU values. Absolute HU difference:
mean **40.32**, median **5**, 75th **20**, 90th **84**, 95th **213**, 99th
**686**, 99.9th **1148**, maximum **3831**. Within the union of the two
foreground GT masks, mean absolute difference is **45.81 HU**. Differences
exceed 10 HU in 36.1% of the full image and 100 HU in 8.75%. The large
whole-image changed fraction includes many low-intensity/background voxels;
the organ measurements below establish that the variation is also anatomical.

| Organ | Original ↔ repeat Dice | Centroid shift (mm) | Volume difference, repeat − original |
|---|---:|---:|---:|
| Esophagus | 0.598 | 4.78 | +0.892 mL (+2.05%) |
| Heart | 0.927 | 4.12 | −0.653 mL (−0.074%) |
| Trachea | 0.595 | 6.29 | +0.732 mL (+3.15%) |
| Aorta | 0.777 | 4.47 | +4.723 mL (+2.23%) |

Thus the repeat differs meaningfully for the narrow esophagus and trachea,
even though organ volumes are broadly similar. The transform from job
27793349 applies to the **repeat** only; the original transform was not saved.

## Coordinate and visual audit

Source CT/GT share one NIfTI shape/affine; target CT/GT share another. The
original warped CT/GT, repeat warped CT/GT, and target CT/GT share the target
shape and affine. All NIfTI axis codes are **LPS**. The existing figure code
reads SimpleITK arrays as `(z,y,x)` and uses the same axial index and array
orientation for a CT and its GT. Its warped-CT panel overlays warped GT on
warped CT. Its third panel intentionally overlays warped GT on **target CT**
to inspect interpatient alignment; it does not show target GT. The old overview's
unregistered-source column is black because its selected target z planes lie
outside the source scan's physical z extent, not because of an axis bug.

The new comparison figures display NIfTI `(x,y,z)` planes transposed to
`(y,x)` for **every** CT and GT, with a common display origin. Each row selects
an organ-present native source slice, a warped target-grid slice, and a
target-GT-present slice; the fourth panel shows a left-warped/right-target
CT split on the warped slice's **same physical target plane**. Physical z and
slice indices are labeled. Source GT visibly follows source anatomy, and
warped GT follows warped-source anatomy in both runs. There is no detected
CT/GT axis swap, flip, or mask/CT slice mismatch.

- [`figures/anatomical_comparison_original.png`](figures/anatomical_comparison_original.png)
- [`figures/anatomical_comparison_repeat.png`](figures/anatomical_comparison_repeat.png)

The warped anatomy nevertheless does not align well with the target anatomy.
Target GT was **used only for this QC**, never for fitting. As context, the
original/repeat warped GT Dice against target GT is 0/0 for esophagus, 0.143/0.175
for heart, 0/0 for trachea, and 0.0185/0.0076 for aorta. Original/repeat
physical centroid distance to the target label is, respectively, 50.4/46.6 mm
(esophagus), 66.9/63.1 mm (heart), 40.3/36.5 mm (trachea), and 54.8/51.4 mm
(aorta). Interpatient shape and size differences are expected and Dice against
target GT is **not** an anatomical acceptance threshold. These large
displacements, zero overlap for two organs, and the visible heart/mediastinum
misalignment are reasons for human review before reuse.

## Repeat-transform deformation

The saved transform is a flattened rigid-plus-B-spline composite mapping
target physical points to source physical points as `rigid(bspline(point))`.
The earlier `reports/qc.json` samples a 126×126×95 grid at **4×4×5 mm**.
It reports B-spline displacement median 3.64 mm, maximum 12.74 mm; full
target-to-source displacement 373.69–433.94 mm, dominated by the large rigid
offset between scanner coordinate origins. The full-transform Jacobian
determinant ranges 0.846–1.225, with zero nonpositive **sampled** voxels.

As a separate local check, the investigation samples 512 repeat-label voxels
per organ and estimates the full transform's physical Jacobian by central
differences at ±1 mm. The B-spline displacement median/maximum is 7.64/9.15 mm
for esophagus, **9.53/11.10 mm for heart**, 4.77/7.61 mm for trachea, and
7.27/9.87 mm for aorta. The sampled Jacobian ranges are 0.862–1.100,
**0.846–1.035**, 0.924–1.069, and 0.857–1.124, respectively; none of these
2,048 sampled values is nonpositive. The heart region has the largest local
B-spline displacement among the sampled organs, but no abnormality can be
declared without a justified threshold and anatomical assessment. Neither
the coarse grid nor these sparse organ samples exclude folding between
points or establish an acceptable anatomical registration.

## Assessment

**Confirmed technical correctness:** the saved CT/GT pairs have matched
geometry and labels, the same transform is used for CT (linear) and GT
(nearest-neighbor), and the saved repeat transform round-trips. The display
audit found no CT/GT coordinate mix-up. This does **not** validate alignment
to the target patient.

**Unresolved reproducibility:** the fixed sampler seed did not produce the
same optimizer result across these four-thread CPU jobs. Multithreaded mutual
information is a plausible explanation; different nodes or unrecorded
runtime state cannot be excluded. No precise cause is established.

**Anatomical suitability:** the two registrations vary materially for small
organs, and both have poor target-organ overlap and conspicuous target-CT
misalignment. Organ-volume differences alone do not explain the centroid gaps
and poor overlap. A human should inspect the two comparison figures
and the full volumes. **Do not start bulk generation or training from this
pilot yet.** A future controlled single-thread reproducibility check and
review of rigid/global alignment would isolate causes; neither is performed
here. No production implementation defect is proven, so no production
correction is proposed as established fact.
