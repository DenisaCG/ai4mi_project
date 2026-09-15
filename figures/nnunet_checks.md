# nnU-Net planner/fingerprint checks -- ported from source

## 1. Dataset integrity (verify_dataset_integrity.py)

- All image/label shapes match: **True**
- All image/label spacings match: **True**
- All image/label affines match: **True**
- Any unexpected label values found (outside {0..4}): **False**
- Label(s) missing in EVERY patient: **[4]** (= esophagus; note this is only detectable because we know 5 classes were expected -- nnU-Net's own verify_labels() does not check for missing expected labels, only unexpected ones)

## 2. crop_to_nonzero (cropping.py)

- Dataset median relative size after crop: **1.0000**
- Triggers mask-restricted normalization (< 0.75)? **False**

## 3. Anisotropy / target spacing (default_experiment_planner.py)

- median spacing (x,y,z): [0.9766, 0.9766, 2.5]
- worst_spacing_axis: 2
- has_aniso_spacing: **False**
- has_aniso_voxels: **False**
- cascade override applied: **False**
- final target spacing: [0.9766, 0.9766, 2.5]

## 4. Connected components per class (paper Sec. 2.5 rule)

| class | n patients present | always single (6-conn) | always single (26-conn) | max seen (6-conn) | max seen (26-conn) |
|---|---|---|---|---|---|
| aorta | 20 | False | True | 2 | 1 |
| heart | 20 | True | True | 1 | 1 |
| trachea | 20 | True | True | 1 | 1 |
| esophagus | 0 | None | None | 0 | 0 |

Aorta's 6-conn≠26-conn mismatch is a connectivity-definition artifact (arch curvature), not a real split -- see `aorta_component_case_study_Patient_02.png`.

## 5. Affine consistency, per patient

- Max |image_affine - seg_affine| across all 20 patients: **0.000000**

## 6. CT normalization parameters (CTNormalization.run(), pooled foreground)

- clip lower (p0.5): **-992.0 HU**
- clip upper (p99.5): **250.0 HU**
- mean: **12.3 HU**, std: **183.5 HU**
- n foreground voxels pooled: 10,219,707

