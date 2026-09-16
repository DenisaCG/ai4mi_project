# nnU-Net planner/fingerprint checks -- ported from source

## 1. Dataset integrity (verify_dataset_integrity.py)

- All image/label shapes match: **True**
- All image/label spacings match: **True**
- All image/label affines match: **True**
- Any unexpected label values found (outside {0..4}): **False**
- Label(s) missing in EVERY patient: **[4]** (= aorta; verify_labels() only reports unexpected labels, so this is checked separately against the 5 expected classes)
- Coordinate orientation consistent across all patients: **True** (axis codes seen: ['LPS'])
- Any voxel above the 12-bit ceiling (>3071 HU): **True**

## 2. crop_to_nonzero (cropping.py)

- Relative size after crop, per patient: min **1.0000**, max **1.0000**
- Triggers mask-restricted normalization (median < 0.75)? **False**
- CT air is about -1000 HU, not 0, so the nonzero mask covers the whole volume; see tissue_footprint.png and z_coverage.png for where tissue and organs actually are.

## 3. Anisotropy / target spacing (default_experiment_planner.py)

- median spacing (x,y,z): [0.9766, 0.9766, 2.5]
- worst_spacing_axis: 2
- has_aniso_spacing: **False**
- has_aniso_voxels: **False**
- cascade override applied: **False**
- final target spacing: [0.9766, 0.9766, 2.5]

## 4. Connected components per class (paper Sec. 2.5 rule)

| class | n patients present | always single (6-conn) | always single (18-conn) | always single (26-conn) | max seen (6-conn) | max seen (18-conn) | max seen (26-conn) |
|---|---|---|---|---|---|---|---|
| label 1 | 20 | False | True | True | 2 | 1 | 1 |
| label 2 | 20 | True | True | True | 1 | 1 | 1 |
| label 3 | 20 | True | True | True | 1 | 1 | 1 |
| label 4 | 0 | None | None | None | 0 | 0 | 0 |

## 5. Affine consistency, per patient

- Max |image_affine - seg_affine| across all 20 patients: **0.000000**

## 6. Organ-pair adjacency (fraction of patients where organ A touches organ B)

| touches -> | label 1 | label 2 | label 3 | label 4 |
|---|---|---|---|---|
| esophagus | - | 100% | 100% | 0% |
| heart | 100% | - | 20% | 0% |
| trachea | 100% | 20% | - | 0% |
| aorta | 0% | 0% | 0% | - |

## 7. CT normalization parameters (CTNormalization.run(), pooled foreground)

- clip lower (p0.5): **-992.0 HU**
- clip upper (p99.5): **250.0 HU**
- mean: **12.3 HU**, std: **183.5 HU**
- n foreground voxels pooled: 10,219,707

