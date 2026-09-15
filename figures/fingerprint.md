# SegTHOR part1 dataset fingerprint (n=20 patients)

## Label mapping (verified against the raw data)

| label | organ |
|---|---|
| 0 | background |
| 1 | aorta |
| 2 | heart |
| 3 | trachea |
| 4 | esophagus |

**Label 1 is the aorta, not the esophagus.** Confirmed from sagittal/coronal reformats (aortic arch + descending aorta on the vertebral column) and from cross-sectional geometry (~31 mm median equivalent diameter, HU ~40-50). The course readme's class order (`esophagus heart trachea aorta`) does not match these GT files. The **esophagus** is the organ with no voxels here, and since every patient has one, that is an annotation gap -- not a structural absence.

## Train / validation split

- Validation patients (5): Patient_01, Patient_11, Patient_15, Patient_17, Patient_19
- Training patients (15): Patient_02, Patient_03, Patient_04, Patient_05, Patient_06, Patient_07, Patient_08, Patient_09, Patient_10, Patient_12, Patient_13, Patient_14, Patient_16, Patient_18, Patient_20
- The split is patient-disjoint (no slice-level leakage). **However, every statistic below is pooled over all 20 patients**, validation included, so any preprocessing derived from this fingerprint is fitted on held-out data.

## Shape & spacing

- Median shape (voxels, x,y,z): [512, 512, 175]
- Shape range: [512, 512, 147] to [512, 512, 284]
- Median in-plane spacing (mm): 0.9766 x 0.9766
- Slice spacing (mm): median 2.500, range 2.000-2.500

### Acquisition protocol groups (x x y x z spacing, mm)

| spacing | n | patients |
|---|---|---|
| 0.9766 x 0.9766 x 2.50 | 11 | P02, P03, P04, P06, P08, P09, P14, P16, P17, P18, P19 |
| 0.9766 x 0.9766 x 2.00 | 6 | P01, P05, P07, P10, P12, P13 |
| 1.2695 x 1.2695 x 2.50 | 1 | P11 |
| 0.8965 x 0.8965 x 2.50 | 1 | P15 |
| 1.3672 x 1.3672 x 2.50 | 1 | P20 |

No DICOM tags survive in these files (every NIfTI text field is zeroed), so scanner/protocol metadata cannot be recovered directly. Spacing groups and the stored integer type are the only protocol proxies available:

| stored dtype | n | patients |
|---|---|---|
| int32 | 17 | P02, P03, P04, P06, P07, P08, P09, P10, P11, P12, P14, P15, P16, P17, P18, P19, P20 |
| int16 | 3 | P01, P05, P13 |

## Class balance

| class | % voxels | % slices present |
|---|---|---|
| background (0) | 98.955 | 100.0 |
| aorta (1) | 0.241 | 63.5 |
| heart (2) | 0.766 | 22.5 |
| trachea (3) | 0.037 | 28.0 |
| esophagus (4) | 0.000 | 0.0 |

## HU intensity

- Whole-image HU range (exact, all voxels): [-1000.0, 31743.0], median -981.0
- Whole-image [0.5, 99.5] percentiles: [-1000.0, 487.0] (from a 1-in-37 voxel subsample)

### nnU-Net CT normalization parameters

nnU-Net clips to the [0.5, 99.5] percentiles of the **foreground** voxels pooled across the dataset and then z-scores with the foreground mean/std. Computed over all labeled voxels:

| statistic | value |
|---|---|
| clip lower (fg p0.5) | -992.0 HU |
| clip upper (fg p99.5) | 250.0 HU |
| mean (fg) | 12.4 HU |
| std (fg) | 183.2 HU |

Note the difference from the whole-image percentiles above ([-1000, 487]): clipping at the whole-image range is very nearly a no-op, because most voxels are air.

| organ | mean HU (fg) | HU [0.5, 99.5] pct range |
|---|---|---|
| aorta | 52.3 | [-642.0, 243.0] |
| heart | 39.9 | [-158.0, 257.0] |
| trachea | -811.5 | [-1000.0, 54.0] |
| esophagus | not labeled in this release | - |

## Organ bounding-box extent (mm, dx x dy x dz, median over patients where present)

| organ | median extent (mm) | # patients present |
|---|---|---|
| aorta | 75.2 x 111.0 x 274.5 | 20 |
| heart | 139.2 x 120.1 x 96.2 | 20 |
| trachea | 86.4 x 64.4 x 122.5 | 20 |
| esophagus | n/a | 0 |

Axis order is the array/voxel order (x, y, z) = (L, P, S); dz is the through-plane extent.

## Organ volume (cm3, per patient)

| organ | median | min | max | max/min | coefficient of variation |
|---|---|---|---|---|---|
| aorta | 255.7 | 127.0 | 522.3 | 4.11x | 32% |
| heart | 862.0 | 438.0 | 1829.3 | 4.18x | 33% |
| trachea | 38.8 | 27.6 | 72.9 | 2.64x | 28% |
| esophagus | 0.0 | 0.0 | 0.0 | - | - (not labeled) |

## Foreground (nonzero-CT) fraction of volume

- median 0.999, range [0.999, 1.000] -> cropping to nonzero region would remove almost nothing for this modality (unlike skull-stripped brain MRI, CT fills the frame). Note this measures `ct != 0`, and 0 HU is water, not air -- it detects zero-padding, not the body.

## HU artifact scan (12-bit reconstruction ceiling = 3071 HU)

Voxels **at** 3071 HU are censored by the scanner's 12-bit clamp (dense bone hitting the ceiling). Voxels **above** 3071 HU cannot come from a 12-bit reconstruction at all and mark metal implants with beam-hardening streaks. A single `> 2000 HU` threshold would merge the two.

| patient | at ceiling (3071) | above ceiling | max HU |
|---|---|---|---|
| Patient_02 | 0 | 4931 | 26613 |
| Patient_18 | 0 | 2946 | 15566 |
| Patient_19 | 1 | 2542 | 31743 |
| Patient_03 | 0 | 1031 | 25292 |
| Patient_15 | 0 | 854 | 17762 |
| Patient_17 | 0 | 852 | 14614 |
| Patient_16 | 1 | 442 | 11362 |
| Patient_11 | 0 | 407 | 8834 |
| Patient_04 | 0 | 6 | 4551 |
| Patient_01 | 158 | 0 | 3071 |
| Patient_05 | 1773 | 0 | 3071 |
| Patient_06 | 3871 | 0 | 3071 |
| Patient_07 | 0 | 0 | 3059 |
| Patient_08 | 2837 | 0 | 3071 |
| Patient_09 | 3705 | 0 | 3071 |
| Patient_10 | 391 | 0 | 3071 |
| Patient_12 | 162 | 0 | 3071 |
| Patient_13 | 607 | 0 | 3071 |
| Patient_14 | 0 | 0 | 1892 |
| Patient_20 | 334 | 0 | 3071 |

9 of 20 patients have voxels above the ceiling. Traced to source: these are metal implants with radiating beam-hardening streaks (e.g. Patient_19 z=180, a shoulder/neck slice). Locations of the most extreme voxel per patient:

- Patient_19: 31743 HU at (x=138, y=186, z=177)
- Patient_02: 26613 HU at (x=228, y=160, z=238)
- Patient_03: 25292 HU at (x=240, y=207, z=145)
- Patient_15: 17762 HU at (x=256, y=123, z=74)
- Patient_18: 15566 HU at (x=312, y=222, z=145)

## Per-patient outlier scan

Two criteria, unioned. PC1+PC2 of the 7 shape/spacing/volume features explain only 52% of the variance, so the projection alone cannot see a patient that is extreme on a single feature; the modified z-score catches those. The PCA threshold is median+3*MAD, not mean+2*std, so one extreme patient cannot inflate the threshold meant to catch it.

- Flagged by PC1-PC2 distance: Patient_15
- Flagged by per-feature modified z-score (|z| > 3.5):
  - Patient_05: z_shape z=+4.60
  - Patient_15: vol_aorta z=+3.94, vol_trachea z=+3.57
  - Patient_18: vol_heart z=+6.68
- **Flagged by either: Patient_05, Patient_15, Patient_18**
