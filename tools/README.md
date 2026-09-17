# Reproducing the figures

Helper scripts for dataset exploration and for turning training output into
shareable PNGs. All are additions to the course codebase; nothing in the original
scripts is modified.

- `explore_data.py` — dataset fingerprint + EDA/QC over the raw NIfTI volumes.
  Writes `fingerprint.json`, `fingerprint.md`, `organ_volumes_cm3.csv` and 9 PNGs.
  Needs no training run and no `results/`.
- `plot_style.py` — shared matplotlib style (titles, subtitles, footnotes, palette)
  used by `explore_data.py`. Not a standalone script.
- `save_viewer.py` — runs `viewer/viewer.py` headlessly and saves its figure to PNG
  (the stock viewer opens a GUI window and has no save option).
- `render_figures.py` — orthogonal-plane and 3D-surface renders from a stitched volume.

All commands are run **from the repo root** with the environment active:

```bash
cd ~/ProjectsMSc/ai4mi_project
conda activate ai4mi
```

## 0. Dataset fingerprint (no training needed)

```bash
python tools/explore_data.py --data-dir data/segthor_part1/train --out-dir figures
```

Runs in about a minute over the 20 part1 patients and needs only
`data/segthor_part1/train`; the defaults already point there, so bare
`python tools/explore_data.py` works from the repo root. `--val-gt-dir`
(default `data/SEGTHOR/val/gt`) is used only to label which patients are
held out — it is skipped silently if that directory does not exist.

**Label mapping.** In `data/segthor_part1`, GT labels are `1=esophagus,
2=heart, 3=trachea, 4=aorta` — matching the top-level readme's class order.
**Aorta (4)** is the class with no voxels: the professor confirmed this is an
intentional omission for the course dataset. See
`dataset_analysis/utils.py:CLASSES` for the canonical mapping.

Requires matplotlib >= 3.9 (`tick_labels=` boxplot kwarg).

## 1. Train (skip if `results/` is already populated)

```bash
python -O main.py --dataset TOY2    --mode full --epochs 25 --dest results/toy2/ce    --gpu
python -O main.py --dataset SEGTHOR --mode full --epochs 15 --dest results/segthor/ce --gpu
```

`--gpu` uses CUDA if present, else Apple Silicon (MPS), else falls back to CPU.
On an M1 Air, SEGTHOR runs about 3 min/epoch on MPS versus ~17 min on CPU.

## 2. 2D comparison figures

```bash
mkdir -p figures

python tools/save_viewer.py --out figures/toy2_viewer.png \
    --img_source data/TOY2/val/img \
    data/TOY2/val/gt results/toy2/ce/iter000/val results/toy2/ce/iter005/val results/toy2/ce/best_epoch/val \
    --show_img -C 256 --no_contour -n 3

python tools/save_viewer.py --out figures/segthor_viewer.png \
    --img_source data/SEGTHOR/val/img \
    data/SEGTHOR/val/gt results/segthor/ce/iter000/val results/segthor/ce/best_epoch/val \
    -n 3 -C 5 --remap "{63: 1, 126: 2, 189: 3, 252: 4}" --legend \
    --class_names background c1 c2 c3 c4
```

The referenced `iterNNN` folders must exist: `iter005` needs at least 6 epochs.

## 3. Stitch 2D predictions back into 3D volumes

```bash
python stitch.py --data_folder results/segthor/ce/best_epoch/val \
    --dest_folder volumes/segthor/ce \
    --num_classes 255 --grp_regex "(Patient_\d\d)_\d\d\d\d" \
    --source_scan_pattern "data/segthor_part1/train/{id_}/GT.nii.gz"
```

Two deviations from the README are required here:

1. `--source_scan_pattern` must point at `data/segthor_part1/`, not `data/segthor_train/`.
2. `stitch.py:76` asserts every one of the 5 classes is present:
   `assert set(np.unique(res_arr)) == set(range(5))`. The `segthor_part1` ground truth
   only contains labels 0-3, so this can never hold. Relax it to a subset check:
   `assert set(np.unique(res_arr)) <= set(range(5))`.

## 4. 3D figures

```bash
python tools/render_figures.py --patient Patient_01 \
    --gt_dir data/segthor_part1/train \
    --pred_dir volumes/segthor/ce \
    --out figures
```

Writes `figures/Patient_01_planes.png` and `figures/Patient_01_surface.png`.
Only labels actually present are drawn, so no phantom organ appears in the legend.
Pass `--class_names background <c1> <c2> <c3> <c4>` once the label mapping is confirmed,
and `--skip_surface` for the faster planes-only render.

## 5. Metric curves

```bash
for m in dice_val loss_val dice_tra loss_tra; do
    python plot.py --metric_file results/segthor/ce/$m.npy \
        --dest results/segthor/ce/$m.png --headless
done
```

## Interactive 3D

The PNGs above are static. For a rotatable view, open the stitched volume next to its
ground truth in [3D Slicer](https://www.slicer.org/) or [ITK-SNAP](http://www.itksnap.org):

```
volumes/segthor/ce/Patient_01.nii.gz
data/segthor_part1/train/Patient_01/GT.nii.gz
```

## Reading the Dice numbers

`dice_val.npy` has shape `(epochs, samples, K)` with `K = 5`, but the data only contains
labels 0-3. Class 4 never appears in either the ground truth or the prediction, and an
empty-vs-empty comparison scores a perfect 1.0 — so any mean over all `K` is inflated.
For the run in this repo at epoch 14:

| classes averaged      | Dice   |
| --------------------- | ------ |
| all (k=0..4)          | 0.8744 |
| excluding absent k=4  | 0.8430 |
| organs only (k=1..3)  | 0.7915 |

Report the organs-only figure, or state explicitly which classes are included.
