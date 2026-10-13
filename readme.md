# AI for medical imaging — Fall 2026 group project

> **Group pipeline:** experiments are run with the config-driven pipeline in [`src/`](src/)
> (`python -m src.train --config configs/<experiment>.yaml`, via the Slurm jobs in
> [`jobsAndOutputs/pipeline/jobs/`](jobsAndOutputs/pipeline/jobs/)). Start with
> [docs/pipeline.md](docs/pipeline.md); adding models/losses/data: [docs/extending.md](docs/extending.md);
> metric definitions and result tables: [docs/metrics.md](docs/metrics.md).
> The course instructions below describe the original `main.py` baseline, which is kept for reference.

<a id="group-23-project"></a>
## Group 23 project: final model and results

### Final model

**ResEnc U-Net with deep supervision**, Dice + cross-entropy loss, trained on the corrected SEGTHOR labels (five classes: background, esophagus, heart, trachea, aorta).

- Pre-processing: resampling to the median spacing (0.98 x 0.98 x 2.5 mm), HU clip to the 0.5-99.5th percentile, z-score, crop to the region of interest.
- Augmentation: rotation (+-15 deg), scaling (0.8-1.2) and shift (10%), each with p = 0.7.
- Training: Adam, constant learning rate 5e-4, 50 epochs. Config: [`configs/final_best_model.yaml`](configs/final_best_model.yaml) (all 40 labelled patients); its cross-validation twin is [`configs/full_cv4_resenc_ds_unet_dice_ce_strong_geometric_50ep.yaml`](configs/full_cv4_resenc_ds_unet_dice_ce_strong_geometric_50ep.yaml).
- Inference: geometric test-time augmentation (rotations, scalings, shifts), softmax averaged over 3 seeds, then the largest connected component of heart and trachea only.

Held-out result (4-fold cross-validation over the 40 patients, mean of 4 folds x 3 seeds, 3D metrics, foreground = mean of the four organs; full table in [docs/final_model_results.md](docs/final_model_results.md)):

| | Dice | HD95 (mm) | ASSD (mm) |
|---|---|---|---|
| Course baseline (ENet + CE, starter pre-processing) | 0.707 | 25.8 | 4.91 |
| Final model, single seed, no TTA | 0.893 | 9.8 | 1.75 |
| Final model, 3-seed average + TTA + largest component (the submitted pipeline) | 0.901 | 7.5 | 1.43 |

### Experiments against the course list

All rows are 4-fold CV over the 40 patients, 12 runs (4 folds x 3 seeds) unless a run count is given; Dice, HD95 and ASSD are 3D foreground means. The baseline is `full_cv4_enet_ce`. Per-run files are in [`metrics/`](metrics/); the augmentation selection is written up in [docs/augmentation_selection.md](docs/augmentation_selection.md).

| Course topic | What we tried | Dice (HD95 mm), vs the row it is compared with | Outcome |
|---|---|---|---|
| Baseline | ENet + CE, 256x256 slices, per-volume min-max, 25 epochs | 0.707 (25.8) | reference |
| Pre-processing and loss | median-spacing resampling, HU window + z-score, ROI crop, Dice + CE (the "improved ENet") | 0.835 (14.9), +0.128 over the baseline | kept. The two changes were made together in CV; the midterm single-split ablation went from 0.437 to 0.802 |
| Loss functions | weighted Dice + CE; boundary loss (two schedules); Tversky + CE | 0.833 / 0.835 / 0.834 / 0.836 (4 runs) against 0.835 | no gain on ENet, so plain Dice + CE stayed. Boundary on the final model: 0.891 (9.2) against 0.893 (9.8) |
| Augmentation | rotation, scaling, shift (together) | ENet 0.835 -> 0.851; ResEnc DS 0.867 -> 0.880 (improved in 12/12 pairs) | kept. Stronger geometric (p 0.7, +-15 deg, 0.8-1.2): ENet 0.862, ResEnc DS 0.886 at 25 epochs |
| Augmentation, rejected | shear, Gaussian noise, blur, brightness/contrast, gamma, elastic | +0.003 to -0.005 (seed 0 only for the single transforms); elastic 0.862, same as without | dropped. HU values are calibrated, and CT geometry is rigid |
| 2.5D input | the slice plus its two neighbours as input channels, on ENet, ResEnc DS and the final recipe with the boundary loss | 0.837 / 0.867 / 0.893 against 0.835 / 0.867 / 0.891 (2D, same boundary loss) | no real gain, so the final model is 2D |
| Architectures | U-Net 0.852; + attention 0.849; + deep supervision 0.861; residual encoder (ResEnc) 0.858; ResEnc + deep supervision 0.867 | against ENet 0.835 | ResEnc + deep supervision is best (HD95 13.8) |
| Non-CNN encoder | frozen pretrained DINOv2 ViT-S/14 encoder with a U-Net decoder | 0.837 (16.9) against 0.852 for the plain U-Net | worse, not used |
| Optimizer, schedule, epochs | cosine decay (U-Net, 25 epochs); 25 -> 50 epochs | cosine 0.844 against constant LR 0.852; 50 epochs: ENet 0.862 -> 0.871, ResEnc DS 0.886 -> 0.893 | constant LR, 50 epochs |
| Post-processing | 3-seed softmax average, largest component of heart and trachea, TTA | 0.893 (9.8) -> 0.901 (7.5) | kept. Largest component on all organs or on the esophagus lowered Dice, and CRF did not help |

Caveats: the augmentation transforms were chosen on the same folds that are reported, so the augmentation gains are slightly optimistic; most differences below 0.003 Dice are within seed and fold variation, and no significance test was run. The midterm numbers (0.437 and 0.802, single split) were computed before the rule for a missed organ was changed (HD95 and ASSD are now the volume diagonal when exactly one of ground truth and prediction is empty, instead of being dropped); no cross-validation run and not the final model is affected.

### Reproducing the results

```
# environment and data: see "Codebase use" below; slicing and the folds are built by jobsAndOutputs/pipeline/jobs/{build_cv_folds,final_build}.job

# 1. cross-validation run of the final recipe: fold x seed array (12 tasks), trains and evaluates in 3D
sbatch --job-name=full_cv4_resenc_ds_unet_dice_ce_strong_geometric_50ep \
       --export=ALL,CONFIG=configs/full_cv4_resenc_ds_unet_dice_ce_strong_geometric_50ep.yaml \
       jobsAndOutputs/pipeline/jobs/cv.job

# 2. final model on all 40 patients, seeds 0-2
sbatch --job-name=final_best_model --export=ALL,CONFIG=configs/final_best_model.yaml \
       jobsAndOutputs/pipeline/jobs/final_train.job

# 3. test prediction (Patient_41-60): TTA, seed average, largest component of heart and trachea
python -m src.predict_test --runs runs/final_best_model/seed0 runs/final_best_model/seed1 runs/final_best_model/seed2 \
       --out predictions/final_best_model_tta        # or sbatch jobsAndOutputs/pipeline/jobs/final_predict.job (TTA=1)

# 4. evaluation: 3D Dice, HD95, ASSD of a run on its validation patients
python -m src.evaluate --run runs/<experiment>/seed<k>
python -m src.eval_tta --run runs/<experiment>/seed<k>    # the same with test-time augmentation
python -m src.aggregate                                   # pooled tables in metrics/
```

Training and evaluation run through Slurm on a GPU (A100); see [docs/pipeline.md](docs/pipeline.md). The 3-seed average and largest-component scoring on the validation folds is `python -m tools.prob_postprocess score` ([docs/final_model_results.md](docs/final_model_results.md), "Sources").

### The submission archive

- **Test predictions** (`test/pred`): Patient_41-60, produced by step 3 above (3 seeds averaged, TTA, largest component).
- **Validation set** (`val/pred`, `val/gt`, the `.npz` metrics): all 40 labelled patients, **out of fold**, with the same post-processing as the test predictions. Each patient is predicted by the three seeds of the fold model that did not train on it (`full_cv4_resenc_ds_unet_dice_ce_strong_geometric_50ep`, folds 0-3 hold 10 patients each): test-time-augmented softmax averaged over the seeds, argmax, then the largest connected component of heart and trachea. The three `.npz` files (Dice, HD95, ASSD; patient to one value per class, background NaN) are computed on exactly these volumes. Ground truth is the corrected `GT.nii.gz`. Their mean (Dice 0.9011, HD95 7.45 mm, ASSD 1.43 mm) is the post-processed + TTA row above. Built by `tools/build_submission_archive.py` (`jobsAndOutputs/pipeline/jobs/build_submission_archive.job`) from the TTA softmax that `src.eval_tta --save-probs` writes.
- **Model** (`bestmodel-group-23.pkl`): seed 0 of `final_best_model`, trained on all 40 patients. Two things follow. The submitted test predictions average seeds 0-2, so this single file alone does not reproduce them exactly (seeds 1 and 2, about 288 MB each, are not shipped; they are `runs/final_best_model/seed{1,2}/checkpoints/best_model.pkl` and `configs/final_best_model.yaml` retrains them). And because this model saw every labelled patient, the validation predictions do not come from it: they come from the four fold models (three seeds each), and re-predicting the validation patients with the pkl would give inflated scores.

<!-- MarkdownTOC autolink="true" autoanchor="true" -->

- [Group 23 project: final model and results](#group-23-project)
- [Project overview](#project-overview)
- [Codebase features](#codebase-features)
- [Codebase use](#codebase-use)
    - [Setting up the environment](#setting-up-the-environment)
        - [Setting up the environment - Some troubleshooting for windows users](#setting-up-the-environment---some-troubleshooting-for-windows-users)
    - [Getting the data](#getting-the-data)
    - [Training a base network](#training-a-base-network)
    - [Viewing the results](#viewing-the-results)
        - [2D viewer](#2d-viewer)
        - [3D viewers](#3d-viewers)
    - [Plotting the metrics](#plotting-the-metrics)
- [Submission and scoring](#submission-and-scoring)
    - [Packing the code](#packing-the-code)
    - [Saving the best model](#saving-the-best-model)
    - [Archiving everything for submission](#archiving-everything-for-submission)
- [Known issues](#known-issues)
    - [Cannot pickle lambda in the dataloader](#cannot-pickle-lambda-in-the-dataloader)
    - [Pytorch not compiled for Numpy 2.0](#pytorch-not-compiled-for-numpy-20)
    - [Viewer on Windows](#viewer-on-windows)

<!-- /MarkdownTOC -->


<a id="project-overview"></a>
## Project overview
The project is based around the SegTHOR challenge data, which was kindly allowed by Caroline Petitjean (challenge organizer) to use for the course. The challenge was originally on the segmentation of different organs: heart, aorta, esophagus and trachea.


<a id="codebase-features"></a>
## Codebase features
This codebase is given as a starting point, to provide an initial neural network that converges during training. (For broader context, this is itself a fork of an [older conference tutorial](https://github.com/LIVIAETS/miccai_weakly_supervised_tutorial) we gave few years ago.) It also provides facilities to locally run some test on a laptop, with a toy dataset and dummy network.

Summary of codebase (in PyTorch)
* slicing the 3D Nifti files to 2D `.png`; **To re-implement as assignment 01**
* stitching 2D `.png` slices to 3D volume compatible with initial nifti files; **To re-implement as assignment 03**
* basic 2D segmentation network;
* basic training and printing with cross-entroly loss and Adam;
* partial cross-entropy alternative as a loss (to disable one class during training);
* debug options and facilities (cpu version, "dummy" network, smaller datasets);
* saving of predictions as `.png`;
* log the 2D DSC and cross-entropy over time, with basic plotting;
* tool to compare different segmentations (`viewer/viewer.py`).

**Some recurrent questions might be addressed here directly.** As such, it is expected that small change or additions to this readme to be made.

<a id="codebase-use"></a>
## Codebase use
In the following, a line starting by `$` usually means it is meant to be typed in the terminal (bash, zsh, fish, ...), whereas no symbol might indicate some python code.

<a id="setting-up-the-environment"></a>
### Setting up the environment
```
$ git clone https://github.com/HKervadec/ai4mi_project.git
$ cd ai4mi_project
$ git submodule init
$ git submodule update
```

This codebase was written for a somewhat recent python (3.10 or more recent). (**Note: Ubuntu and some other Linux distributions might make the distasteful choice to have `python` pointing to 2.+ version, and require to type `python3` explicitly.**) The required packages are listed in [`requirements.txt`](requirements.txt) and a [virtual environment](https://docs.python.org/3/library/venv.html) can easily be created from it through [pip](https://pypi.org/):
```
$ python -m venv ai4mi
$ source ai4mi/bin/activate
$ which python  # ensure this is not your system's python anymore
$ python -m pip install -r requirements.txt
```
Conda is an alternative to pip, but is recommended not to mix `conda install` and `pip install`.

<a id="setting-up-the-environment---some-troubleshooting-for-windows-users"></a>
#### Setting up the environment - Some troubleshooting for windows users
These steps assume you are using Git Bash + Anaconda + an IDE (e.g., PyCharm).

Open git bash and run:
Step 1:
```
$ git clone https://github.com/HKervadec/ai4mi_project.git
$ cd ai4mi_project
$ git submodule init
$ git submodule update
```
Step 2:
```
# 1) Create a fresh conda env with Python 3.10+ (matches project note)
$ conda create -n ai4mi python=3.10 -y

# 2) Activate it
$ conda activate ai4mi

# 3) (Optional but nice) make sure pip is present/updated
$ python -m pip install --upgrade pip

# 4) From the repo folder, install dependencies with pip
$ python -m pip install -r requirements.txt
```

Some common troubleshooting for windows users:

In case in bash u got - conda: command not found

Open Anaconda Prompt:
```
conda init bash

#Find where conda is installed
where conda
```
Yous should get sth like - C:\Users\<YourName>\anaconda3

Close and open git bash - change CONDA_HOME in the code below

```
$ CONDA_HOME="/c/Users/<YourName>/anaconda3"
if [ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]; then
    . "$CONDA_HOME/etc/profile.d/conda.sh"
else
    export PATH="$CONDA_HOME:$CONDA_HOME/Scripts:$CONDA_HOME/Library/bin:$PATH"
fi
```
You can also create new conda environment in anaconda prompt 

<a id="getting-the-data"></a>
### Getting the data
The synthetic dataset is generated randomly, whereas for Segthor it is required to put the file [`segthor_part1.zip`](https://amsuni-my.sharepoint.com/:u:/g/personal/h_t_g_kervadec_uva_nl/IQBJLXRY5wedSYEuofqRtuylAWiiHp2ciems5XSCu3DFMkA?e=qa3Ujf) (required a UvA account) in the `data/` folder. If the computer running it is powerful enough, the recipe for `data/SEGTHOR` can be modified in the [Makefile](Makefile) to enable multi-processing (`-p -1` option, see `python slice_segthor.py --help` or its code directly).
```
$ make data/TOY2
$ make data/SEGTHOR
```


For windows users, you can use the following instead
```
$ rm -rf data/TOY2_tmp data/TOY2
$ python gen_two_circles.py --dest data/TOY2_tmp -n 1000 100 -r 25 -wh 256 256
$ mv data/TOY2_tmp data/TOY2

$ sha256sum -c data/segthor_train.sha256
$ unzip -q data/segthor_train.zip

$ rm -rf data/SEGTHOR_tmp data/SEGTHOR
$ python  slice_segthor.py --source_dir data/segthor_train --dest_dir data/SEGTHOR_tmp \
         --shape 256 256 --retain 10
$ mv data/SEGTHOR_tmp data/SEGTHOR
````

<a id="training-a-base-network"></a>
### Training a base network
Running a training
```
$ python main.py --help
usage: main.py [-h] [--epochs EPOCHS] [--dataset {TOY2,SEGTHOR}] [--mode {partial,full}] --dest DEST [--gpu] [--debug]

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS
  --dataset {TOY2,SEGTHOR}
  --mode {partial,full}
  --dest DEST           Destination directory to save the results (predictions and weights).
  --gpu
  --debug               Keep only a fraction (10 samples) of the datasets, to test the logic around epochs and logging easily.
$ python main.py --dataset TOY2 --mode full --epoch 25 --dest results/toy2/ce --gpu
```

The codebase uses a lot of assertions for control and self-documentation, they can easily be disabled with the `-O` option (for faster training) once everything is known to be correct (for instance run the previous command for 1/2 epochs, then kill it and relaunch it):
```
$ python -O main.py --dataset TOY2 --mode full --epoch 25 --dest results/toy2/ce --gpu
```

<a id="viewing-the-results"></a>
### Viewing the results
<a id="2d-viewer"></a>
#### 2D viewer
Comparing some predictions with the provided [viewer](viewer/viewer.py) (right-click to go to the next set of images, left-click to go back):
```
$ python viewer/viewer.py --img_source data/TOY2/val/img \
    data/TOY2/val/gt results/toy2/ce/iter000/val results/toy2/ce/iter005/val results/toy2/ce/best_epoch/val \
    --show_img -C 256 --no_contour
```
![Example of the viewer on the TOY example](viewer_toy.png)
**Note:** if using it from a SSH session, it requires X to be forwarded ([Unix/BSD](https://man.archlinux.org/man/ssh.1#X), [Windows](https://mobaxterm.mobatek.net/documentation.html#1_4)) for it to work. Note that X forwarding also needs to be enabled on the server side.


```
$ python viewer/viewer.py --img_source data/SEGTHOR/val/img \
    data/SEGTHOR/val/gt results/segthor/ce/iter000/val results/segthor/ce/best_epoch/val \
    -n 2 -C 5 --remap "{63: 1, 126: 2, 189: 3, 252: 4}" \
    --legend --class_names background esophagus heart trachea aorta
```
<!-- ![Example of the viewer on SegTHOR](viewer_segthor.png) -->

<a id="3d-viewers"></a>
#### 3D viewers
To look at the results in 3D, it is necessary to reconstruct the 3D volume from the individual 2D predictions saved as images.
To stitch the `.png` back to a nifti file:
```
$ python stitch.py --data_folder results/segthor/ce/best_epoch/val \
    --dest_folder volumes/segthor/ce \
    --num_classes 255 --grp_regex "(Patient_\d\d)_\d\d\d\d" \
    --source_scan_pattern "data/segthor_train/train/{id_}/GT.nii.gz"
```

[3D Slicer](https://www.slicer.org/) and [ITK Snap](http://www.itksnap.org) are two popular viewers for medical data, here comparing `GT.nii.gz` and the corresponding stitched prediction `Patient_01.nii.gz`:
![Viewing label and prediction](3dslicer.png)

Zooming on the prediction with smoothing disabled:
![Viewing the prediction without smoothing](3dslicer_zoom.png)


<a id="plotting-the-metrics"></a>
### Plotting the metrics
There are some facilities to plot the metrics saved by [`main.py`](main.py):
```
$ python plot.py --help
usage: plot.py [-h] --metric_file METRIC_MODE.npy [--dest METRIC_MODE.png] [--headless]

Plot data over time

options:
  -h, --help            show this help message and exit
  --metric_file METRIC_MODE.npy
                        The metric file to plot.
  --dest METRIC_MODE.png
                        Optional: save the plot to a .png file
  --headless            Does not display the plot and save it directly (implies --dest to be provided.
$ python plot.py --metric_file results/segthor/ce/dice_val.npy --dest results/segthor/ce/dice_val.png
```
![Validation DSC](dice_val.png)


<a id="submission-and-scoring"></a>
## Submission and scoring
Groups will have to submit:
* archive of the git repo with the whole project, which includes:
    * slicing (if any) and any other pre-processing;
    * training;
    * post-processing when applicable;
    * inference;
    * metrics computation/scripts to run the metrics submodule;
* the best trained model;
* predictions on the test set (`sha256sum -c data/test.zip.sha256` as optional checksum);
* predictions on the group's internal validation set, the labels of their validation set, and the metrics they computed (akin to Assignment 3).

The main criterions for scoring will include (listed here only for convenience, please see Canvas for reference rubric):
* improvement or lack thereof of performances over baseline;
* code quality/clear [git use](git.md);
* the [final choice of metrics](https://metrics-reloaded.dkfz.de/) (they need to be in 3D);
* correctness of the computed metrics (on the validation set);
* oral presentation.


<a id="packing-the-code"></a>
### Packing the code
`$ git bundle create group-XX.bundle master`

<a id="saving-the-best-model"></a>
### Saving the best model
`torch.save(net, args.dest / "bestmodel-group-XX.pkl")`

<a id="archiving-everything-for-submission"></a>
### Archiving everything for submission
All files should be grouped in single folder with the following structure
```
group-XX/
    test/
        pred/
            Patient_41.nii.gz
            Patient_42.nii.gz
            ...
    val/
        pred/
            Patient_21.nii.gz
            Patient_32.nii.gz
            ...
        gt/
            Patient_21.nii.gz
            Patient_32.nii.gz
            ...
        metric01.npz
        metric02.npz
        ...
    group-XX.bundle
    bestmodel-group-XX.pkl
```
The metrics should be a `.npz` archives, that maps patient ID (e.g., `Patient_21`) to a `ndarray` with shape `KxD` (or `K` if `D = 1`), with `K` the number of classes and `D` the eventual dimensionality of the metric (can be simply 1). Ultimately it is the same format as Distorch from Assignment 3.


The folder should then be [tarred](https://xkcd.com/1168/) and compressed, e.g.:
```
Example using Zstandard:
$ tar cf - group-XX/ | zstd -T0 -3 > group-XX.tar.zst
Example using gunzip:
$ tar cf group-XX.tar.gz - group-XX/
```

<a id="known-issues"></a>
## Known issues
<a id="cannot-pickle-lambda-in-the-dataloader"></a>
### Cannot pickle lambda in the dataloader
Some installs (probably due to Python/Pytorch version mismatch) throw an error about an inability to pickle lambda functions (at the dataloader stage). Short of reinstalling everything, setting the number of workers to 0 seems to get around the problem (`--num_workers 0`).

<a id="pytorch-not-compiled-for-numpy-20"></a>
### Pytorch not compiled for Numpy 2.0
It may happen that Pytorch, when installed through pip, was compiled for Numpy 1.x, which creates some inconsistencies. Downgrading Numpy seems to solve it: `pip install --upgrade "numpy<2"`

<a id="viewer-on-windows"></a>
### Viewer on Windows
Windows has different paths names (`\` in stead of `/`), so the default regex in the viewer needs to be changed to `--id_regex=".*\\\\(.*).png"`.
