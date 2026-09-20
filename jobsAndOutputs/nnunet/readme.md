# nnU-Net on the corrected SegTHOR data

nnU-Net is installed from PyPI (`nnunetv2==2.8.1`) into its own venv, separate from the `ai4mi` conda env. Only the conversion script and the job live in this repo.

1. Put the corrected data at `data/segthor_part1_corrected/train` (one `Patient_XX` folder per patient).
2. Build the venv once: `sbatch jobsAndOutputs/nnunet/jobs/setup_venv.job` (CPU node), or run `bash nnunet/setup_venv.sh` on a login node. It creates `~/.venv_nnunet_ai4mi`.
3. Submit from the project root: `sbatch jobsAndOutputs/nnunet/jobs/run_nnunet.job`.

`jobs/run_nnunet.job` does everything in one go: convert to nnU-Net raw format (`Dataset102_SegTHOR_corrected`), plan and preprocess, write the split, train fold 0 with nnU-Net's default trainer.

- Configuration: edit `CONFIGURATION` at the top of the job (`3d_fullres` or `2d`). Each configuration is preprocessed once.
- Split: one fold, validating on Patient_01, 11, 15, 17, 19 (the same five as `data/SEGTHOR/val`) and training on the other 15. `nnunet/prepare_dataset.py` writes it to `splits_final.json` after planning.
- Resubmitting continues from the last checkpoint. A fold that is trained and validated is skipped.

Everything nnU-Net produces stays out of git: `data/nnUNet_raw`, `data/nnUNet_preprocessed` and `results/nnunet`.

## Results

nnU-Net writes to `results/nnunet/Dataset102_SegTHOR_corrected/nnUNetTrainer__nnUNetPlans__<configuration>/fold_0/`:

- `progress.png` and `training_log_*.txt`: training curves. The pseudo Dice in the log is measured on patches and is only for monitoring.
- `checkpoint_final.pth`: the trained weights.
- `validation/*.nii.gz` and `validation/summary.json`: predictions for the five validation patients at original resolution, with per-class 3D Dice and IoU.

## Tests

The conversion tests need `nnunetv2` importable, so run them from the nnU-Net venv:

```bash
source ~/.venv_nnunet_ai4mi/bin/activate
python -m unittest discover -s nnunet
```
