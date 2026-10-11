# Sourced by the full40_* jobs: nnU-Net venv, scratch dirs, the 40-patient dataset. Not a job itself.
VENV="$HOME/.venv_nnunet_ai4mi"
DATASET=Dataset103_SegTHOR_full40
DATASET_ID=103
TRAINER="${TRAINER:-nnUNetTrainer_250epochs}"   # nnUNetTrainer for the full 1000 epochs
DATA_DIR="$HOME/ai4mi_project/data/segthor_train_full/train"   # 40 patients, corrected GT
BASE="/scratch-shared/$USER/nnunet_full40"

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
source "$VENV/bin/activate"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

export TEMP="$TMPDIR" TMP="$TMPDIR"
export nnUNet_raw="$BASE/raw" nnUNet_preprocessed="$BASE/preprocessed" nnUNet_results="$BASE/results"
export nnUNet_n_proc_DA="${SLURM_CPUS_PER_TASK:-12}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
