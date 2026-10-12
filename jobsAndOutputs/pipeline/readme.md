# Pipeline jobs

Generic Slurm jobs for the config-driven pipeline; submit them from the repo root. Full usage is in
[docs/pipeline.md](../../docs/pipeline.md).

| job | partition | does |
|---|---|---|
| `jobs/test.job` | genoa (CPU) | test suite: legacy parity, resume, stitching, metric conventions |
| `jobs/smoke.job` | gpu_mig | 2 epochs on 16 slices + full 3D eval, for `CONFIG` |
| `jobs/train.job` | gpu_a100 | train `CONFIG` (+ `SET` overrides), then 3D eval; resubmit to resume |
| `jobs/sweep.job` | gpu_a100 | array job, one line of `SWEEP` per task |
| `jobs/cv.job` | gpu_a100 | shared Full40 fold × seed training and 3D evaluation for ordinary controls |
| `../augmentation/jobs/online/cv.job` | gpu_a100 | online augmentation CV; runs the shared `cv.job` implementation with logs under `../augmentation/outputs/online/` |
| `jobs/eval.job` | genoa (CPU) | re-run 3D evaluation of `RUN` |
| `jobs/keepalive_scratch.job` | staging | keeps `/scratch-shared/$USER/ai4mi_project/` from being purged; resubmits itself every 10 days, submit once |

`jobs/env_setup.sh` is sourced by all of them (modules, conda env, repo root).
Generic pipeline logs go to `outputs/`; online augmentation logs go to the
augmentation output directory above. These output directories exist before submission.
Each run also keeps its own `train.log` / `eval.log`.
