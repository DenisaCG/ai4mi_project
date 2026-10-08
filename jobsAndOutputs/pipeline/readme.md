# Pipeline jobs

Generic Slurm jobs for the config-driven pipeline; submit them from the repo root. Full usage is in
[docs/pipeline.md](../../docs/pipeline.md).

| job | partition | does |
|---|---|---|
| `jobs/test.job` | genoa (CPU) | test suite: legacy parity, resume, stitching, metric conventions |
| `jobs/smoke.job` | gpu_mig | 2 epochs on 16 slices + full 3D eval, for `CONFIG` |
| `jobs/train.job` | gpu_a100 | train `CONFIG` (+ `SET` overrides), then 3D eval; resubmit to resume |
| `jobs/sweep.job` | gpu_a100 | array job, one line of `SWEEP` per task |
| `jobs/cv.job` | gpu_a100 | one Full40 fold × training seed per array task, then 3D evaluation; also used for the [six online augmentation arms](../../docs/data_augmentation.md) |
| `jobs/registration.job` | genoa (CPU) | build one fold's offline synthetic 3D registrations; `FOLD=0 LIMIT=1 VISUALIZE=1` runs the [first-pair pilot](../../docs/registration_augmentation.md) |
| `jobs/eval.job` | genoa (CPU) | re-run 3D evaluation of `RUN` |
| `jobs/keepalive_scratch.job` | staging | keeps `/scratch-shared/$USER/ai4mi_project/` from being purged; resubmits itself every 10 days, submit once |

`jobs/env_setup.sh` is sourced by all of them (modules, conda env, repo root). Slurm logs go to
`outputs/<job-name>_<jobid>.out`; each run also keeps its own `train.log` / `eval.log`.
