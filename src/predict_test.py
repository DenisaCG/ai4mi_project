"""Predict the test CTs with one or several final runs (no ground truth, so no scores).

    python -m src.predict_test --runs runs/final_best_model/seed0 runs/final_best_model/seed1 \
        runs/final_best_model/seed2 --out predictions/final_best_model

For every test patient and every run it does one forward pass over the patient's slices (timed), saves the
softmax probabilities, averages them over the runs, takes the argmax per slice, stitches the label slices back
onto the CT's original grid and keeps only the largest 3D connected component of the heart and the trachea
(never esophagus or aorta: the esophagus is often legitimately split, and LCC on them lowered Dice in CV).

Writes
  <out>/final/<patient>.nii.gz             seed average + LCC(heart, trachea): the submission
  <out>/ensemble_no_lcc/<patient>.nii.gz   seed average only
  <out>/seed<k>/<patient>.nii.gz           each run on its own, no post-processing
  <out>/checks.csv                         shape / affine / label-set check against each test CT
  <out>/inference_time.csv                 seconds of forward pass per patient and run (after one untimed warm-up)
  <run>/probs/test/<patient>.npy           float16 softmax (Z, K, H, W) on the sliced grid, one file per run
                                           (probs_tta/ and eval/inference_time_tta.csv with --tta)
If <data.root>/test is missing (test CTs arrived after training) it is sliced first with the training
normalisation, spacing and ROI size (slice_segthor.py --test_only).
"""

import argparse
import csv
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

from src.checkpoint import load_checkpoint
from src.config import REPO, read_yaml
from src.data import build_dataset, patient_of
from src.engine import build_model
from src.eval_tta import VARIANTS, average_probs, fill_value
from src.evaluate import stitch
from src.run import read_json, resolve_device

LCC_CLASSES = (2, 3)  # heart, trachea (class order background/esophagus/heart/trachea/aorta)
EXPECTED_LABELS = set(range(5))


def largest_component(volume: np.ndarray, classes=LCC_CLASSES) -> np.ndarray:
    """Keep the largest 26-connected component of each class in `classes`; the rest of that class becomes background."""
    out = volume.copy()
    for k in classes:
        labels, n = ndimage.label(volume == k, structure=np.ones((3, 3, 3)))
        if n > 1:
            sizes = ndimage.sum_labels(np.ones_like(labels), labels, index=range(1, n + 1))
            out[(volume == k) & (labels != 1 + int(np.argmax(sizes)))] = 0
    return out


def load_net(run_dir: Path, device: torch.device):
    cfg = read_yaml((run_dir / "config.yaml").read_text())
    net = build_model(cfg, device)
    net.load_state_dict(load_checkpoint(run_dir / "checkpoints" / "best.pt")["model"])  # a final run: best.pt = last epoch
    return cfg, net.eval()


def slices_by_patient(cfg: dict, batch_size: int, split: str = "test") -> dict[str, tuple[list[str], torch.Tensor]]:
    """All slices of `split` per patient, in z order: (stems, images (Z, C, H, W))."""
    loader = torch.utils.data.DataLoader(build_dataset(cfg, split), batch_size=batch_size, shuffle=False, num_workers=0)
    stems, images = defaultdict(list), defaultdict(list)
    for batch in loader:
        for stem, img in zip(batch["stems"], batch["images"]):
            patient = patient_of(stem, cfg["data"]["patient_regex"])
            stems[patient].append(stem)
            images[patient].append(img)
    out = {}
    for patient in sorted(stems):
        order = sorted(range(len(stems[patient])), key=lambda i: stems[patient][i])
        out[patient] = ([stems[patient][i] for i in order], torch.stack([images[patient][i] for i in order]))
    return out


@torch.no_grad()
def forward_timed(net, images: torch.Tensor, batch_size: int, device: torch.device,
                  tta: list[str] | None = None, fill: float = 0.0) -> tuple[np.ndarray, float]:
    """Softmax probabilities (Z, K, H, W) and the seconds of the network forward passes alone (warping included
    when `tta` names the extra geometric passes of src.eval_tta, e.g. VARIANTS["all"])."""
    batches = [b.to(device) for b in images.split(batch_size)]  # transfer is not timed
    if device.type == "cuda":
        torch.cuda.synchronize()
    start, probs = time.perf_counter(), []
    for b in batches:
        probs.append(average_probs(net, b, tta or [], fill))
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return torch.cat(probs).cpu().numpy(), seconds


def save_volume(labels: np.ndarray, ref: nib.Nifti1Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(labels.astype(np.uint8), ref.affine, ref.header), path)


def check_volume(labels: np.ndarray, ref: nib.Nifti1Image, saved: Path) -> dict:
    out = nib.load(saved)
    return {
        "shape_matches_ct": tuple(out.shape) == tuple(ref.shape),
        "affine_matches_ct": bool(np.allclose(out.affine, ref.affine, atol=1e-4)),
        "labels": " ".join(map(str, sorted(int(v) for v in np.unique(labels)))),
        "labels_within_0_4": set(np.unique(labels).tolist()) <= EXPECTED_LABELS,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", type=Path, required=True, help="final run directories (one per seed)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--tta", action="store_true",
                   help="average each run's prediction over the geometric test-time augmentations of src.eval_tta "
                        "(rotations, scalings, shifts; +0.001 Dice in cross-validation) before the seed average")
    args = p.parse_args(argv)
    device = resolve_device(args.device)

    runs = [r.resolve() for r in args.runs]
    loaded = [load_net(r, device) for r in runs]
    cfg = loaded[0][0]
    cfg = cfg | {"train": cfg["train"] | {"debug_samples": 0}}  # always every test slice, even for a run trained with --smoke
    for r, (c, _) in zip(runs, loaded):
        if (c["data"], c["model"]) != (cfg["data"], cfg["model"]):
            raise ValueError(f"{r} was not trained with the same data/model settings as {runs[0]}")
    d = cfg["data"]
    if not d["test_source_pattern"]:
        sys.exit("data.test_source_pattern is not set in the run's config: no test CTs to stitch onto")

    root = REPO / d["root"]
    if not (root / "test").is_dir():
        pre = d["preprocess"]
        subprocess.run([sys.executable, "slice_segthor.py", "--source_dir", pre["source_dir"], "--dest_dir", str(root),
                        "--test_only", "--shape", *map(str, pre["shape"])], cwd=REPO, check=True)
    tta, fill = (VARIANTS["all"] if args.tta else None), fill_value(cfg)
    suffix = "_tta" if args.tta else ""  # keeps the plain probabilities and timings of the same runs
    resampled = bool((d.get("preprocess") or {}).get("resample"))
    use_crop = bool((d.get("preprocess") or {}).get("crop"))
    params = [sum(q.numel() for q in net.parameters()) for _, net in loaded]
    print(f"{len(runs)} run(s), {params[0] / 1e6:.2f}M parameters each, device {device}")

    data = slices_by_patient(cfg, args.batch_size)
    print(f"{len(data)} test patients: {', '.join(data)}")
    times, checks = [], []
    warm = next(iter(data.values()))[1][: args.batch_size]
    for _, net in loaded:  # untimed warm-up: cuDNN autotune and allocator
        forward_timed(net, warm, args.batch_size, device, tta, fill)

    for patient, (stems, images) in data.items():
        ref = nib.load(REPO / d["test_source_pattern"].format(patient=patient))
        crop = read_json(root / "roi_crop" / f"{patient}.json") if use_crop else None
        total = 0.0
        for k, (run, (_, net)) in enumerate(zip(runs, loaded)):
            probs, seconds = forward_timed(net, images, args.batch_size, device, tta, fill)
            times.append({"patient": patient, "run": str(run), "slices": len(stems), "seconds": round(seconds, 4)})
            probs_dir = run / f"probs{suffix}" / "test"
            probs_dir.mkdir(parents=True, exist_ok=True)
            np.save(probs_dir / f"{patient}.npy", probs.astype(np.float16))
            seed_labels = stitch(dict(enumerate(probs.argmax(axis=1).astype(np.uint8))), ref, patient, resampled, crop)
            save_volume(seed_labels, ref, args.out / f"seed{k}" / f"{patient}.nii.gz")
            total = total + probs if k else probs
        ensemble = stitch(dict(enumerate((total / len(runs)).argmax(axis=1).astype(np.uint8))), ref, patient, resampled, crop)
        save_volume(ensemble, ref, args.out / "ensemble_no_lcc" / f"{patient}.nii.gz")
        final = largest_component(ensemble)
        saved = args.out / "final" / f"{patient}.nii.gz"
        save_volume(final, ref, saved)
        checks.append({"patient": patient, "ct": ref.get_filename(), **check_volume(final, ref, saved),
                       "voxels_removed_by_lcc": int((ensemble != final).sum())})
        print(f"{patient}: {len(stems)} slices, forward {sum(t['seconds'] for t in times[-len(runs):]):.2f}s for {len(runs)} run(s)")

    for path, rows in ((args.out / "checks.csv", checks), (args.out / "inference_time.csv", times)):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    secs = np.array([t["seconds"] for t in times])
    with (args.out / "timing_summary.csv").open("w", newline="") as f:  # one row per model, for the timing slide
        w = csv.writer(f)
        w.writerow(["experiment", "runs", "parameters", "device", "tta", "patients", "mean_s_per_patient_per_run",
                    "std_s", "mean_s_per_patient_all_runs"])
        w.writerow([cfg["experiment"], len(runs), params[0], gpu, bool(args.tta), len(data), round(secs.mean(), 4),
                    round(secs.std(), 4), round(secs.sum() / len(data), 4)])
    for run in runs:
        mine = [t["seconds"] for t in times if t["run"] == str(run)]
        (run / "eval").mkdir(parents=True, exist_ok=True)
        with (run / "eval" / f"inference_time{suffix}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patient", "slices", "seconds"])
            w.writerows((t["patient"], t["slices"], t["seconds"]) for t in times if t["run"] == str(run))
        print(f"{run}: mean forward {np.mean(mine):.3f} s per patient over {len(mine)} patients")
    bad = [c["patient"] for c in checks if not (c["shape_matches_ct"] and c["affine_matches_ct"] and c["labels_within_0_4"])]
    if bad:
        sys.exit(f"CHECK FAILED for {bad}, see {args.out / 'checks.csv'}")
    print(f"done: {len(checks)} patients -> {args.out / 'final'}")


if __name__ == "__main__":
    main()
