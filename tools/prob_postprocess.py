"""Post-processing that needs the network's softmax probabilities: averaging the seeds' probabilities,
gating organs by slice confidence, and a DenseCRF (pydensecrf, Kraehenbuehl and Koltun) on the probabilities.

    python -m tools.prob_postprocess infer --experiment full_cv4_resenc_ds_unet_dice_ce     # GPU
    python -m tools.prob_postprocess score --experiment full_cv4_resenc_ds_unet_dice_ce --workers 9

infer: for each run, re-predicts the val split with runs/<run>/checkpoints/best.pt and saves the softmax of
every patient as float16 (Z, K, 256, 256) under --probs/<run>/<patient>.npy.
score: stitches the label maps back onto the source grid exactly as src.evaluate does and scores them
against the source GT with src.metrics_3d. Variants (per fold; seeds' probabilities averaged unless noted):
  repro_seed<s>        argmax of that seed's probabilities (must equal the stored volume up to GPU rounding)
  avg, avg_lcc_heart_trachea
  avg_conf_<tau>       organ removed on slices where its highest probability on the slice is below tau
  avg_crf_<name>       DenseCRF on the averaged probabilities, per axial slice, on the 256x256 network grid
  seed0_crf_<name>     DenseCRF on seed 0's probabilities alone
Reference rows (baseline, gate, lcc_heart_trachea, seed_vote*) are read from results/seed_ensemble/rows.csv.
The CRF configurations are a small grid picked on the scored folds, so its numbers are slightly optimistic.
"""

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from src.config import REPO, read_yaml
from tools.gating_oracle import largest_component, recovery_table, run_dirs, table, volume_rows, write_csv
from tools.postprocess_filters import take_organs
from tools.seed_ensemble import scaled_count_table

CONF_THRESHOLDS = (0.6, 0.8, 0.9)
# (sxy, compat) of the Gaussian smoothness kernel; (sxy, srgb, compat) of the bilateral appearance kernel
CRF_CONFIGS = {
    "a": {"gauss": (3, 3), "bilateral": (20, 10, 5)},
    "b": {"gauss": (3, 3), "bilateral": (50, 5, 5)},
    "c": {"gauss": (5, 5), "bilateral": (40, 10, 10)},
}
CRF_ITERATIONS = 5
CRF_MIN_ORGAN_PROB = 0.2  # slices where no organ reaches this are left as the argmax (CRF would keep them background)
PROBS_DEFAULT = Path("/scratch-shared") / Path.home().name / "ai4mi_project" / "postprocess_probs"
LONG_HD95 = 20.0  # mm


def confidence_gate(probs: np.ndarray, labels: np.ndarray, tau: float, classes: list[int]) -> np.ndarray:
    """Sets organ k to background on the z slices where its highest probability among the pixels labelled k
    is below `tau`. probs (Z, K, H, W), labels (Z, H, W)."""
    out = labels.copy()
    for k in classes:
        mask = labels == k
        confidence = np.where(mask, probs[:, k], 0).max(axis=(1, 2))
        out[mask & ((confidence > 0) & (confidence < tau))[:, None, None]] = 0
    return out


def crf_labels(probs: np.ndarray, images: np.ndarray, config: dict) -> np.ndarray:
    """DenseCRF mean-field labels per axial slice. probs (Z, K, H, W) softmax, images (Z, H, W) uint8 (the
    windowed CT the network saw). Unary = -log probability; a Gaussian kernel on position and a bilateral
    kernel on position and intensity, as in the DenseCRF paper (pydensecrf)."""
    from pydensecrf import densecrf
    from pydensecrf.utils import unary_from_softmax

    z_count, k_count, height, width = probs.shape
    out = probs.argmax(axis=1).astype(np.uint8)
    for z in range(z_count):
        p = np.ascontiguousarray(probs[z], dtype=np.float32)
        if p[1:].max() < CRF_MIN_ORGAN_PROB:
            continue
        crf = densecrf.DenseCRF2D(width, height, k_count)
        crf.setUnaryEnergy(np.ascontiguousarray(unary_from_softmax(p, clip=1e-4)))
        crf.addPairwiseGaussian(sxy=config["gauss"][0], compat=config["gauss"][1])
        rgb = np.ascontiguousarray(np.repeat(images[z][..., None], 3, axis=2))
        sxy, srgb, compat = config["bilateral"]
        crf.addPairwiseBilateral(sxy=sxy, srgb=srgb, rgbim=rgb, compat=compat)
        q = np.array(crf.inference(CRF_ITERATIONS))
        out[z] = q.argmax(axis=0).reshape(height, width)
    return out


def stitch_labels(labels: np.ndarray, ref: nib.Nifti1Image, patient: str, cfg: dict) -> np.ndarray:
    """Label maps (Z, H, W) of a patient back onto the source grid, as src.evaluate."""
    from src.evaluate import stitch

    d = cfg["data"]
    crop = None
    if (d.get("preprocess") or {}).get("crop"):
        crop = json.loads((REPO / d["root"] / "roi_crop" / f"{patient}.json").read_text())
    slices = {z: labels[z] for z in range(len(labels))}
    return stitch(slices, ref, patient, resampled=bool((d.get("preprocess") or {}).get("resample")), crop=crop)


def infer(run_dir: Path, probs_dir: Path, device) -> None:
    """Saves the softmax of every val patient of `run_dir` unless it is already there."""
    import torch
    from torch.utils.data import DataLoader

    from src.checkpoint import load_checkpoint
    from src.data import build_dataset
    from src.engine import build_model
    from src.evaluate import group_by_patient

    cfg = read_yaml((run_dir / "config.yaml").read_text())
    out = probs_dir / run_dir.relative_to(REPO / "runs")
    loader = DataLoader(
        build_dataset(cfg | {"train": cfg["train"] | {"debug_samples": 0}}, "val"),
        batch_size=cfg["data"]["batch_size"], num_workers=cfg["data"]["num_workers"], shuffle=False,
    )
    net = build_model(cfg, device)
    net.load_state_dict(load_checkpoint(run_dir / "checkpoints" / "best.pt")["model"])
    net.eval()
    preds = {}
    with torch.no_grad():
        for batch in loader:
            probs = net(batch["images"].to(device)).softmax(dim=1).half().cpu().numpy()
            preds.update(zip(batch["stems"], probs))
    out.mkdir(parents=True, exist_ok=True)
    for patient, slices in group_by_patient(preds, cfg["data"]["patient_regex"]).items():
        if sorted(slices) != list(range(len(slices))):
            raise ValueError(f"{patient}: slices {sorted(slices)[:3]}... are not 0..n-1")
        np.save(out / f"{patient}.npy", np.stack([slices[z] for z in range(len(slices))]))


def score_patient(task: tuple, probs_dir: Path) -> list[dict]:
    """Rows of the requested variants for one patient of one fold, plus per seed the share of voxels that
    differ from the stored volume (as variant repro_seed<s>, in the column pred_voxels of a row with class_idx -1)."""
    run_dirs_, pattern, patient, names, classes, variants = task
    cfg = read_yaml((Path(run_dirs_[0]) / "config.yaml").read_text())
    ref = nib.load(REPO / pattern.format(patient=patient))
    gt = np.asarray(ref.dataobj)
    spacing = tuple(float(s) for s in ref.header.get_zooms()[:3])
    heart, trachea = names.index("heart"), names.index("trachea")
    rels = [Path(r).relative_to(REPO / "runs") for r in run_dirs_]
    probs = [np.load(probs_dir / rel / f"{patient}.npy") for rel in rels]
    img_dir = REPO / cfg["data"]["root"] / "val" / "img"
    images = np.stack([np.array(Image.open(img_dir / f"{patient}_{z:04d}.png").convert("L")) for z in range(len(probs[0]))])
    fold_run = Path(run_dirs_[0]).parent / "avg"
    rows, scored = [], {}

    def lcc_ht(vol: np.ndarray) -> np.ndarray:
        return take_organs(vol, largest_component(vol, [heart, trachea]), [heart, trachea])

    def score(run: Path, variant: str, vol: np.ndarray) -> None:
        rows.extend(volume_rows(run, patient, variant, vol, gt, spacing, names, classes, scored.setdefault(run, {})))

    def add(run: Path, variant: str, labels: np.ndarray) -> np.ndarray:
        vol = stitch_labels(labels, ref, patient, cfg)
        score(run, variant, vol)
        return vol

    avg = np.mean([p.astype(np.float32) for p in probs], axis=0)
    avg_labels = avg.argmax(axis=1).astype(np.uint8)
    if "repro" in variants:
        for run, p, rel in zip(run_dirs_, probs, rels):
            vol = stitch_labels(p.argmax(axis=1).astype(np.uint8), ref, patient, cfg)
            stored = np.asarray(nib.load(Path(run) / "volumes" / "val" / f"{patient}.nii.gz").dataobj)
            rows.append({"run": rel.as_posix(), "patient": patient, "variant": f"repro_{rel.name}", "class_idx": -1,
                         "class_name": "all", "fp_slices": 0, "fp_voxels": 0,
                         "pred_voxels": int((vol != stored).sum()), "dice": float("nan"), "hd95": float("nan"), "assd": float("nan")})
    if "avg" in variants:
        vol = add(fold_run, "avg", avg_labels)
        score(fold_run, "avg_lcc_heart_trachea", lcc_ht(vol))
    for tau in CONF_THRESHOLDS:
        if f"conf_{tau}" in variants:
            add(fold_run, f"avg_conf_{tau}", confidence_gate(avg, avg_labels, tau, classes))
    for name, config in CRF_CONFIGS.items():
        if f"crf_{name}" in variants:
            add(fold_run, f"avg_crf_{name}", crf_labels(avg, images, config))
        if f"seed0_crf_{name}" in variants:
            add(Path(run_dirs_[0]), f"seed0_crf_{name}", crf_labels(probs[0].astype(np.float32), images, config))
    return rows


def read_rows(path: Path, variants: tuple[str, ...]) -> list[dict]:
    """Rows of results/seed_ensemble/rows.csv for the given variants, numbers parsed."""
    ints, floats = ("class_idx", "fp_slices", "fp_voxels", "pred_voxels"), ("dice", "hd95", "assd")
    with path.open(newline="") as f:
        return [
            r | {k: int(r[k]) for k in ints} | {k: float(r[k]) for k in floats}
            for r in csv.DictReader(f)
            if r["variant"] in variants
        ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("infer", "score"))
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--probs", type=Path, default=PROBS_DEFAULT)
    parser.add_argument("--out", type=Path, default=REPO / "results" / "prob_postprocess")
    parser.add_argument("--reference", type=Path, default=REPO / "results" / "seed_ensemble" / "rows.csv")
    parser.add_argument(
        "--variants", nargs="+",
        default=["repro", "avg", *(f"conf_{t}" for t in CONF_THRESHOLDS), *(f"crf_{n}" for n in CRF_CONFIGS)],
        help="repro avg conf_<tau> crf_<name> seed0_crf_<name>",
    )
    args = parser.parse_args(argv)

    dirs = run_dirs(args.experiment, args.folds, args.seeds)
    if args.command == "infer":
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for d in dirs:
            if all((args.probs / d.relative_to(REPO / "runs") / f"{p.name.removesuffix('.nii.gz')}.npy").exists()
                   for p in (d / "volumes" / "val").glob("*.nii.gz")):
                continue
            infer(d, args.probs, device)
            print(f"inferred {d.relative_to(REPO / 'runs')}", flush=True)
        return

    cfg = read_yaml((dirs[0] / "config.yaml").read_text())
    names, classes = cfg["data"]["class_names"], cfg["eval"]["classes"]
    pattern = cfg["data"]["source_pattern"]
    tasks = []
    for fold in range(args.folds):
        fold_runs = [str(d) for d in dirs if d.parent.name.endswith(f"_fold{fold}")]
        patients = sorted(p.name.removesuffix(".nii.gz") for p in (Path(fold_runs[0]) / "volumes" / "val").glob("*.nii.gz"))
        tasks += [(fold_runs, pattern, p, names, classes, tuple(args.variants)) for p in patients]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [r for part in pool.map(partial(score_patient, probs_dir=args.probs), tasks) for r in part]

    repro = [r for r in rows if r["class_idx"] == -1]
    rows = [r for r in rows if r["class_idx"] != -1]
    reference = read_rows(args.reference, ("baseline", "gate", "lcc_heart_trachea", "seed_vote", "seed_vote_lcc_heart_trachea"))
    reference += [r | {"variant": "baseline_seed0"} for r in reference if r["variant"] == "baseline" and r["run"].endswith("/seed0")]
    rows = reference + rows
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "rows.csv", rows)
    voxels = [(r["run"], r["patient"], r["pred_voxels"]) for r in repro]
    worst = max((v for *_, v in voxels), default=0)
    summary = [
        f"# Probability-based post-processing: {args.experiment}, {len(args.seeds)} seeds, {args.folds} folds, {len(tasks)} patients",
        "",
        f"Re-predicted argmax vs the stored volumes: {sum(v for *_, v in voxels)} voxels differ in total, at most {worst} in one patient volume (GPU rounding; 0 expected otherwise).",
        "Reference rows (baseline, gate, lcc_heart_trachea, seed_vote*) come from results/seed_ensemble/rows.csv; baseline_seed0 is seed 0 alone.",
        "avg*: seeds' probabilities averaged, one result per fold. CRF settings (sxy, compat; sxy, srgb, compat): "
        + "; ".join(f"{n} gauss {c['gauss']} bilateral {c['bilateral']}" for n, c in CRF_CONFIGS.items()) + f"; {CRF_ITERATIONS} iterations.",
        "",
        table("fg Dice and per-organ Dice", "dice", rows, classes, names, ".4f"),
        "",
        table("fg HD95 and per-organ HD95 (mm)", "hd95", rows, classes, names, ".2f"),
        "",
        table("fg ASSD and per-organ ASSD (mm)", "assd", rows, classes, names, ".3f"),
        "",
        recovery_table(rows, classes),
        "",
        scaled_count_table(f"Rows with HD95 above {LONG_HD95:.0f} mm, per 40 patient x fold rows", rows, classes, names,
                           lambda r: int(r["hd95"] > LONG_HD95)),
        "",
        scaled_count_table("False-positive slices, per 40 patient x fold rows", rows, classes, names, lambda r: r["fp_slices"]),
        "",
        scaled_count_table("Rows with an empty prediction (undefined HD95), per 40 patient x fold rows", rows, classes, names,
                           lambda r: int(r["pred_voxels"] == 0)),
    ]
    (args.out / "summary.md").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
