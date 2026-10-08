"""Majority vote over the seeds of each CV fold: what does ensembling the finished runs' stitched 3D
predictions do, compared with the mean of the single seeds?

    python -m tools.seed_ensemble --experiment full_cv4_resenc_ds_unet_dice_ce --seeds 0 1 2

Per fold and patient, the label volumes of the seeds (runs/<experiment>_fold<k>/seed<s>/volumes/val) are
combined by tools.postprocess_filters.majority_vote and scored against the source GT with src.metrics_3d.
Variants (single-seed rows are scored per run, ensemble rows once per fold, as run <fold>/vote):
  baseline                      each seed as is (must reproduce its eval/metrics_3d.csv)
  gate                          oracle: organ set to background on slices where the GT has none
  lcc_heart_trachea             largest component for heart and trachea
  seed_vote                     the majority vote
  seed_vote_lcc_heart_trachea   the vote, then largest component for heart and trachea
  vote_t<t>                     an organ voxel needs at least t seeds (1 = union, 3 = intersection)
  vote_per_organ[_hd95]         per organ the t with the best mean Dice (or lowest HD95 among the t losing at
                                most 0.002 Dice against the majority t); _lcc_heart_trachea adds that filter
Fold means are over the seeds for the single-seed variants and the fold's one ensemble otherwise, so
the fg means are comparable. Writes rows.csv, means.csv and summary.md into --out.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

from src.config import REPO, read_yaml
from tools.gating_oracle import (
    HD95_DICE_TOLERANCE,
    aggregate,
    check_baseline,
    gate,
    largest_component,
    means_rows,
    recovery_table,
    run_dirs,
    table,
    variants_of,
    volume_rows,
    write_csv,
)
from tools.postprocess_filters import majority_vote, take_organs, threshold_vote

LONG_HD95 = 20.0  # mm
ROWS_PER_ORGAN = 40  # patient x fold rows of one organ in the ensemble: counts are scaled to this


def choose_thresholds(
    rows: list[dict], classes: list[int], n_seeds: int, by: str = "dice"
) -> dict[int, int]:
    """Per organ, the vote threshold t (rows variant vote_t<t>) with the best mean Dice over the folds;
    by="hd95": the lowest mean HD95 among the t that lose at most HD95_DICE_TOLERANCE Dice against the
    majority threshold. The majority threshold wins ties."""
    ts = range(1, n_seeds + 1)
    voted = [r for r in rows if r["variant"].startswith("vote_t")]
    dice, hd95 = aggregate(voted, "dice", classes), aggregate(voted, "hd95", classes)
    majority = (n_seeds + 1) // 2
    choices = {}
    for k in classes:
        best = majority
        for t in ts:
            d, h = dice[f"vote_t{t}"][k, "all"], hd95[f"vote_t{t}"][k, "all"]
            if by == "dice":
                better = d > dice[f"vote_t{best}"][k, "all"]
            else:
                ok = d >= dice[f"vote_t{majority}"][k, "all"] - HD95_DICE_TOLERANCE
                better = ok and h < hd95[f"vote_t{best}"][k, "all"]
            if better:
                best = t
        choices[k] = best
    return choices


def score_patient(
    task: tuple[list[str], str, str, list[str], list[int], dict[str, dict[int, int]] | None],
) -> list[dict]:
    """Rows of every variant for one patient of one fold: the seeds' runs, then the ensembles. With
    `choices` ({name: {organ: threshold}}) only those per-organ threshold votes (and their
    _lcc_heart_trachea versions) are scored."""
    runs, pattern, patient, names, classes, choices = task
    ref = nib.load(REPO / pattern.format(patient=patient))
    gt = np.asarray(ref.dataobj)
    spacing = tuple(float(s) for s in ref.header.get_zooms()[:3])
    heart, trachea = names.index("heart"), names.index("trachea")
    preds = [
        np.asarray(nib.load(Path(r) / "volumes" / "val" / f"{patient}.nii.gz").dataobj)
        for r in runs
    ]

    def lcc_ht(vol: np.ndarray) -> np.ndarray:
        return take_organs(vol, largest_component(vol, [heart, trachea]), [heart, trachea])

    rows = []
    if choices is None:
        for run, pred in zip(runs, preds):
            scored: dict = {}
            for variant, vol in (
                ("baseline", pred),
                ("gate", gate(pred, gt, classes)),
                ("lcc_heart_trachea", lcc_ht(pred)),
            ):
                rows += volume_rows(Path(run), patient, variant, vol, gt, spacing, names, classes, scored)
        votes = [("seed_vote", majority_vote(preds))]
        votes.append(("seed_vote_lcc_heart_trachea", lcc_ht(votes[0][1])))
        votes += [
            (f"vote_t{t}", threshold_vote(preds, classes, dict.fromkeys(classes, t)))
            for t in range(1, len(preds) + 1)
        ]
    else:
        votes = []
        for name, chosen in choices.items():
            vol = threshold_vote(preds, classes, chosen)
            votes += [(name, vol), (f"{name}_lcc_heart_trachea", lcc_ht(vol))]
    vote_run = Path(runs[0]).parent / "vote"
    scored = {}
    for variant, vol in votes:
        rows += volume_rows(vote_run, patient, variant, vol, gt, spacing, names, classes, scored)
    return rows


def scaled_count_table(
    title: str, rows: list[dict], classes: list[int], names: list[str], count
) -> str:
    """Per variant and organ the sum of `count(row)`, scaled to ROWS_PER_ORGAN rows (a single-seed
    variant has three times as many rows as the ensemble)."""
    head = ["variant", *(names[k] for k in classes), "all"]
    lines = [f"### {title}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for variant in variants_of(rows):
        cells = []
        for k in classes:
            sel = [r for r in rows if r["variant"] == variant and r["class_idx"] == k]
            cells.append(sum(count(r) for r in sel) * ROWS_PER_ORGAN / len(sel))
        lines.append(f"| {variant} | " + " | ".join(f"{c:.1f}" for c in [*cells, sum(cells)]) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, default=REPO / "results" / "seed_ensemble")
    args = parser.parse_args(argv)

    dirs = run_dirs(args.experiment, args.folds, args.seeds)
    cfg = read_yaml((dirs[0] / "config.yaml").read_text())
    names, classes = cfg["data"]["class_names"], cfg["eval"]["classes"]
    pattern = cfg["data"]["source_pattern"]
    fold_tasks = []
    for fold in range(args.folds):
        fold_runs = [str(d) for d in dirs if d.parent.name.endswith(f"_fold{fold}")]
        patients = sorted(
            p.name.removesuffix(".nii.gz")
            for p in (Path(fold_runs[0]) / "volumes" / "val").glob("*.nii.gz")
        )
        fold_tasks += [(fold_runs, pattern, p, names, classes) for p in patients]

    def run_all(choices: dict[str, dict[int, int]] | None) -> list[dict]:
        with ProcessPoolExecutor(args.workers) as pool:
            parts = pool.map(score_patient, [(*t, choices) for t in fold_tasks])
            return [row for part in parts for row in part]

    rows = run_all(None)
    choices = {
        "vote_per_organ": choose_thresholds(rows, classes, len(args.seeds)),
        "vote_per_organ_hd95": choose_thresholds(rows, classes, len(args.seeds), "hd95"),
    }
    rows += run_all(choices)
    tasks = fold_tasks

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "rows.csv", rows)
    write_csv(args.out / "means.csv", means_rows(rows, classes, names))
    worst = check_baseline(rows, dirs)
    summary = [
        f"# Seed majority vote: {args.experiment}, {len(args.seeds)} seeds, {args.folds} folds, {len(tasks)} patients",
        "",
        f"Baseline rows vs the runs' own metrics_3d.csv: max |difference| {worst:.2e} (must be ~0).",
        "Single-seed variants: mean over the seeds of each fold; seed_vote*: the fold's one ensemble. fg and fold columns are comparable; gate is the oracle.",
        "Ties between three different labels go to seed 0's label.",
        "",
        table("fg Dice and per-organ Dice", "dice", rows, classes, names, ".4f"),
        "",
        table("fg HD95 and per-organ HD95 (mm)", "hd95", rows, classes, names, ".2f"),
        "",
        table("fg ASSD and per-organ ASSD (mm)", "assd", rows, classes, names, ".3f"),
        "",
        recovery_table(rows, classes),
        "",
        "### Vote threshold chosen per organ (seeds needed to keep a voxel; picked on the scored folds)",
        "",
        "| choice | " + " | ".join(names[k] for k in classes) + " |",
        "|" + "---|" * (len(classes) + 1),
        *(
            f"| {name} | " + " | ".join(str(chosen[k]) for k in classes) + " |"
            for name, chosen in choices.items()
        ),
        "",
        scaled_count_table(
            f"Rows with HD95 above {LONG_HD95:.0f} mm, per {ROWS_PER_ORGAN} patient x fold rows",
            rows, classes, names, lambda r: int(r["hd95"] > LONG_HD95),
        ),
        "",
        scaled_count_table(
            f"False-positive slices (organ predicted where the GT has none), per {ROWS_PER_ORGAN} patient x fold rows",
            rows, classes, names, lambda r: r["fp_slices"],
        ),
        "",
        scaled_count_table(
            f"Rows with an empty prediction (undefined HD95), per {ROWS_PER_ORGAN} patient x fold rows",
            rows, classes, names, lambda r: int(r["pred_voxels"] == 0),
        ),
    ]
    (args.out / "summary.md").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
