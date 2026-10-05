"""Build comparison tables from every run copied into metrics/.

    python -m src.aggregate                 # -> metrics/comparison_runs.csv, metrics/comparison.md
    python -m src.aggregate --filter enet   # only experiments whose name contains "enet"

comparison_runs.csv : one row per run (experiment x seed), all numbers, for plotting/pandas.
comparison.md       : one row per experiment, mean ± std over seeds, paste-ready for the report.

Cross-validation experiments (runs named `<experiment>_fold<k>`) also get, per `<experiment>`:
cv_summary.md                    : per fold, overall and fold-to-fold mean ± std for Dice, HD95 and ASSD, with completeness.
cv_pooled_<experiment>.csv       : every patient x organ, averaged over the seeds, with its fold.
Both are rewritten on every run from the runs that pass --filter, so a filtered run replaces the
other experiments' section in cv_summary.md. Use the unfiltered command to get all of them in one file.
"""
import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

from src.config import REPO

CV_NAME = re.compile(r"(?P<experiment>.+)_fold(?P<fold>\d+)")
CV_FOLDS, CV_SEEDS = 4, 3
CV_METRICS = (("Dice", "dice", 3), ("HD95 mm", "hd95", 1), ("ASSD mm", "assd", 2))

def report_columns(runs: list[dict]) -> list[tuple[str, str, int]]:
    """(markdown label, flattened summary key, digits); one 3D Dice column per evaluated class."""
    per_class = []
    for r in runs:
        per_class += [k for k in r if k.startswith("eval.val_dice_") and k != "eval.val_dice_fg"
                      and k not in per_class]
    return ([("val Dice 2D", "best.val_dice_fg", 3), ("3D Dice fg", "eval.val_dice_fg", 3)]
            + [(f"3D Dice {k.removeprefix('eval.val_dice_')}", k, 3) for k in per_class]
            + [("HD95 mm", "eval.val_hd95_fg", 1), ("ASSD mm", "eval.val_assd_fg", 2)])


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out |= flatten(v, f"{prefix}{k}.")
        else:
            out[f"{prefix}{k}"] = v
    return out


def load_runs(metrics_dir: Path, name_filter: str) -> list[dict]:
    runs = [flatten(json.loads(p.read_text())) for p in sorted(metrics_dir.glob("*/*/summary.json"))]
    return [r for r in runs if name_filter in r["experiment"]]


def fmt(values: list, digits: int) -> str:
    values = [v for v in values if isinstance(v, (int, float)) and v == v]  # drop missing / NaN
    if not values:
        return "–"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}"


def finite(values: list) -> list[float]:
    return [v for v in values if not math.isnan(v)]


def nan_free_mean(values: list) -> float:
    """Mean of the non-NaN values; NaN if there are none."""
    values = finite(values)
    return statistics.mean(values) if values else float("nan")


def cv_runs(runs: list[dict]) -> dict[str, dict[tuple[int, int], dict]]:
    """Evaluated cross-validation runs, as {experiment: {(fold, seed): run}}."""
    groups: dict[str, dict[tuple[int, int], dict]] = {}
    for r in runs:
        match = CV_NAME.fullmatch(r["experiment"])
        if match and "eval.val_dice_fg" in r:
            groups.setdefault(match["experiment"], {})[
                int(match["fold"]), r["seed"]
            ] = r
    return groups


def read_patient_rows(metrics_dir: Path, run: dict) -> list[dict]:
    """The per-patient, per-organ rows of a run's metrics_3d.csv, with the metrics as floats."""
    with (metrics_dir / run["experiment"] / run["run"] / "metrics_3d.csv").open() as f:
        return [
            row | {key: float(row[key]) for _, key, _ in CV_METRICS}
            for row in csv.DictReader(f)
        ]


def pool_patients(patient_rows: dict[tuple[int, int], list[dict]]) -> list[dict]:
    """One row per patient and organ with each metric averaged over the seeds, plus the patient's fold."""
    pooled: dict[tuple[str, str], dict] = {}
    for (fold, _), rows in patient_rows.items():
        for row in rows:
            entry = pooled.setdefault(
                (row["patient"], row["class_name"]),
                {
                    "patient": row["patient"],
                    "fold": fold,
                    "class_name": row["class_name"],
                    "n_seeds": 0,
                    **{key: [] for _, key, _ in CV_METRICS},
                },
            )
            entry["n_seeds"] += 1
            for _, key, _ in CV_METRICS:
                entry[key].append(row[key])
    return [
        e | {key: nan_free_mean(e[key]) for _, key, _ in CV_METRICS}
        for _, e in sorted(pooled.items())
    ]


def cv_section(
    experiment: str,
    found: dict[tuple[int, int], dict],
    patient_rows: dict[tuple[int, int], list[dict]],
) -> str:
    """Markdown for one cross-validation experiment: a table per metric, rows per fold, overall and fold-to-fold."""
    missing = [
        f"fold{f}/seed{s}"
        for f in range(CV_FOLDS)
        for s in range(CV_SEEDS)
        if (f, s) not in found
    ]
    expected = CV_FOLDS * CV_SEEDS
    status = "complete" if not missing else f"INCOMPLETE, missing {', '.join(missing)}"
    lines = [f"## {experiment}", "", f"{len(found)}/{expected} runs found: {status}"]
    first = next(iter(found.values()))
    organs = [
        k.removeprefix("eval.val_dice_")
        for k in first
        if k.startswith("eval.val_dice_")
    ]
    organs = [o for o in organs if o != "fg"]
    for label, key, digits in CV_METRICS:
        header = ["", *organs, "fg"]
        lines += [
            "",
            f"### {label}",
            "",
            "| " + " | ".join(header) + " |",
            "|" + "---|" * len(header),
        ]
        rows = {f"fold {f}": [k for k in found if k[0] == f] for f in range(CV_FOLDS)}
        rows["all runs"] = list(found)
        for name, group in rows.items():
            cells = [f"{name} ({len(group)} runs)"]
            for organ in [*organs, "fg"]:
                values = [found[k][f"eval.val_{key}_{organ}"] for k in group]
                if organ == "fg":  # organ values missing from the runs' fg means
                    nans = sum(
                        math.isnan(found[k][f"eval.val_{key}_{o}"])
                        for k in group
                        for o in organs
                    )
                else:
                    nans = sum(
                        row[key] != row[key]
                        for k in group
                        for row in patient_rows[k]
                        if row["class_name"] == organ
                    )
                cells.append(fmt(values, digits) + (f" ({nans} NaN)" if nans else ""))
            lines.append("| " + " | ".join(cells) + " |")
        fold_cells = ["std of fold means"]
        for organ in [*organs, "fg"]:
            fold_means = finite(
                [
                    nan_free_mean(
                        [found[k][f"eval.val_{key}_{organ}"] for k in rows[f"fold {f}"]]
                    )
                    for f in range(CV_FOLDS)
                ]
            )
            fold_cells.append(
                f"{statistics.stdev(fold_means):.{digits}f}"
                if len(fold_means) > 1
                else "–"
            )
        lines.append("| " + " | ".join(fold_cells) + " |")
    return "\n".join(lines)


def write_cv_results(runs: list[dict], metrics_dir: Path) -> str:
    """Writes cv_summary.md and one cv_pooled_<experiment>.csv per cross-validation experiment.

    Args:
        runs: Flattened summaries from `load_runs`.
        metrics_dir: Directory holding `<experiment>/seed<s>/` and receiving the output files.

    Returns:
        The markdown written to cv_summary.md, or "" when `runs` has no cross-validation experiment.
    """
    sections = []
    for experiment, found in sorted(cv_runs(runs).items()):
        patient_rows = {
            key: read_patient_rows(metrics_dir, r) for key, r in found.items()
        }
        pooled = pool_patients(patient_rows)
        columns = [
            "patient",
            "fold",
            "class_name",
            "n_seeds",
            *(key for _, key, _ in CV_METRICS),
        ]
        with (metrics_dir / f"cv_pooled_{experiment}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(pooled)
        sections.append(cv_section(experiment, found, patient_rows))
    if not sections:
        return ""
    note = (
        "\n\nmean ± std over the runs of a fold (seeds) or over all runs; `std of fold means` is the std of the "
        "per-fold means. Values are 3D, best checkpoint, foreground (fg) = eval.classes. NaN = patients whose "
        "HD95/ASSD is undefined (empty prediction or GT); they are left out of the mean and counted per organ. "
        "In the fg column the count is the number of organ values (run x organ) missing from the fg means. "
        "Pooled CSV: each patient's value averaged over its seeds.\n"
    )
    report = "\n\n".join(sections) + note
    (metrics_dir / "cv_summary.md").write_text(report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics-dir", type=Path, default=REPO / "metrics")
    parser.add_argument("--filter", default="", help="substring of experiment names to keep")
    args = parser.parse_args(argv)

    runs = load_runs(args.metrics_dir, args.filter)
    if not runs:
        raise SystemExit(f"no summary.json under {args.metrics_dir}/<experiment>/<seed>/")

    columns = sorted({k for r in runs for k in r}, key=lambda k: (k.count("."), k))
    with (args.metrics_dir / "comparison_runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(runs)

    by_experiment: dict[str, list[dict]] = {}
    for r in runs:
        by_experiment.setdefault(r["experiment"], []).append(r)
    report = report_columns(runs)
    header = ["experiment", "model", "loss", "seeds", *(label for label, _, _ in report)]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, group in sorted(by_experiment.items()):
        cells = [name, group[0]["model"], group[0]["loss"], str(len(group))]
        cells += [fmt([r.get(key) for r in group], digits) for _, key, digits in report]
        lines.append("| " + " | ".join(cells) + " |")
    note = ("\n\nmean ± std over seeds. 2D Dice: patient-level at 256x256 (training selection metric). "
            "3D: best checkpoint on the original CT grid, foreground = eval.classes. See docs/metrics.md.\n")
    (args.metrics_dir / "comparison.md").write_text("\n".join(lines) + note)
    print("\n".join(lines))
    print(f"\n{len(runs)} runs -> {args.metrics_dir / 'comparison_runs.csv'}, {args.metrics_dir / 'comparison.md'}")

    cv_report = write_cv_results(runs, args.metrics_dir)
    if cv_report:
        print(
            f"\n{cv_report}\ncross-validation -> {args.metrics_dir / 'cv_summary.md'}, "
            f"{args.metrics_dir / 'cv_pooled_<experiment>.csv'}"
        )


if __name__ == "__main__":
    main()
