"""Table cells (fg Dice, per-organ Dice, fg HD95, fg ASSD) of variants in a rows.csv written by `tools.prob_postprocess score`.

    python -m src.aggregate_pp_tta results/pp_tta_ctrl avg avg_lcc_heart_trachea

Pooled with tools.postprocess_eval.aggregate, as the summary.md of results/pp_*: per run the organ mean over patients, fg the
mean of the organs, then the mean over runs (here one "run" per fold: the seeds' probabilities are averaged first).
"""
import argparse
from pathlib import Path

from tools.postprocess_eval import aggregate
from tools.prob_postprocess import read_rows


def cells(rows: list[dict], variant: str, classes: list[int], names: list[str]) -> dict[str, float]:
    rows = [r for r in rows if r["variant"] == variant and r["class_idx"] in classes]
    if not rows:
        raise ValueError(f"no rows of variant {variant!r}")
    dice, hd95, assd = (aggregate(rows, m, classes)[variant] for m in ("dice", "hd95", "assd"))
    return {"fg_dice": dice["fg", "all"], **{f"{names[k]}_dice": dice[k, "all"] for k in classes},
            "fg_hd95": hd95["fg", "all"], "fg_assd": assd["fg", "all"]}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir", type=Path)
    p.add_argument("variants", nargs="+")
    p.add_argument("--classes", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--names", nargs="+", default=["background", "esophagus", "heart", "trachea", "aorta"])
    args = p.parse_args(argv)
    rows = read_rows(args.out_dir / "rows.csv", tuple(args.variants))
    for v in args.variants:
        print(v, {k: round(x, 4) for k, x in cells(rows, v, args.classes, args.names).items()})


if __name__ == "__main__":
    main()
