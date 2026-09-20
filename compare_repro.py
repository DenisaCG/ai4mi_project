"""Reproduction check: original main.py runs vs standardized pipeline runs, on the metrics both produce.

Shared quantities (same reductions as tests/test_training.py::test_parity_with_legacy_main):
  legacy val Dice = dice_val[:, :, 1:].mean((1, 2))  <->  epochs.csv val_dice_legacy_fg
  train loss      = loss_tra.mean(1)                 <->  train_loss
  val loss        = loss_val.mean(1)                 <->  val_loss
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
COLORS = {"original": "#2a78d6", "pipeline": "#eb6834"}
STYLES = {"original": "-", "pipeline": "--"}
QUANTITIES = (("val_legacy", "Legacy val Dice (higher = better)"),
              ("train_loss", "Train loss"), ("val_loss", "Val loss"))


def load_original(run: Path) -> dict:
    dice = np.load(run / "dice_val.npy")
    return {"val_legacy": dice[:, :, 1:].mean((1, 2)),
            "train_loss": np.load(run / "loss_tra.npy").mean(1),
            "val_loss": np.load(run / "loss_val.npy").mean(1),
            "per_class": dice.mean(1)}  # (epochs, K), per-slice smoothed Dice


def load_pipeline(run: Path, class_names: list[str]) -> dict:
    with open(run / "epochs.csv") as f:
        rows = list(csv.DictReader(f))
    col = lambda name: np.array([float(r[name]) for r in rows])
    return {"val_legacy": col("val_dice_legacy_fg"), "train_loss": col("train_loss"), "val_loss": col("val_loss"),
            "per_class": np.stack([col(f"val_dice_{c}") for c in class_names], axis=1),
            "best_epoch": json.loads((run / "summary.json").read_text())["best_epoch"]}


def band(runs: list[dict], key: str):
    a = np.stack([r[key] for r in runs])
    return a.mean(0), a.min(0), a.max(0)


def rng(x) -> str:
    return f"{np.mean(x):.3f} [{np.min(x):.3f}, {np.max(x):.3f}]"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", type=Path, nargs="+", required=True, help="results/.../run0 run1 ...")
    p.add_argument("--pipeline", type=Path, nargs="+", required=True, help="runs/<experiment>/seed0 seed1 ...")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--class-names", nargs="+", default=["esophagus", "heart", "trachea", "aorta"])
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    orig = [load_original(r) for r in a.original]
    pipe = [load_pipeline(r, a.class_names) for r in a.pipeline]
    epochs = len(orig[0]["val_legacy"])
    assert all(len(r["val_legacy"]) == epochs for r in orig + pipe), "runs have different numbers of epochs"

    lines = [f"# Reproduction check: {len(orig)} original runs vs {len(pipe)} pipeline runs, {epochs} epochs",
             "", f"original: {', '.join(map(str, a.original))}", f"pipeline: {', '.join(map(str, a.pipeline))}", "",
             "Cells are `mean [min, max]` over runs. Legacy Dice is the course metric (per-slice, smoothed, all 4 foreground classes).", ""]

    lines += ["## Summary", "", "| quantity | original | pipeline | pipeline mean inside original [min,max] |", "|---|---|---|---|"]
    o_best = [int(np.argmax(r["val_legacy"])) for r in orig]
    for key, label in QUANTITIES:
        om, olo, ohi = band(orig, key)
        pm, _, _ = band(pipe, key)
        inside = np.mean((pm >= olo) & (pm <= ohi))
        lines.append(f"| {label}, final epoch | {rng([r[key][-1] for r in orig])} | {rng([r[key][-1] for r in pipe])} | "
                     f"{'yes' if olo[-1] <= pm[-1] <= ohi[-1] else 'no'} |")
        lines.append(f"| {label}, mean over all epochs | {rng([r[key].mean() for r in orig])} | {rng([r[key].mean() for r in pipe])} | "
                     f"{100 * inside:.0f}% of epochs |")
    lines.append(f"| Legacy val Dice, best over epochs | {rng([r['val_legacy'].max() for r in orig])} | "
                 f"{rng([r['val_legacy'].max() for r in pipe])} | n/a |")
    lines.append(f"| Legacy val Dice at each run's own selected epoch | {rng([r['val_legacy'][e] for r, e in zip(orig, o_best)])} "
                 f"(original picks the legacy argmax) | {rng([r['val_legacy'][r['best_epoch']] for r in pipe])} "
                 f"(pipeline picks by val_dice_fg) | n/a |")
    lines += ["", f"selected epochs: original {o_best}, pipeline {[r['best_epoch'] for r in pipe]}", ""]

    lines += ["## Per class at the final epoch (different definitions: not comparable across columns)", "",
              "| class | original: legacy per-slice Dice (empty/empty = 1) | pipeline: patient-level Dice |", "|---|---|---|"]
    for k, name in enumerate(a.class_names, start=1):
        lines.append(f"| {name} | {rng([r['per_class'][-1, k] for r in orig])} | {rng([r['per_class'][-1, k - 1] for r in pipe])} |")

    x = np.arange(epochs)

    def draw(ax, key, label, legend):
        ax.set_facecolor(SURFACE)
        for name, runs in (("original", orig), ("pipeline", pipe)):
            m, lo, hi = band(runs, key)
            ax.fill_between(x, lo, hi, color=COLORS[name], alpha=0.16, linewidth=0)
            ax.plot(x, m, STYLES[name], color=COLORS[name], linewidth=2, label=f"{name} (n={len(runs)})")
        ax.set_title(label, color=INK, fontsize=12, loc="left")
        ax.set_xlabel("epoch", color=INK_2)
        ax.grid(axis="y", color="#e7e7e7", linewidth=0.7)
        ax.tick_params(colors=INK_2, length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#999999")
        if "loss" in key:
            ax.set_yscale("log")
        if legend:
            ax.legend(frameon=False, labelcolor=INK, loc="upper right" if "loss" in key else "lower right")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)
    for ax, (key, label) in zip(axes, QUANTITIES):
        draw(ax, key, label, legend=key == "val_legacy")
    fig.suptitle("Original main.py vs standardized pipeline: line = mean, band = min-max over runs", color=INK_2, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(a.out / "repro_curves.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    for key, label in QUANTITIES:
        fig, ax = plt.subplots(figsize=(5.2, 4.2), facecolor=SURFACE)
        draw(ax, key, label, legend=True)
        fig.tight_layout()
        fig.savefig(a.out / f"repro_{key}.png", dpi=160, facecolor=SURFACE)
        plt.close(fig)

    (a.out / "repro_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {a.out}/repro_summary.md and repro_curves.png")


if __name__ == "__main__":
    main()
