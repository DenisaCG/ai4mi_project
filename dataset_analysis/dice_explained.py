"""Slide figure: the same predictions scored with the legacy per-slice Dice and with patient-level Dice."""
import argparse
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from utils import BACKGROUND, COLORS, INK, MUTED, NAMES, decorate, pyplot, read_csv, tint

EMPTY, FALSE_POSITIVE = BACKGROUND, INK
SMOOTH = 1e-8  # the course dice_coef smoothing: both empty -> 1.0


def load_slices(analysis_dir):
    """{(patient, class): (gt, pred, intersection)} as pixel-count arrays ordered by slice."""
    rows = {}
    for r in read_csv(analysis_dir / "tables/baseline_slice_metrics.csv"):
        rows.setdefault((r["patient_id"], int(r["class_id"])), []).append(
            (int(r["slice_index"]), int(r["gt_area"]), int(r["pred_area"]), int(r["intersection"])))
    return {key: tuple(np.array(sorted(rs))[:, 1:].T) for key, rs in rows.items()}


def legacy_dice(g, p, i):
    """Course metric per slice: smoothed Dice, both empty scores 1."""
    return (2 * i + SMOOTH) / (g + p + SMOOTH)


def patient_dice(g, p, i):
    """One Dice from all of a patient's slices; undefined if the organ is absent from the ground truth."""
    return 2 * i.sum() / (g.sum() + p.sum()) if g.sum() > 0 else np.nan


def slice_strip(g, p, i, ramp):
    """One colour per slice: sand = empty in both, dark = false alarm, ramp = Dice where the organ is present."""
    row = np.zeros((1, len(g), 3))
    for n in range(len(g)):
        if g[n] == 0 and p[n] == 0:
            row[0, n] = to_rgb(EMPTY)
        elif g[n] == 0:
            row[0, n] = to_rgb(FALSE_POSITIVE)
        else:
            row[0, n] = ramp(2 * i[n] / (g[n] + p[n]))[:3]
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis-dir", type=Path, required=True, help="analysis folder with tables/baseline_slice_metrics.csv")
    ap.add_argument("--output", type=Path, required=True, help="PNG to write")
    ap.add_argument("--organ", type=int, default=1, help="class id shown slice by slice")
    ap.add_argument("--run-label", required=True, help="text naming the run, e.g. 'corrected labels, median run (seed 7)'")
    args = ap.parse_args()

    slices = load_slices(args.analysis_dir)
    patients = sorted({p for p, _ in slices})
    k = args.organ
    n_slices = sum(len(slices[p, k][0]) for p in patients)
    plt = pyplot()
    fig = plt.figure(figsize=(13, 6.8))
    ax, bx = fig.subplots(1, 2, gridspec_kw=dict(width_ratios=[1.3, 1]))

    # Left: one organ, every slice of every patient.
    ramp = LinearSegmentedColormap.from_list("organ", [tint(COLORS[k], .12), COLORS[k]])
    for row, patient in enumerate(patients):
        g, p, i = slices[patient, k]
        ax.imshow(slice_strip(g, p, i, ramp), extent=(0, 1, row + .4, row - .4), aspect="auto", interpolation="nearest")
        slice_avg = legacy_dice(g, p, i).mean()
        ax.text(1.04, row, f"{slice_avg:.2f}", transform=ax.get_yaxis_transform(), va="center", fontsize=12, color=INK)
        ax.text(1.26, row, f"{patient_dice(g, p, i):.2f}", transform=ax.get_yaxis_transform(), va="center",
                fontsize=12, weight="bold", color=INK)
    ax.text(1.04, -.75, "Slice\naverage", transform=ax.get_yaxis_transform(), va="bottom", fontsize=10, color=MUTED)
    ax.text(1.26, -.75, "Patient\nDice", transform=ax.get_yaxis_transform(), va="bottom", fontsize=10, color=MUTED, weight="bold")
    ax.set(xlim=(0, 1), ylim=(len(patients) - .5, -.5), yticks=range(len(patients)),
           yticklabels=[p.replace("Patient_", "P") for p in patients], xticks=[0, 1], xticklabels=["first slice", "last slice"])
    ax.grid(False)
    ax.tick_params(length=0, pad=6)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(f"{NAMES[k]}: every slice of every patient", fontsize=13, weight="bold", loc="left", pad=34)
    ax.legend(handles=[Patch(facecolor=EMPTY, label="organ absent, none predicted: legacy scores 1.0"),
                       Patch(facecolor=FALSE_POSITIVE, label="organ absent, but predicted: 0"),
                       Patch(facecolor=COLORS[k], label="organ present: that slice's Dice (light = 0, dark = 1)")],
              loc="upper left", bbox_to_anchor=(-.12, -.09), ncol=1, fontsize=10)

    # Right: all organs, both scores, plus the score of a model that predicts nothing.
    for c in sorted(NAMES):
        gs, ps, is_ = (np.concatenate(v) for v in zip(*(slices[p, c] for p in patients)))
        legacy = legacy_dice(gs, ps, is_).mean()
        patient = np.nanmean([patient_dice(*slices[p, c]) for p in patients])
        nothing = (gs == 0).mean()  # legacy score of an all-background prediction
        bx.bar(c - .2, legacy, width=.36, facecolor="white", edgecolor=COLORS[c], linewidth=2)
        bx.bar(c + .2, patient, width=.36, color=COLORS[c])
        bx.text(c - .2, legacy + .02, f"{legacy:.2f}", ha="center", fontsize=10, color=INK)
        bx.text(c + .2, patient + .02, f"{patient:.2f}", ha="center", fontsize=10, weight="bold", color=INK)
        bx.hlines(nothing, c - .40, c - .06, color=INK, linewidth=2, linestyles=(0, (3, 2)), zorder=4)
    bx.set(ylim=(0, 1.08), xlim=(.4, max(NAMES) + .6), yticks=np.arange(0, 1.01, .2), xticks=list(NAMES),
           xticklabels=list(NAMES.values()))
    bx.set_axisbelow(True)
    bx.tick_params(length=0, pad=6)
    bx.set_title("Same run, all organs", fontsize=13, weight="bold", loc="left", pad=34)
    bx.legend(handles=[Patch(facecolor="white", edgecolor=MUTED, linewidth=2, label="Legacy: average over all slices"),
                       Patch(facecolor=MUTED, label="Patient-level: one Dice per patient, then average"),
                       Line2D([], [], color=INK, linewidth=2, linestyle=(0, (3, 2)), label="Legacy score of a model that predicts nothing")],
              loc="upper left", bbox_to_anchor=(-.05, -.09), ncol=1, fontsize=10)

    decorate(fig, "Same predictions, two Dice scores",
             "45–78% of slices contain no organ: the legacy score counts them as perfect, the patient-level score does not  |  "
             f"{args.run_label}  |  {len(patients)} validation patients, {n_slices} slices per organ  |  scored slice by slice on the 256 × 256 grid",
             "Legacy (course) metric: Dice of every slice, averaged over all slices, so empty slices count. "
             "Patient-level metric: per patient, add up overlap and volume over all slices, compute one Dice, then average the patients.")
    # The per-axes legends hang below the plots: make room for them above the footnote.
    fig.subplots_adjust(bottom=fig.subplotpars.bottom + .045, wspace=.75)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(f"written -> {args.output}")


if __name__ == "__main__":
    main()
