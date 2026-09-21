"""Before/after comparison of the SEGTHOR label correction, from two finished analysis folders."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from matplotlib.ticker import MaxNLocator

from figures import orthogonal_plane
from utils import BACKGROUND, COLORS, INK, MUTED, NAMES, clean_axis, decorate, pyplot, read_csv, tint, write_csv
from plot_style import legend_below

BEFORE = "Before: original labels"
AFTER = "After: corrected labels"
# (column in eval/metrics_3d.csv, panel title, reading direction, mean-label format)
METRICS = (("dice", "Dice", "higher is better", "{:.2f}"),
           ("hd95", "HD95 (mm)", "lower is better", "{:.0f}"),
           ("assd", "ASSD (mm)", "lower is better", "{:.1f}"))


def label_changes(original_data, corrected_data):
    """Per-patient voxel bookkeeping: did the correction only split old label 1 into esophagus + aorta?"""
    import nibabel as nib
    rows = []
    for corrected_path in sorted(corrected_data.glob("Patient_*/GT.nii.gz")):
        patient = corrected_path.parent.name
        before_nii = nib.load(original_data / patient / "GT.nii.gz")
        after_nii = nib.load(corrected_path)
        if before_nii.shape != after_nii.shape or not np.allclose(before_nii.affine, after_nii.affine):
            raise ValueError(f"Original and corrected GT are not on the same grid: {patient}")
        before, after = np.asanyarray(before_nii.dataobj), np.asanyarray(after_nii.dataobj)
        old_label_1 = before == 1
        changed = before != after
        rows.append({"patient_id": patient,
                     "esophagus_voxels_before": int(old_label_1.sum()),
                     "esophagus_voxels_after": int((after == 1).sum()),
                     "aorta_voxels_after": int((after == 4).sum()),
                     "changed_voxels": int(changed.sum()),
                     "changed_outside_old_label_1": int((changed & ~old_label_1).sum()),
                     "old_label_1_not_new_1_or_4": int((old_label_1 != np.isin(after, (1, 4))).sum())})
    if not rows:
        raise FileNotFoundError(f"No Patient_*/GT.nii.gz under {corrected_data}")
    return rows


def class_counts(analysis_dir):
    """Pooled (voxel count, share of annotated foreground) per organ from an analysis folder."""
    rows = {int(r["class_id"]): r for r in read_csv(analysis_dir / "tables/class_frequency_original.csv")
            if r["split"] == "all"}
    return {k: (int(rows[k]["voxel_count"]), float(rows[k]["fraction_foreground_voxels"] or 0)) if k in rows
            else (0, 0.0) for k in NAMES}


def run_dice(run_dirs, annotated):
    """{class_id: [one mean 3D Dice per run, over the validation patients]} from each run's eval summary.

    Organs without ground truth in the run's label version are left empty, not scored as zero.
    """
    scores = {k: [] for k in NAMES}
    for run in run_dirs:
        evaluation = json.loads((run / "summary.json").read_text()).get("eval", {})
        for k, name in NAMES.items():
            value = evaluation.get(f"val_dice_{name.lower()}")
            if annotated[k] and value is not None and np.isfinite(value):
                scores[k].append(value)
    return scores


def legend_before_after(ax, marker=False):
    """Queue the before/after key; hollow means before, filled means after."""
    shape = "o" if marker else "s"
    ax.plot([], [], shape, markersize=8, markerfacecolor="white", markeredgecolor=MUTED, markeredgewidth=2, label=BEFORE)
    ax.plot([], [], shape, markersize=8, color=MUTED, label=AFTER)
    legend_below(ax, ncol=2)


def paired_bars(ax, data, fmt, xlabel):
    ks = sorted(NAMES)
    top = max(max(pair) for pair in data.values())
    for y, k in enumerate(ks):
        before, after = data[k]
        ax.barh(y - .19, before, height=.34, facecolor="white", edgecolor=COLORS[k], linewidth=1.8)
        ax.barh(y + .19, after, height=.34, color=COLORS[k])
        ax.text(before + .02 * top, y - .19, fmt(before) if before else "not annotated",
                va="center", fontsize=10, color=INK)
        ax.text(after + .02 * top, y + .19, fmt(after), va="center", fontsize=10, color=INK, weight="bold")
    ax.set(xlim=(0, top * 1.3), ylim=(len(ks) - .5, -.5), yticks=range(len(ks)),
           yticklabels=[NAMES[k] for k in ks], xlabel=xlabel)
    ax.spines["left"].set_visible(False)
    clean_axis(ax, "x")


def class_balance(plt, out, before, after, changes):
    ratio = (sum(r["esophagus_voxels_before"] for r in changes)
             / sum(r["esophagus_voxels_after"] for r in changes))
    fig = plt.figure(figsize=(12, 5.6))
    left, right = fig.subplots(1, 2, sharey=True)
    paired_bars(left, {k: (100 * before[k][1], 100 * after[k][1]) for k in NAMES},
                lambda v: f"{v:.1f}%", "Share of annotated foreground voxels (%)")
    paired_bars(right, {k: (before[k][0] / 1e6, after[k][0] / 1e6) for k in NAMES},
                lambda v: f"{v:.2f} M", "Annotated voxels (millions)")
    right.tick_params(labelleft=False)
    legend_before_after(left)
    decorate(fig, "Label correction: what changed in the ground truth",
        f"Original vs corrected NIfTI ground truth  |  {len(changes)} patients pooled  |  Aorta was not annotated before",
        "Left: each organ's share of annotated foreground voxels (background excluded). Right: annotated voxels per organ. "
        f"The original label 1 covered esophagus and aorta together, so the esophagus ground truth was {ratio:.1f}x larger than the corrected one.")
    fig.savefig(out / "plots/before_after_class_balance.png")
    plt.close(fig)


def dice_comparison(plt, out, before, after):
    n_before, n_after = max(map(len, before.values())), max(map(len, after.values()))
    fig = plt.figure(figsize=(10, 6.4))
    ax = fig.add_subplot(111)
    rows = []
    for k in sorted(NAMES):
        means = {}
        for x, values, filled in ((k - .17, before[k], False), (k + .17, after[k], True)):
            offsets = np.linspace(-.09, .09, len(values)) if len(values) > 1 else [0]
            for offset, value in zip(offsets, values):
                ax.scatter(x + offset, value, s=60, zorder=3,
                           **(dict(color=COLORS[k], edgecolor="white", linewidth=1) if filled
                              else dict(facecolor="white", edgecolor=COLORS[k], linewidth=1.8)))
            if values:
                ax.hlines(np.mean(values), x - .13, x + .13, color=INK, linewidth=2.5, zorder=4)
            zeros = sum(v == 0 for v in values)
            if zeros:
                ax.text(x, .075, f"{zeros} of {len(values)}\nat 0", ha="center", va="bottom", fontsize=9, color=MUTED)
            means[filled] = np.mean(values) if values else None
        ax.text(k, 1.06, f"{'–' if means[False] is None else f'{means[False]:.2f}'} → "
                         f"{'–' if means[True] is None else f'{means[True]:.2f}'}",
                ha="center", fontsize=13, weight="bold")
        if not before[k]:
            ax.text(k - .17, .5, "no ground\ntruth", ha="center", va="center", fontsize=9, color=MUTED)
        rows.append({"class_id": k, "class_name": NAMES[k],
                     "n_runs_before": len(before[k]), "mean_dice_3d_before": means[False],
                     "min_before": min(before[k], default=None), "max_before": max(before[k], default=None),
                     "runs_at_zero_before": sum(v == 0 for v in before[k]),
                     "n_runs_after": len(after[k]), "mean_dice_3d_after": means[True],
                     "min_after": min(after[k], default=None), "max_after": max(after[k], default=None),
                     "runs_at_zero_after": sum(v == 0 for v in after[k])})
    ax.set(xlim=(.4, max(NAMES) + .6), ylim=(-.04, 1.14), ylabel="Run-level 3D Dice (0–1)")
    ax.set_xticks(list(NAMES), list(NAMES.values()))
    ax.set_yticks(np.arange(0, 1.01, .2))
    clean_axis(ax)
    legend_before_after(ax, marker=True)
    decorate(fig, "Label correction: effect on 3D segmentation quality",
        f"Same ENet + cross-entropy recipe  |  one dot = one training run  |  {n_before} original-label vs {n_after} corrected-label runs",
        "Each dot is that run's mean 3D Dice over the 5 validation patients (reconstructed predictions vs original-grid ground truth). "
        "Black bars mark the mean over runs; the numbers on top read before → after. A run at exactly 0 predicted no voxels of that organ. "
        "Esophagus is not like-for-like: its old ground truth also contained the aorta.")
    fig.savefig(out / "plots/before_after_dice_3d.png")
    plt.close(fig)
    return rows


def run_metrics(run):
    """{class_id: {"annotated": bool, metric: [one value per validation patient]}} from a run's 3D evaluation.

    Undefined values (empty prediction or empty ground truth) are NaN; an organ with no ground truth
    voxels in any patient is marked not annotated.
    """
    rows = read_csv(run / "eval/metrics_3d.csv")
    result = {}
    for k in NAMES:
        rs = [r for r in rows if int(r["class_idx"]) == k]
        result[k] = {"annotated": any(int(r["gt_voxels"]) for r in rs),
                     **{m: [float(r[m]) for r in rs] for m, *_ in METRICS}}
    return result


def finite(values):
    a = np.asarray(values, float)
    return a[np.isfinite(a)]


def swarm_levels(values, gap):
    """Row level (0, 1, -1, 2, ...) per value, so dots closer than `gap` on the x axis never overlap."""
    n = len(values)
    order = sorted(range(-n, n + 1), key=lambda level: (abs(level), -level))
    levels = []
    for i, v in enumerate(values):
        levels.append(next(level for level in order
                           if all(level != levels[j] or abs(v - values[j]) >= gap for j in range(i))))
    return levels


def slug(label):
    return re.sub(r"\W+", "_", label.lower()).strip("_")


def glow_star(ax, x, y, color):
    """A star in the organ colour with a soft halo (a few large, faint discs behind it)."""
    for size, alpha in ((1000, .07), (640, .12), (380, .20)):
        ax.scatter(x, y, s=size, marker="o", color=color, alpha=alpha, linewidths=0, zorder=5, clip_on=False)
    ax.scatter(x, y, s=420, marker="*", color=color, edgecolor="white", linewidth=1.3, zorder=6, clip_on=False)


def metrics_by_organ(plt, out, before_run, after_run, nnunet=None):
    """Dice, HD95 and ASSD for every organ: one dot per validation patient (after), diamond = mean before.

    `nnunet` maps a legend label to a folder with eval/metrics_3d.csv; each gets a glowing star at its mean,
    on its own vertical lane so variants with near-identical means do not sit on top of each other.
    """
    before, after = run_metrics(before_run), run_metrics(after_run)
    nnunet = {label: run_metrics(run) for label, run in (nnunet or {}).items()}
    lanes = np.linspace(.38, -.38, len(nnunet)) if len(nnunet) > 1 else [0.]
    ks = sorted(NAMES)
    n_patients = max(len(after[k]["dice"]) for k in ks)
    fig = plt.figure(figsize=(14, 8.2))
    axes = fig.subplots(len(ks), len(METRICS), sharex="col")
    rows, dropped = [], 0
    for j, (metric, title, direction, fmt) in enumerate(METRICS):
        shown = ([finite(after[k][metric]) for k in ks] + [finite(before[k][metric]) for k in ks if before[k]["annotated"]]
                 + [finite(n[k][metric]) for n in nnunet.values() for k in ks])
        top = 1.0 if metric == "dice" else max([float(a.max()) for a in shown if len(a)] or [1.0])
        span = 1.09 * top
        axes[-1][j].set_xlim(-.04 * top, 1.05 * top)
        if metric == "dice":
            axes[-1][j].set_xticks(np.arange(0, 1.01, .2))
        else:
            axes[-1][j].xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10], min_n_ticks=3))
        for i, k in enumerate(ks):
            ax = axes[i][j]
            values = np.sort(finite(after[k][metric]))
            dropped += len(after[k][metric]) - len(values)
            mean = float(values.mean()) if len(values) else None
            before_values = finite(before[k][metric]) if before[k]["annotated"] else []
            before_mean = float(np.mean(before_values)) if len(before_values) else None
            if before[k]["annotated"]:
                dropped += len(before[k][metric]) - len(before_values)
            ax.scatter(values, [.26 * level for level in swarm_levels(values, .055 * span)], s=110,
                       color=COLORS[k], edgecolor="white", linewidth=1.2, zorder=3, clip_on=False)
            if mean is not None:
                ax.vlines(mean, -.55, .55, color=INK, linewidth=3.5, zorder=4)
                ax.text(mean, .72, fmt.format(mean), ha="center", va="bottom", fontsize=14, weight="bold", color=INK)
            if before_mean is not None:
                ax.scatter(before_mean, 0, s=110, marker="D", facecolor="none", edgecolor=MUTED,
                           linewidth=2.2, zorder=5)
                if i == 0 and j == 0:
                    ax.annotate("before correction", (before_mean, 0), xytext=(12, 0), textcoords="offset points",
                                ha="left", va="center", fontsize=10, color=MUTED)
            nnunet_means = {}
            for lane, (label, scores) in zip(lanes, nnunet.items()):
                scored = finite(scores[k][metric])
                if not len(scored):
                    continue
                nnunet_means[label] = float(scored.mean())
                glow_star(ax, nnunet_means[label], lane, COLORS[k])
                if i == 0 and j == 0:  # name the stars once, in the first panel, like the diamond
                    left, right = axes[-1][j].get_xlim()
                    above = lane >= 0
                    ax.annotate(label, (nnunet_means[label], lane), xytext=(0, 17 if above else -17),
                                textcoords="offset points", va="bottom" if above else "top", fontsize=10, color=MUTED,
                                ha="right" if (nnunet_means[label] - left) / (right - left) > .7 else "center")
            if j == 0 and not before[k]["annotated"]:
                ax.text(.02, .08, f"no {NAMES[k].lower()} label before correction", transform=ax.transAxes,
                        ha="left", va="bottom", fontsize=10, color=MUTED)
            ax.set_ylim(-1, 1.35)
            ax.set_yticks([])
            clean_axis(ax, "x")
            ax.spines["left"].set_visible(False)
            if i < len(ks) - 1:
                ax.spines["bottom"].set_visible(False)
            if i % 2 == 0:
                ax.set_facecolor(tint(BACKGROUND, .3))
            if j == 0:
                ax.set_ylabel(NAMES[k], color=COLORS[k], fontsize=17, fontweight="bold",
                              rotation=0, ha="right", va="center", labelpad=16)
            if i == 0:
                ax.set_title(title, fontsize=17, fontweight="bold", pad=26)
                ax.text(.5, 1.03, direction, transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=10.5, color=MUTED)
            rows.append({"class_id": k, "class_name": NAMES[k], "metric": metric,
                         "n_patients_after": len(values), "mean_after": mean,
                         "n_patients_before": len(before_values), "mean_before": before_mean,
                         **{f"mean_{slug(label)}": nnunet_means.get(label) for label in nnunet}})
    if dropped:
        print(f"Metrics figure: {dropped} undefined patient values (empty prediction) left out of dots and means")
    decorate(fig, "Baseline Results: All Metrics, All Organs",
        f"Each dot is one of the {n_patients} validation patients, the black bar is the mean, "
        "the hollow diamond is the mean before label correction"
        + (", the glowing star is the nnU-Net mean." if nnunet else "."),
        f"After: {after_run.parent.name}/{after_run.name} (corrected labels). Before: {before_run.parent.name}/{before_run.name} "
        "(original labels). 3D metrics on the original CT grid against original-grid ground truth. "
        "Patients with an undefined distance (empty prediction) are left out of that dot and mean. "
        "Esophagus is not like-for-like: its old ground truth also contained the aorta, which had no label before the correction."
        + (" nnU-Net: 2D, 100 epochs, fold 0, corrected labels, same validation patients, scored with the same 3D metric code."
           if nnunet else ""))
    fig.savefig(out / "plots/before_after_metrics_by_organ.png")
    plt.close(fig)
    return rows


def example_slice(plt, out, original_data, corrected_data, patient):
    import nibabel as nib
    from matplotlib.colors import to_rgba
    ct_nii = nib.load(original_data / patient / f"{patient}.nii.gz")
    before_nii = nib.load(original_data / patient / "GT.nii.gz")
    after_nii = nib.load(corrected_data / patient / "GT.nii.gz")
    if not (ct_nii.shape == before_nii.shape == after_nii.shape):
        raise ValueError(f"CT and ground truth shapes differ: {patient}")
    ct = ct_nii.get_fdata(dtype=np.float32)
    before, after = np.asanyarray(before_nii.dataobj), np.asanyarray(after_nii.dataobj)
    spacing = np.abs(np.diag(after_nii.affine[:3, :3]))
    z = int(np.argmax((after == 4).sum(axis=(0, 1))))  # axial slice where the aorta is largest
    ii, jj = np.nonzero((before[:, :, z] > 0) | (after[:, :, z] > 0))
    mi, mj = (int(round(30 / s)) for s in spacing[:2])  # 30 mm of context
    i0, i1 = max(ii.min() - mi, 0), min(ii.max() + mi + 1, ct.shape[0])
    j0, j1 = max(jj.min() - mj, 0), min(jj.max() + mj + 1, ct.shape[1])

    def crop(array):
        return orthogonal_plane(array, 2, z)[j0:j1, i0:i1]

    def overlay(labels):
        rgba = np.zeros(labels.shape + (4,))
        for k, colour in COLORS.items():
            rgba[labels == k] = to_rgba(colour, .65)
        return rgba

    changed = crop(np.where(before != after, after.astype(np.int16), -1))
    changed_rgba = overlay(changed)
    changed_rgba[changed == 0] = to_rgba(BACKGROUND, .65)  # a voxel that became background
    panels = (("Original ground truth\n(label 1 = esophagus + aorta)", overlay(crop(before))),
              ("Corrected ground truth", overlay(crop(after))),
              ("Voxels whose label changed\n(coloured by new label)", changed_rgba))
    fig = plt.figure(figsize=(12, 5.2))
    axes = fig.subplots(1, 3)
    for ax, (title, rgba) in zip(axes, panels):
        ax.imshow(crop(ct), cmap="gray", vmin=-160, vmax=240, origin="lower", aspect=spacing[1] / spacing[0])
        ax.imshow(rgba, origin="lower", aspect=spacing[1] / spacing[0], interpolation="nearest")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_axis_off()
    for k in sorted(NAMES):
        axes[0].plot([], [], "s", markersize=9, color=COLORS[k], label=NAMES[k])
    if (changed == 0).any():
        axes[0].plot([], [], "s", markersize=9, color=BACKGROUND, label="Set to background")
    legend_below(axes[0], ncol=len(NAMES) + int((changed == 0).any()))
    decorate(fig, "Label correction: one example slice",
        f"{patient}  |  Axial slice {z} (largest aorta cross-section)  |  Original vs corrected ground truth on the same CT",
        "CT window: −160 to 240 HU (level 40, width 400). In the original labels the aorta was merged into label 1 (colored as the esophagus). "
        "The correction relabels part of old label 1 as aorta; the right panel shows exactly which voxels changed.")
    fig.savefig(out / "plots/before_after_example_slice.png")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before-dir", type=Path, required=True, help="analysis folder, original labels")
    p.add_argument("--after-dir", type=Path, required=True, help="analysis folder, corrected labels")
    p.add_argument("--original-data", type=Path, default=Path("data/segthor_part1/train"))
    p.add_argument("--corrected-data", type=Path, default=Path("data/segthor_part1_corrected/train"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--example-patient", default="Patient_01")
    p.add_argument("--before-runs", type=Path, nargs="+", required=True, help="pipeline run folders, original labels")
    p.add_argument("--after-runs", type=Path, nargs="+", required=True, help="pipeline run folders, corrected labels")
    p.add_argument("--before-run", type=Path, required=True,
                   help="the one pipeline run shown per patient in the metrics figure, original labels (median run)")
    p.add_argument("--after-run", type=Path, required=True,
                   help="the one pipeline run shown per patient in the metrics figure, corrected labels (median run)")
    p.add_argument("--nnunet", action="append", default=[], metavar="LABEL=DIR",
                   help="optional, repeatable: add a glowing star for an nnU-Net variant; DIR is a folder with "
                        "eval/metrics_3d.csv (see score_nnunet.py), e.g. 'nnU-Net 2D (TTA)=dataset_analysis/results/nnunet/2d_tta'")
    args = p.parse_args()
    nnunet = {label: Path(path) for label, _, path in (item.partition("=") for item in args.nnunet)}
    if len(nnunet) != len(args.nnunet) or not all(nnunet.values()):
        p.error("--nnunet must be LABEL=DIR, with a distinct label each")
    out = args.output_dir
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    plt = pyplot()

    changes = label_changes(args.original_data, args.corrected_data)
    write_csv(out / "tables/label_changes.csv", changes)
    stray = sum(r["old_label_1_not_new_1_or_4"] + r["changed_outside_old_label_1"] for r in changes)
    print(f"Label check: {stray} voxels differ beyond a split of old label 1 into esophagus/aorta "
          f"({sum(r['changed_voxels'] for r in changes):,} changed in total)")

    before_counts, after_counts = class_counts(args.before_dir), class_counts(args.after_dir)
    class_balance(plt, out, before_counts, after_counts, changes)
    write_csv(out / "tables/dice_3d_before_after.csv",
              dice_comparison(plt, out,
                              run_dice(args.before_runs, {k: before_counts[k][0] > 0 for k in NAMES}),
                              run_dice(args.after_runs, {k: after_counts[k][0] > 0 for k in NAMES})))
    write_csv(out / "tables/metrics_3d_median_runs.csv", metrics_by_organ(plt, out, args.before_run, args.after_run, nnunet))
    example_slice(plt, out, args.original_data, args.corrected_data, args.example_patient)
    print(f"Comparison written -> {out}")


if __name__ == "__main__":
    main()
