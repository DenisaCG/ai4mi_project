"""Before/after comparison of the SEGTHOR label correction, from two finished analysis folders."""
import argparse
import glob
import json
import textwrap
import zlib
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
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


UNCORRECTED = "Uncorrected labels"
BASELINE = "Corrected labels (baseline)"
OVERALL_HALF = .26  # half-height of the black overall line, in rows
GROUP_COLORS = ("#D45F00", INK)  # row-name colour per --group in order of first use: dark orange (not an organ colour), then black
COLLAPSE_BELOW = .1  # an organ whose 3D Dice in a seed (mean over the validation patients) is below this has collapsed; same as compare_losses.py


def find_runs(patterns):
    """Pipeline run folders matching the glob patterns that have finished evaluating (eval/metrics_3d.csv)."""
    found = sorted({Path(match) for pattern in patterns if pattern for match in glob.glob(pattern)})
    done = [run for run in found if (run / "eval/metrics_3d.csv").is_file()]
    if len(done) < len(found):
        print(f"  skipping {len(found) - len(done)} run(s) without eval/metrics_3d.csv (still running?): {patterns}")
    return done


def with_overall(means):
    """Add the "overall" entry: per metric, the mean over the organs that have a value."""
    means["overall"] = {}
    for metric, *_ in METRICS:
        organs = [means[k][metric] for k in NAMES if means[k][metric] is not None]
        means["overall"][metric] = float(np.mean(organs)) if organs else None
    return means


def experiment_means(runs):
    """{class_id or "overall": {metric: value or None}} for one experiment, plus the undefined values left out.

    A value is the mean over the validation patients within each seed, then over seeds. `means["collapsed"]` holds,
    per organ, how many seeds collapsed (Dice below COLLAPSE_BELOW) out of the seeds scored. An organ with no
    ground truth in the experiment's label version has no value, so "overall" averages 3 organs for the
    original labels (no aorta) and 4 otherwise.
    """
    per_run = [run_metrics(run) for run in runs]
    means, dropped, collapsed = {k: {} for k in NAMES}, 0, {}
    for k in NAMES:
        for metric, *_ in METRICS:
            seed_means = []
            for run in per_run:
                if not run[k]["annotated"]:
                    continue
                values = finite(run[k][metric])
                dropped += len(run[k][metric]) - len(values)
                if len(values):
                    seed_means.append(float(values.mean()))
            means[k][metric] = float(np.mean(seed_means)) if seed_means else None
            if metric == "dice":
                collapsed[k] = (sum(m < COLLAPSE_BELOW for m in seed_means), len(seed_means))  # (collapsed, scored) organ-seeds
    means["collapsed"] = collapsed
    return with_overall(means), dropped


def placeholder_means(label, baseline):
    """Invented values for an experiment with no results yet: the baseline moved by up to +-25 %, seeded by the label."""
    rng = np.random.default_rng(zlib.crc32(label.encode()))
    means = {k: {} for k in NAMES}
    for k in NAMES:
        for metric, *_ in METRICS:
            value = baseline[k][metric] * (1 + rng.uniform(-.25, .25))
            means[k][metric] = min(value, 1.0) if metric == "dice" else value
    means["collapsed"] = None  # nothing was run, so nothing collapsed
    return with_overall(means)


def draw_experiments(plt, path, experiments, n_patients, organ_alpha, groups, notes):
    """One figure: a row per experiment with a dot per organ, a grey line across the organs and a black line for the overall.

    `organ_alpha` is the opacity of the organ dots and their grey line; the black overall line is always fully opaque.
    Placeholder rows are drawn fainter still, with a dotted line. `groups` maps a row label to a group name; those
    row names are coloured by group and the groups are listed in a legend. `notes` maps a row label to a small line
    under its name. Row names, title, subtitle, legend and footnote are placed here (decorate() can only centre them).
    """
    n = len(experiments)
    height = .55 * n + 2.65
    group_color = {g: GROUP_COLORS[i % len(GROUP_COLORS)] for i, g in enumerate(dict.fromkeys(groups.values()))}
    fig = plt.figure(figsize=(14, height))
    axes = fig.subplots(1, len(METRICS), sharey=True)
    fig.subplots_adjust(left=.30, top=1 - 1.7 / height, right=.90, bottom=.95 / height, wspace=.07)
    for j, (metric, title, direction, fmt) in enumerate(METRICS):
        ax = axes[j]
        shown = [e["means"][key][metric] for e in experiments for key in (*NAMES, "overall") if e["means"][key][metric] is not None]
        top = 1.0 if metric == "dice" else max(shown)  # the axis fits the means
        ax.set_xlim(-.03 * top, 1.04 * top)
        ax.set_ylim(n - .5, -.5)
        ax.set_yticks(range(n))
        if metric == "dice":
            ax.set_xticks(np.arange(0, 1.01, .2))
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10], min_n_ticks=3))
        clean_axis(ax, "x")
        ax.spines["left"].set_visible(False)
        for i, e in enumerate(experiments):
            if e["label"] == BASELINE:
                ax.axhspan(i - .5, i + .5, color=tint(BACKGROUND, .55), zorder=.2, linewidth=0)
            elif i % 2 == 0:
                ax.axhspan(i - .5, i + .5, color=tint(BACKGROUND, .3), zorder=.2, linewidth=0)
            faded = e["placeholder"]
            organs = {k: e["means"][k][metric] for k in NAMES if e["means"][k][metric] is not None}
            if len(organs) > 1:
                ax.hlines(i, min(organs.values()), max(organs.values()), color=BACKGROUND, linewidth=4.5,
                          linestyle=":" if faded else "-", alpha=organ_alpha * (.8 if faded else 1), zorder=2)
            for key, value in organs.items():
                ax.scatter(value, i, s=150, color=COLORS[key], edgecolor="white", linewidth=1.5,
                           alpha=organ_alpha * (.4 if faded else 1), zorder=3, clip_on=False)
            overall = e["means"]["overall"][metric]
            ax.vlines(overall, i - OVERALL_HALF, i + OVERALL_HALF, color=INK, linewidth=3.5, alpha=.4 if faded else 1, zorder=5)
            ax.text(overall, i - OVERALL_HALF - .03, fmt.format(overall), ha="center", va="bottom", fontsize=10, weight="bold",
                    color=INK, alpha=.4 if faded else 1, zorder=5)
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=24)
        ax.annotate(direction, (0, 1), xycoords="axes fraction", xytext=(0, 6), textcoords="offset points",
                    ha="left", va="bottom", fontsize=10.5, color=MUTED)
    for i, e in enumerate(experiments):
        note = notes.get(e["label"])
        in_group = e["label"] in groups
        color = group_color[groups[e["label"]]] if in_group else MUTED if e["placeholder"] else INK
        alpha = .6 if e["placeholder"] and in_group else 1
        axes[0].annotate(e["label"] + ("  [placeholder]" if e["placeholder"] else ""), (0, i), xycoords=axes[0].get_yaxis_transform(),
                         xytext=(-8, 6 if note else 0), textcoords="offset points", ha="right", va="center", fontsize=12,
                         fontweight="bold" if e["label"] == BASELINE else "normal", style="italic" if e["placeholder"] else "normal",
                         color=color, alpha=alpha, annotation_clip=False)
        if note:
            axes[0].annotate(note, (0, i), xycoords=axes[0].get_yaxis_transform(), xytext=(-8, -8), textcoords="offset points",
                             ha="right", va="center", fontsize=9.5, color=MUTED, alpha=.6 if e["placeholder"] else 1,
                             annotation_clip=False)
    for ax in axes:
        ax.tick_params(labelleft=False)
    last = axes[-1]  # text column to the right of the last panel: collapsed organ-seeds / organ-seeds scored
    last.annotate("Collapsed", (1, 1), xycoords="axes fraction", xytext=(14, 24), textcoords="offset points",
                  ha="left", va="baseline", fontsize=14, fontweight="bold", annotation_clip=False)
    last.annotate("organ-seeds", (1, 1), xycoords="axes fraction", xytext=(14, 6), textcoords="offset points",
                  ha="left", va="bottom", fontsize=10.5, color=MUTED, annotation_clip=False)
    for i, e in enumerate(experiments):
        counts = e["means"]["collapsed"]
        if counts is None:
            text, color, weight, alpha = "–", MUTED, "normal", .6
        else:
            hits, scored = (sum(c[n] for c in counts.values()) for n in (0, 1))
            text, color, weight, alpha = f"{hits} / {scored}", INK if hits else MUTED, "bold" if hits else "normal", 1
        last.annotate(text, (1, i), xycoords=last.get_yaxis_transform(), xytext=(14, 0), textcoords="offset points",
                      ha="left", va="center", fontsize=12, color=color, fontweight=weight, alpha=alpha, annotation_clip=False)

    fig.text(.015, 1 - .12 / height, "All Experiments, Absolute Values", fontsize=22, fontweight="bold", ha="left", va="top")
    fig.text(.015, 1 - .62 / height, "One row per experiment, a dot per organ, a black line for the overall. "
             "Dotted lines and faded dots = runs not finished.", fontsize=13.5, color=INK, ha="left", va="top")
    handles = [Line2D([], [], marker="o", linestyle="", markersize=11, color=COLORS[k], markeredgecolor="white", label=NAMES[k])
               for k in sorted(NAMES)]
    handles.append(Line2D([], [], marker="|", linestyle="", markersize=14, markeredgewidth=3.5, color=INK, label="Overall"))
    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(.985, 1 - .98 / height), ncol=len(handles), frameon=False,
               fontsize=12, handletextpad=.3, columnspacing=1.6)
    if group_color:
        fig.legend(handles=[Line2D([], [], marker="s", linestyle="", markersize=10, color=color, label=group)
                            for group, color in group_color.items()],
                   loc="center left", bbox_to_anchor=(.015, 1 - .98 / height), ncol=len(group_color), frameon=False,
                   fontsize=12, handletextpad=.3, columnspacing=1.6)
    fig.text(.015, .12 / height, textwrap.fill(
        f"Means over the {n_patients} validation patients and each experiment's seeds. Faded rows are invented placeholders, not results. "
        "Uncorrected labels have no aorta (their overall averages 3 organs) and their esophagus is not like-for-like. "
        "Empty predictions are left out of the distance means. Collapsed = an organ with 3D Dice below "
        f"{COLLAPSE_BELOW:g} in a seed, counted over organs x seeds.", width=215),
        fontsize=9, style="italic", color=MUTED, ha="left", va="bottom")
    fig.savefig(path)
    plt.close(fig)


def metrics_by_experiment(plt, out, before_runs, after_runs, compare=None, groups=None, notes=None):
    """Dice, HD95 and ASSD, one row per experiment, drawn twice: with solid organ dots and with shaded ones.

    Rows are the uncorrected labels (`before_runs`), the corrected-label baseline (`after_runs`), then `compare`
    (label -> run glob patterns) in the order given. Every value is the mean over the validation patients and
    the experiment's seeds. An experiment with no finished runs gets invented placeholder values and is drawn
    faded. `groups` (row label -> group name) colours row names; `notes` (row label -> text) adds a line under a name. Returns the long table (one row per experiment, organ or overall, and metric).
    """
    baseline, dropped = experiment_means(after_runs)
    n_patients = len(run_metrics(after_runs[0])[min(NAMES)]["dice"])
    specs = ([(UNCORRECTED, list(before_runs)), (BASELINE, None)]
             + [(label, find_runs(patterns)) for label, patterns in (compare or {}).items()])
    experiments = []
    for label, runs in specs:
        if runs is None:  # the corrected-label baseline, already computed
            means, runs = baseline, list(after_runs)
        elif runs:
            means, undefined = experiment_means(runs)
            dropped += undefined
        else:
            means = placeholder_means(label, baseline)
        experiments.append({"label": label, "means": means, "n_seeds": len(runs), "placeholder": not runs})
        print(f"  {label}: " + (f"{len(runs)} run(s)" if runs else "no finished runs, placeholder values"))
    if dropped:
        print(f"Metrics figure: {dropped} undefined patient values (empty prediction) left out of the means")
    for version, organ_alpha in (("solid", 1.0), ("shaded", .45)):
        draw_experiments(plt, out / f"plots/before_after_metrics_by_experiment_{version}.png", experiments, n_patients, organ_alpha, groups or {}, notes or {})
    rows = []
    for metric, *_ in METRICS:
        for e in experiments:
            for key in (*NAMES, "overall"):
                counts = e["means"]["collapsed"]
                pairs = None if metric != "dice" or counts is None else list(counts.values()) if key == "overall" else [counts[key]]
                rows.append({"experiment": e["label"], "class_name": NAMES.get(key, "Overall"), "metric": metric,
                             "n_seeds": e["n_seeds"], "placeholder": e["placeholder"], "mean": e["means"][key][metric],
                             "collapsed_seeds": None if pairs is None else sum(c for c, _ in pairs),
                             "seeds_scored": None if pairs is None else sum(n for _, n in pairs)})
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
    p.add_argument("--compare", action="append", default=[], metavar="LABEL=PATTERN",
                   help="optional, repeatable, in row order after the two baselines: one row for an experiment; PATTERN is a "
                        "glob of folders with eval/metrics_3d.csv (pipeline runs, or nnU-Net folders from score_nnunet.py), e.g. "
                        "'Dice loss=runs/segthor_enet_dice_fg_corrected/seed*'. Repeat a LABEL to add more runs. "
                        "No finished run (or an empty PATTERN) draws a faded placeholder row instead.")
    p.add_argument("--group", action="append", default=[], metavar="GROUP=LABEL",
                   help="optional, repeatable: colour the name of row LABEL by GROUP (one colour per GROUP, in order of first use) "
                        "and list the groups in a legend, e.g. 'Loss on foreground only=Dice loss'")
    p.add_argument("--note", action="append", default=[], metavar="LABEL=TEXT",
                   help="optional, repeatable: a small line under the name of row LABEL, e.g. 'Dice loss=soft Dice on the foreground'")
    args = p.parse_args()
    compare = {}
    for item in args.compare:
        label, sep, pattern = item.partition("=")
        if not label or not sep:
            p.error(f"--compare must be LABEL=PATTERN (PATTERN may be empty for a placeholder): {item!r}")
        compare.setdefault(label, []).append(pattern)
    groups = {}
    for item in args.group:
        group, sep, label = item.partition("=")
        if not group or not sep:
            p.error(f"--group must be GROUP=LABEL: {item!r}")
        if label not in {UNCORRECTED, BASELINE, *compare}:
            p.error(f"--group label {label!r} is not a row; rows are {[UNCORRECTED, BASELINE, *compare]}")
        groups[label] = group
    notes = {}
    for item in args.note:
        label, sep, text = item.partition("=")
        if not label or not sep or label not in {UNCORRECTED, BASELINE, *compare}:
            p.error(f"--note must be LABEL=TEXT with LABEL one of the rows {[UNCORRECTED, BASELINE, *compare]}: {item!r}")
        notes[label] = text
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
    write_csv(out / "tables/metrics_3d_by_experiment.csv",
              metrics_by_experiment(plt, out, args.before_runs, args.after_runs, compare, groups, notes))
    example_slice(plt, out, args.original_data, args.corrected_data, args.example_patient)
    print(f"Comparison written -> {out}")


if __name__ == "__main__":
    main()
