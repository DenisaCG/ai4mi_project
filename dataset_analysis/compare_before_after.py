"""Before/after comparison of the SEGTHOR label correction, from two finished analysis folders."""
import argparse
import json
from pathlib import Path

import numpy as np

from figures import orthogonal_plane
from style import BACKGROUND, COLORS, INK, NAMES, frame, pyplot
from utils import read_csv, write_csv

MUTED = "#494949"
BEFORE = "Before: original labels"
AFTER = "After: corrected labels"


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


def legend_before_after(fig):
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor="white", edgecolor=MUTED, linewidth=2, label=BEFORE),
                        Patch(facecolor=MUTED, label=AFTER)],
               loc="lower left", bbox_to_anchor=(.055, .155), ncol=2, fontsize=13)


def paired_bars(ax, data, fmt, xlabel):
    ks = sorted(NAMES)
    top = max(max(pair) for pair in data.values())
    for y, k in enumerate(ks):
        before, after = data[k]
        ax.barh(y - .19, before, height=.34, facecolor="white", edgecolor=COLORS[k], linewidth=2)
        ax.barh(y + .19, after, height=.34, color=COLORS[k])
        ax.text(before + .02 * top, y - .19, fmt(before) if before else "not annotated",
                va="center", fontsize=12, color=INK)
        ax.text(after + .02 * top, y + .19, fmt(after), va="center", fontsize=12, color=INK, weight="bold")
    ax.set(xlim=(0, top * 1.3), ylim=(len(ks) - .5, -.5), yticks=range(len(ks)),
           yticklabels=[NAMES[k] for k in ks], xlabel=xlabel)
    ax.spines["left"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E7E7E7", linewidth=.7)
    ax.tick_params(length=0, pad=8)


def class_balance(plt, out, before, after, changes):
    ratio = (sum(r["esophagus_voxels_before"] for r in changes)
             / sum(r["esophagus_voxels_after"] for r in changes))
    fig = frame(plt, "Label correction: what changed in the ground truth",
        f"Original vs corrected NIfTI ground truth  |  {len(changes)} patients pooled  |  Aorta was not annotated before",
        "Left: each organ's share of annotated foreground voxels (background excluded). Right: annotated voxels per organ.\n"
        f"The original label 1 covered esophagus and aorta together, so the esophagus ground truth was {ratio:.1f}x larger than the corrected one.")
    left, right = fig.subplots(1, 2, sharey=True)
    paired_bars(left, {k: (100 * before[k][1], 100 * after[k][1]) for k in NAMES},
                lambda v: f"{v:.1f}%", "Share of annotated foreground voxels (%)")
    paired_bars(right, {k: (before[k][0] / 1e6, after[k][0] / 1e6) for k in NAMES},
                lambda v: f"{v:.2f} M", "Annotated voxels (millions)")
    right.tick_params(labelleft=False)
    legend_before_after(fig)
    fig.savefig(out / "plots/before_after_class_balance.png")
    plt.close(fig)


def dice_comparison(plt, out, before, after):
    n_before, n_after = max(map(len, before.values())), max(map(len, after.values()))
    fig = frame(plt, "Label correction: effect on 3D segmentation quality",
        f"Same ENet + cross-entropy recipe  |  one dot = one training run  |  {n_before} original-label vs {n_after} corrected-label runs\n"
        "Each dot is that run's mean 3D Dice over the 5 validation patients (reconstructed predictions vs original-grid ground truth)",
        "Black bars mark the mean over runs; the numbers on top read before → after. A run at exactly 0 predicted no voxels of that organ.\n"
        "Esophagus is not like-for-like: its old ground truth also contained the aorta.")
    ax = fig.add_subplot(111)
    rows = []
    for k in sorted(NAMES):
        means = {}
        for x, values, filled in ((k - .17, before[k], False), (k + .17, after[k], True)):
            offsets = np.linspace(-.09, .09, len(values)) if len(values) > 1 else [0]
            for offset, value in zip(offsets, values):
                ax.scatter(x + offset, value, s=90, zorder=3,
                           **(dict(color=COLORS[k], edgecolor="white", linewidth=1) if filled
                              else dict(facecolor="white", edgecolor=COLORS[k], linewidth=2)))
            if values:
                ax.hlines(np.mean(values), x - .13, x + .13, color=INK, linewidth=3, zorder=4)
            zeros = sum(v == 0 for v in values)
            if zeros:
                ax.text(x, .075, f"{zeros} of {len(values)}\nat 0", ha="center", va="bottom", fontsize=11, color=MUTED)
            means[filled] = np.mean(values) if values else None
        ax.text(k, 1.06, f"{'–' if means[False] is None else f'{means[False]:.2f}'} → "
                         f"{'–' if means[True] is None else f'{means[True]:.2f}'}",
                ha="center", fontsize=16, weight="bold")
        if not before[k]:
            ax.text(k - .17, .5, "no ground\ntruth", ha="center", va="center", fontsize=11, color=MUTED)
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
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E7E7E7", linewidth=.7)
    ax.tick_params(length=0, pad=8)
    legend_before_after(fig)
    fig.savefig(out / "plots/before_after_dice_3d.png")
    plt.close(fig)
    return rows


def example_slice(plt, out, original_data, corrected_data, patient):
    import nibabel as nib
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch
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
    fig = frame(plt, "Label correction: one example slice",
        f"{patient}  |  Axial slice {z} (largest aorta cross-section)  |  Original vs corrected ground truth on the same CT",
        "CT window: −160 to 240 HU (level 40, width 400). In the original labels the aorta was merged into label 1 (purple).\n"
        "The correction relabels part of old label 1 as aorta; the right panel shows exactly which voxels changed.")
    for ax, (title, rgba) in zip(fig.subplots(1, 3), panels):
        ax.imshow(crop(ct), cmap="gray", vmin=-160, vmax=240, origin="lower", aspect=spacing[1] / spacing[0])
        ax.imshow(rgba, origin="lower", aspect=spacing[1] / spacing[0], interpolation="nearest")
        ax.set_title(title, fontsize=15)
        ax.set_axis_off()
    handles = [Patch(facecolor=COLORS[k], label=NAMES[k]) for k in sorted(NAMES)]
    if (changed == 0).any():
        handles.append(Patch(facecolor=BACKGROUND, label="Set to background"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .155), ncol=len(handles), fontsize=13)
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
    args = p.parse_args()
    out = args.output_dir
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    plt = pyplot(out)

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
    example_slice(plt, out, args.original_data, args.corrected_data, args.example_patient)
    print(f"Comparison written -> {out}")


if __name__ == "__main__":
    main()
