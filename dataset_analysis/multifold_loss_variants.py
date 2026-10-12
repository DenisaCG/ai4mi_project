"""Compare ENet loss variants across multiple CV folds and seeds.

Example:
    python dataset_analysis/multifold_loss_variants.py \
      --groups 'Dice + CE=metrics/full_cv4_enet_dice_ce' \
               'Weighted CE + Dice=metrics/full_cv4_enet_weighted_dice_ce_preprocessing' \
               'CE + Dice + Boundary=metrics/full_cv4_enet_dice_ce_boundary_preprocessing' \
      --folds 0 1 2 3 --out figures/multifold_loss_variants

Each DIR is a prefix: for fold F, the script looks for DIR_foldF/seed*/summary.json.
It also supports DIR/foldF/seed*/summary.json. Each point is one (fold, seed)
run. The large marker is the pooled mean, with whiskers for sample SD across
all available runs. Each run is equally weighted (unequal seed counts matter).
Writes PNG + PDF for Dice, HD95, ASSD, validation curves, and paired Dice/ASSD gain plots, plus per-run CSV.
"""

import argparse
import csv
import json
import math
import sys
import textwrap
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke

from utils import CLASSES, INK, REPO, clean_axis, decorate, pyplot, write_csv
from plot_style import PALETTE, legend_below

sys.path.append(str(REPO))
from architecture_progress import FG, FG_COLOR, ORGAN_COLOR, save, tighten
from src.aggregate import CV_METRICS, nan_free_mean


def group_spec(spec):
    label, sep, directory = spec.rpartition('=')
    if not (sep and label and directory):
        raise argparse.ArgumentTypeError(f'{spec!r} must be LABEL=DIR_PREFIX')
    return label.replace('\\n', '\n'), Path(directory)


def load_groups(folder, folds):
    runs = {}
    for fold in folds:
        candidates = [Path(f'{folder}_fold{fold}'), folder / f'fold{fold}']
        found = next((p for p in candidates if p.is_dir()), None)
        if found is None:
            print(f'Warning: no directory for fold {fold}: {candidates}', file=sys.stderr)
            continue
        for file in sorted(found.glob('seed*/summary.json')):
            try:
                seed = int(file.parent.name.removeprefix('seed'))
                evaluation = json.loads(file.read_text())['eval']
                epoch_file = file.parent / 'epochs.csv'
                epochs = []
                if epoch_file.exists():
                    with epoch_file.open(newline='') as f:
                        epochs = [
                            {k: float(v) for k, v in row.items()
                             if k.startswith('val_dice_') and v not in ('', None)}
                            for row in csv.DictReader(f)
                        ]
                runs[(fold, seed)] = {'eval': evaluation, 'epochs': epochs}
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                print(f'Warning: skipping {file}: {exc}', file=sys.stderr)
    if not runs:
        raise SystemExit(f'{folder}: no runs found for folds {folds}')
    return runs


def values(runs, key, organ):
    return np.array([
        r['eval'].get(f'val_{key}_{organ}', np.nan) for r in runs.values()
    ], dtype=float)


def metric_figure(plt, groups, organs, metric, out, title, note, folds):
    label, key, digits = metric
    panels = [*organs, FG]
    x = np.arange(len(groups))
    fig, axes = plt.subplots(
        len(panels), 1, sharex=True,
        figsize=(max(9, 2.15 * len(groups) + 2), 1.95 * len(panels) + 1.6),
        squeeze=False,
    )
    fold_colors = [PALETTE[i % len(PALETTE)] for i in range(len(folds))]
    color_by_fold = dict(zip(folds, fold_colors))
    for ax, organ in zip(axes[:, 0], panels):
        color = ORGAN_COLOR.get(organ, FG_COLOR)
        for i, (_, runs) in enumerate(groups):
            xy = [(fold, seed, float(r['eval'].get(f'val_{key}_{organ}', np.nan)))
                  for (fold, seed), r in runs.items()]
            for fold, seed, val in xy:
                if np.isfinite(val):
                    # Deterministic offsets: distinguish fold groups visually.
                    offset = (folds.index(fold) - (len(folds) - 1) / 2) * 0.07
                    seed_offset = (seed % 5 - 2) * 0.011
                    ax.scatter(i + offset + seed_offset, val, s=26,
                               c=[color_by_fold[fold]], alpha=0.75,
                               edgecolors='none', zorder=3)
            v = values(runs, key, organ)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            mean = v.mean()
            sd = v.std(ddof=1) if len(v) > 1 else 0.0
            ax.errorbar(i, mean, yerr=sd, fmt='o', color=color, markersize=9,
                        markeredgecolor='white', markeredgewidth=1.3,
                        capsize=6, elinewidth=2, zorder=5)
            ax.annotate(f'{mean:.{digits}f} ± {sd:.{digits}f}', (i, mean),
                        xytext=(0, 13), textcoords='offset points',
                        ha='center', fontsize=8.5, fontweight='bold', color=INK,
                        path_effects=[withStroke(linewidth=2.8, foreground='white')],
                        zorder=7)
        ax.set_title('All organs (mean)' if organ == FG else organ.capitalize(),
                     loc='left', fontsize=11, fontweight='bold')
        ax.set_ylabel(label)
        ax.margins(x=0.12, y=0.38)
        clean_axis(ax)
    bottom = axes[-1, 0]
    bottom.set_xticks(x, [
        '\n'.join(textwrap.fill(line, 18) for line in name.split('\n'))
        + f'\n(n={len(runs)} runs)' for name, runs in groups
    ], fontsize=9)
    for fold in folds:
        bottom.scatter([], [], s=25, color=color_by_fold[fold], label=f'Fold {fold}')
    bottom.plot([], [], 'o', color=INK, ms=8, label='Mean ± SD (runs)')
    legend_below(bottom, ncol=min(len(folds)+1, 5))
    decorate(
        fig, title.format(metric=label.split()[0]),
        subtitle=f'ENet · folds {", ".join(map(str, folds))} · each small point = one fold–seed run'
                 + (f' · {note}' if note else ''),
        footnote_text=(
            f'{label}, 3D on original CT grid (best checkpoint); each run is '
            'a patient-averaged score. Large marker = mean and SD across all '
            'available fold–seed runs, with equal weight per run. '
            'Y-axes differ by organ. Missing runs are not imputed.'
        ),
    )
    tighten(fig, 0.5, 0.35)
    fig.subplots_adjust(hspace=0.55)
    save(plt, fig, out, f'multifold_loss_variants_{key}')


def curves_figure(plt, groups, out, folds, note):
    """Per-fold validation trajectories; no averaging across incompatible folds."""
    metrics = [('val_dice_fg', 'All organs'), ('val_dice_esophagus', 'Esophagus')]
    fig, axes = plt.subplots(len(folds), len(metrics),
                             figsize=(12, max(4.5, 2.9 * len(folds))),
                             squeeze=False, sharex=True)
    for row, fold in enumerate(folds):
        for col, (metric, metric_label) in enumerate(metrics):
            ax = axes[row, col]
            for i, (label, runs) in enumerate(groups):
                curves = []
                for (run_fold, _), run in runs.items():
                    if run_fold != fold:
                        continue
                    vals = [r[metric] for r in run['epochs'] if metric in r]
                    if vals:
                        curves.append(np.array(vals, dtype=float))
                if not curves:
                    continue
                color = PALETTE[i % len(PALETTE)]
                for v in curves:
                    ax.plot(np.arange(len(v)), v, color=color, lw=0.7, alpha=0.3)
                n = min(map(len, curves))
                ax.plot(np.arange(n), np.mean([v[:n] for v in curves], axis=0),
                        color=color, lw=2.1, label=' '.join(label.split()))
            ax.set_title(f'Fold {fold} · {metric_label}', loc='left', fontsize=10)
            ax.set_ylabel('2D val Dice')
            if row == len(folds)-1:
                ax.set_xlabel('Epoch')
            clean_axis(ax)
    legend_below(axes[-1, 0], ncol=2)
    decorate(fig, 'Validation Dice by fold and loss variant',
             subtitle='Thick = mean across seeds within fold · thin = individual seeds'
                      + (f' · {note}' if note else ''),
             footnote_text='2D validation Dice per epoch; not 3D test-volume evaluation.')
    save(plt, fig, out, 'multifold_loss_variants_curves')


def gain_figures(plt, groups, organs, out, folds):
    """
    Plots paired changes relative to the first group (baseline = Dice + CE).

    For Dice:
        delta = variant - baseline
        positive = improvement

    For ASSD:
        delta = baseline - variant
        positive = improvement (lower ASSD is better)
    """
    baseline_label, baseline_runs = groups[0]
    comparisons = groups[1:]

    metrics = [
        ("dice", "Dice gain vs Dice + CE", 3),
        ("assd", "ASSD reduction (mm) vs Dice + CE", 2),
    ]

    organ_labels = {
        "esophagus": "Es",
        "heart": "He",
        "trachea": "Tr",
        "aorta": "Ao",
    }

    for key, ylabel, digits in metrics:
        # -----------------------------
        # Collect paired deltas first
        # -----------------------------
        panel_data = []
        all_values = []

        for comp_label, comp_runs in comparisons:
            matched = sorted(set(baseline_runs) & set(comp_runs))
            organ_to_vals = {}
            avg_vals = []

            for organ in organs:
                vals = []
                for pair in matched:
                    b = baseline_runs[pair]["eval"].get(f"val_{key}_{organ}", np.nan)
                    c = comp_runs[pair]["eval"].get(f"val_{key}_{organ}", np.nan)

                    if not (np.isfinite(b) and np.isfinite(c)):
                        continue

                    delta = (c - b) if key == "dice" else (b - c)
                    vals.append(float(delta))

                organ_to_vals[organ] = np.array(vals, dtype=float)
                all_values.extend(vals)

            # average across organs for each matched run
            for pair in matched:
                per_organ = []
                for organ in organs:
                    b = baseline_runs[pair]["eval"].get(f"val_{key}_{organ}", np.nan)
                    c = comp_runs[pair]["eval"].get(f"val_{key}_{organ}", np.nan)

                    if not (np.isfinite(b) and np.isfinite(c)):
                        continue

                    delta = (c - b) if key == "dice" else (b - c)
                    per_organ.append(float(delta))

                if per_organ:
                    avg_vals.append(float(np.mean(per_organ)))

            avg_vals = np.array(avg_vals, dtype=float)
            all_values.extend(avg_vals.tolist())

            panel_data.append((comp_label, organ_to_vals, avg_vals, len(matched)))

        if not all_values:
            continue

        ymin = min(all_values)
        ymax = max(all_values)
        span = ymax - ymin
        if span == 0:
            span = 0.01 if key == "dice" else 0.1
        margin = 0.18 * span
        ymin -= margin
        ymax += margin

        # make sure both sides of zero are visible
        ymin = min(ymin, -0.001 if key == "dice" else -0.05)
        ymax = max(ymax,  0.001 if key == "dice" else  0.05)

        # -----------------------------
        # Plot
        # -----------------------------
        fig, axes = plt.subplots(
            1, len(panel_data),
            figsize=(max(10, 3.8 * len(panel_data)), 5.8),
            squeeze=False,
            sharey=True
        )
        axes = axes[0]

        for ax, (comp_label, organ_to_vals, avg_vals, n_runs) in zip(axes, panel_data):
            # Improvement / worse shading
            ax.axhspan(0, ymax, alpha=0.12, zorder=0)
            ax.axhspan(ymin, 0, alpha=0.08, zorder=0)
            ax.axhline(0, color=INK, lw=1.6, zorder=1)

            x_positions = np.arange(len(organs) + 1)
            labels = [organ_labels.get(o, o[:2].title()) for o in organs] + ["Avg"]

            # plot organs
            for i, organ in enumerate(organs):
                vals = organ_to_vals[organ]
                if len(vals) == 0:
                    continue

                color = ORGAN_COLOR.get(organ, FG_COLOR)

                if len(vals) > 1:
                    parts = ax.violinplot([vals], positions=[i], widths=0.72,
                                          showmeans=False, showmedians=False, showextrema=False)
                    for body in parts["bodies"]:
                        body.set_facecolor(color)
                        body.set_alpha(0.18)
                        body.set_edgecolor(color)
                        body.set_linewidth(1.0)

                jitter = np.linspace(-0.10, 0.10, len(vals)) if len(vals) > 1 else np.array([0.0])
                ax.scatter(np.full(len(vals), i) + jitter, vals, s=22,
                           color=color, alpha=0.7, edgecolors="none", zorder=3)

                mean = float(np.mean(vals))
                ax.plot(i, mean, "D", color=color, ms=7, mec="white", mew=1.0, zorder=4)
                ax.annotate(
                    f"{mean:+.{digits}f}",
                    (i, mean),
                    xytext=(0, 10 if mean >= 0 else -14),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color=INK,
                    path_effects=[withStroke(linewidth=2.5, foreground="white")],
                    zorder=5,
                )

            # plot average across organs
            if len(avg_vals):
                avg_x = len(organs)

                if len(avg_vals) > 1:
                    parts = ax.violinplot([avg_vals], positions=[avg_x], widths=0.72,
                                          showmeans=False, showmedians=False, showextrema=False)
                    for body in parts["bodies"]:
                        body.set_facecolor(INK)
                        body.set_alpha(0.15)
                        body.set_edgecolor(INK)
                        body.set_linewidth(1.0)

                jitter = np.linspace(-0.10, 0.10, len(avg_vals)) if len(avg_vals) > 1 else np.array([0.0])
                ax.scatter(np.full(len(avg_vals), avg_x) + jitter, avg_vals, s=24,
                           color=INK, alpha=0.7, edgecolors="none", zorder=3)

                mean_avg = float(np.mean(avg_vals))
                ax.plot(avg_x, mean_avg, "*", color=INK, ms=12, mec="white", mew=1.0, zorder=4)
                ax.annotate(
                    f"{mean_avg:+.{digits}f}",
                    (avg_x, mean_avg),
                    xytext=(0, 10 if mean_avg >= 0 else -14),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                    color=INK,
                    path_effects=[withStroke(linewidth=2.6, foreground="white")],
                    zorder=5,
                )

            ax.set_xticks(x_positions, labels, fontsize=10)
            ax.set_ylim(ymin, ymax)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{comp_label}\n(n={n_runs} matched runs)", fontsize=10.5, loc="left", fontweight="bold")
            clean_axis(ax)

            # Improvement / worse labels
            ax.text(0.98, 0.98, "Improvement", ha="right", va="top",
                    transform=ax.transAxes, fontsize=9, fontweight="bold")
            ax.text(0.98, 0.02, "Worse", ha="right", va="bottom",
                    transform=ax.transAxes, fontsize=9, fontweight="bold")

        title_metric = "Dice" if key == "dice" else "ASSD"
        decorate(
            fig,
            f"Upgraded ENet: change from Dice + CE ({title_metric})",
            subtitle=(
                "Each point = one matched fold–seed run. "
                "Above 0 = improvement, below 0 = worse."
            ),
            footnote_text=(
            "Changes relative to Dice + CE. Positive = improvement "
            "(higher Dice or lower ASSD). "
            "Avg = mean change across four organs per run."
        ),
        )
        tighten(fig, 0.5, 0.35)

        fig.subplots_adjust(
            bottom=0.10,
            top=0.78,
            wspace=0.28
        )

        save(plt, fig, out, f"multifold_loss_variants_gain_{key}")

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--groups', nargs='+', type=group_spec, metavar='LABEL=DIR_PREFIX',
                        default=[
                            group_spec('Dice + CE=/home/scur0049/ai4mi_project/metrics/full_cv4_enet_dice_ce'),
                            group_spec('Weighted CE + Dice=/home/scur0049/ai4mi_project/metrics/full_cv4_enet_weighted_dice_ce_preprocessing'),
                            group_spec('CE + Dice + Constant Boundary=/home/scur0049/ai4mi_project/metrics/full_cv4_enet_dice_ce_boundary_preprocessing'),
                            group_spec('CE + Dice + Increasing Boundary=/home/scur0049/ai4mi_project/metrics/full_cv4_enet_dice_ce_boundary_v2'),
                        ])
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--out', type=Path, default=REPO / 'figures/multifold_loss_variants_v2')
    parser.add_argument('--title', default='{metric} per organ across CV folds and loss variants')
    parser.add_argument('--note', default='')
    args = parser.parse_args(argv)
    groups = [(label, load_groups(folder, args.folds)) for label, folder in args.groups]
    first = next(iter(groups[0][1].values()))['eval']
    organs = [name for name in CLASSES.values() if f'val_dice_{name}' in first]
    args.out.mkdir(parents=True, exist_ok=True)
    plt = pyplot()
    table = []
    for metric in CV_METRICS:
        _, key, _ = metric
        metric_figure(plt, groups, organs, metric, args.out, args.title, args.note, args.folds)
        for (label, runs), (_, folder) in zip(groups, args.groups):
            for organ in [*organs, FG]:
                for (fold, seed), run in runs.items():
                    table.append({
                        'metric': key, 'variant': ' '.join(label.split()),
                        'folder_prefix': str(folder), 'organ': organ,
                        'fold': fold, 'seed': seed,
                        'value': run['eval'].get(f'val_{key}_{organ}', float('nan')),
                    })
    curves_figure(plt, groups, args.out, args.folds, args.note)
    gain_figures(plt, groups, organs, args.out, args.folds)
    write_csv(args.out / 'multifold_loss_variants.csv', table)
    print(f'{len(groups)} variants; {sum(len(r) for _, r in groups)} runs -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
