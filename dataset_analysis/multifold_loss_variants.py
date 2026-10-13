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
Writes PNG + PDF for Dice, HD95, ASSD, validation curves, and paired Dice/ASSD/HD95 gain plots, plus per-run CSV.
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
    """Paired per-run improvements versus the first group (baseline).

    DICE: variant - baseline, positive is better.
    ASSD and HD95: baseline - variant, positive is better (in mm).
    Each sample is one matching (fold, seed); point color denotes fold.
    """
    if len(groups) < 2:
        return

    baseline_label, baseline_runs = groups[0]
    comparisons = groups[1:]
    organs = [organ for organ in organs if organ != FG]
    organ_labels = {'esophagus': 'Es', 'heart': 'He',
                    'trachea': 'Tr', 'aorta': 'Ao'}
    fold_colors = dict(zip(folds, [PALETTE[i % len(PALETTE)]
                                    for i in range(len(folds))]))

    metrics = [
        ('dice', 'Dice gain vs Dice + CE', 3),
        ('assd', 'ASSD reduction (mm) vs Dice + CE', 2),
        ('hd95', 'HD95 reduction (mm) vs Dice + CE', 2),
    ]

    for key, ylabel, digits in metrics:
        panel_data = []
        all_values = []

        for comp_label, comp_runs in comparisons:
            matched = sorted(set(baseline_runs) & set(comp_runs))
            organ_to_points = {}
            average_points = []

            for organ in organs:
                points = []
                for (fold, seed) in matched:
                    metric_name = f'val_{key}_{organ}'
                    b = float(baseline_runs[(fold, seed)]['eval'].get(metric_name, np.nan))
                    c = float(comp_runs[(fold, seed)]['eval'].get(metric_name, np.nan))
                    if not (np.isfinite(b) and np.isfinite(c)):
                        continue
                    change = (c - b) if key == 'dice' else (b - c)
                    points.append((fold, seed, change))
                    all_values.append(change)
                organ_to_points[organ] = points

            # Average across the available organs separately for EACH run.
            for (fold, seed) in matched:
                changes = []
                for organ in organs:
                    metric_name = f'val_{key}_{organ}'
                    b = float(baseline_runs[(fold, seed)]['eval'].get(metric_name, np.nan))
                    c = float(comp_runs[(fold, seed)]['eval'].get(metric_name, np.nan))
                    if np.isfinite(b) and np.isfinite(c):
                        changes.append((c - b) if key == 'dice' else (b - c))
                if changes:
                    change = float(np.mean(changes))
                    average_points.append((fold, seed, change))
                    all_values.append(change)

            panel_data.append((comp_label, organ_to_points, average_points, len(matched)))

        if not all_values:
            print(f'Warning: no paired values for {key}', file=sys.stderr)
            continue

        ymin, ymax = min(all_values), max(all_values)
        span = ymax - ymin or (0.01 if key == 'dice' else 0.1)
        ymin = min(ymin - 0.18 * span, -0.001 if key == 'dice' else -0.05)
        ymax = max(ymax + 0.18 * span,  0.001 if key == 'dice' else  0.05)

        fig, axes = plt.subplots(1, len(panel_data),
                                 figsize=(max(10, 3.8 * len(panel_data)), 5.8),
                                 squeeze=False, sharey=True)
        axes = axes[0]

        for ax, (comp_label, organ_to_points, average_points, n_runs) in zip(axes, panel_data):
            ax.axhspan(0, ymax, alpha=0.12, zorder=0)
            ax.axhspan(ymin, 0, alpha=0.08, zorder=0)
            ax.axhline(0, color=INK, lw=1.6, zorder=1)
            categories = [organ_to_points[organ] for organ in organs] + [average_points]
            labels = [organ_labels.get(o, o[:2].title()) for o in organs] + ['Avg']

            for i, points in enumerate(categories):
                if not points:
                    continue
                values_arr = np.array([change for _, _, change in points], dtype=float)
                color = ORGAN_COLOR.get(organs[i], FG_COLOR) if i < len(organs) else INK
                if len(values_arr) > 1 and np.ptp(values_arr) > 1e-12:
                    parts = ax.violinplot([values_arr], positions=[i], widths=0.72,
                                          showmeans=False, showmedians=False,
                                          showextrema=False)
                    for body in parts['bodies']:
                        body.set_facecolor(color)
                        body.set_edgecolor(color)
                        body.set_alpha(0.18)
                        body.set_linewidth(1.0)

                # Fold positions are stable; seeds are offset within each fold.
                for fold, seed, change in points:
                    fold_offset = (folds.index(fold) - (len(folds)-1)/2) * 0.065
                    seed_offset = (seed % 5 - 2) * 0.014
                    ax.scatter(i + fold_offset + seed_offset, change, s=30,
                               color=fold_colors[fold], alpha=0.85,
                               edgecolors='white', linewidths=0.35, zorder=3)

                mean = float(values_arr.mean())
                ax.plot(i, mean, 'D' if i < len(organs) else '*',
                        color=INK if i == len(organs) else color,
                        ms=7 if i < len(organs) else 12,
                        mec='white', mew=1.0, zorder=4)
                ax.annotate(f'{mean:+.{digits}f}', (i, mean),
                            xytext=(0, 10 if mean >= 0 else -14),
                            textcoords='offset points', ha='center',
                            fontsize=8.5, fontweight='bold', color=INK,
                            path_effects=[withStroke(linewidth=2.5, foreground='white')],
                            zorder=5)

            ax.set_xticks(np.arange(len(categories)), labels, fontsize=10)
            ax.set_ylim(ymin, ymax)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{comp_label}\n(n={n_runs} matched runs)',
                         fontsize=10.5, loc='left', fontweight='bold')
            clean_axis(ax)
            ax.text(0.98, 0.98, 'Improvement', ha='right', va='top',
                    transform=ax.transAxes, fontsize=9, fontweight='bold')
            ax.text(0.98, 0.02, 'Worse', ha='right', va='bottom',
                    transform=ax.transAxes, fontsize=9, fontweight='bold')

        # Shared legend: point color indicates fold (including Avg points).
        legend_items = [Line2D([0], [0], marker='o', linestyle='none',
                               markerfacecolor=fold_colors[f], markeredgecolor='white',
                               markersize=7, label=f'Fold {f}') for f in folds]
        legend_items.append(Line2D([0], [0], marker='D', linestyle='none',
                                   color=INK, markersize=7, label='Mean change'))
        fig.legend(handles=legend_items, loc='lower center', ncol=len(legend_items),
                   bbox_to_anchor=(0.5, 0.02), frameon=False)

        title_metric = key.upper() if key == 'hd95' else key.capitalize()
        decorate(fig,
                 f'Upgraded ENet: change from Dice + CE ({title_metric})',
                 subtitle='Each point = one matched fold–seed run · point color = fold · above 0 = improvement',
                 footnote_text='Positive = improvement (higher Dice, or lower ASSD/HD95). '
                               'Avg = mean change across four organs per run.')
        tighten(fig, 0.5, 0.35)
        fig.subplots_adjust(bottom=0.19, top=0.78, wspace=0.28)
        save(plt, fig, out, f'multifold_loss_variants_gain_{key}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--groups', nargs='+', type=group_spec, metavar='LABEL=DIR_PREFIX',
                        default=[
                            group_spec('Dice + CE=/home/scur0049/ai4mi_project/metrics/full_cv4_enet_dice_ce'),
                            group_spec('Weighted CE + Dice=/home/scur0049/ai4mi_project/metrics/full_cv4_enet_weighted_dice_ce_preprocessing'),
                            group_spec('Constant boundary=/home/scur0049/ai4mi_project/metrics/full_cv4_enet_dice_ce_boundary_preprocessing'),
                            group_spec('Increasing boundary=/home/scur0049/ai4mi_project/metrics/full_cv4_enet_dice_ce_boundary_v2'),
                        ])
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--out', type=Path, default=REPO / 'figures/multifold_loss_variants_v3')
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
