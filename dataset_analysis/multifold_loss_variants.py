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
Writes PNG + PDF for Dice, HD95, ASSD and validation curves, plus per-run CSV.
"""

import argparse
import csv
import json
import math
import sys
import textwrap
from pathlib import Path

import numpy as np
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--groups', nargs='+', type=group_spec, required=True, metavar='LABEL=DIR_PREFIX')
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--out', type=Path, default=REPO / 'figures/multifold_loss_variants')
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
    write_csv(args.out / 'multifold_loss_variants.csv', table)
    print(f'{len(groups)} variants; {sum(len(r) for _, r in groups)} runs -> {args.out}')


if __name__ == '__main__':
    main()
