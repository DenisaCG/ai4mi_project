"""Seven self-contained figures, with all numerical summaries supplied by the analysis."""
from pathlib import Path

import numpy as np

from style import BACKGROUND, COLORS, NAMES, clean_axis, frame, pyplot, save


def select_shape_examples(rows):
    """Volume extremes and closest-to-median case; ties resolve by patient ID."""
    selected = []
    for k in NAMES:
        rs = [r for r in rows if int(r['class_id']) == k]
        median = np.median([float(r['volume_ml']) for r in rs])
        choices = {
            'small': min(rs, key=lambda r: (float(r['volume_ml']), r['patient_id'])),
            'typical': min(rs, key=lambda r: (round(abs(float(r['volume_ml']) - median), 9), r['patient_id'])),
            'large': min(rs, key=lambda r: (-float(r['volume_ml']), r['patient_id'])),
        }
        for category, row in choices.items():
            selected.append(dict(row, category=category,
                filename=f'{NAMES[k].lower()}_{category}_volume.png'))
    return selected


def orthogonal_plane(array, axis, index):
    """Index-order plane transposed for imshow(origin='lower'); no resampling."""
    return np.take(array, index, axis=axis).T


def shape_examples(output, original):
    """Nine original CT/GT examples; descriptors are read verbatim from the CSV."""
    import nibabel as nib
    from matplotlib.colors import to_rgba
    from matplotlib.transforms import blended_transform_factory
    from utils import read_csv, write_csv

    plt = pyplot(output)
    selected = select_shape_examples(read_csv(output / 'tables/shape_descriptors_3d.csv'))
    records = []
    for row in selected:
        k, patient = int(row['class_id']), row['patient_id']
        root = original / patient
        ct_nii = nib.load(root / f'{patient}.nii.gz')
        gt_nii = nib.load(root / 'GT.nii.gz')
        if (ct_nii.shape != gt_nii.shape or not np.allclose(ct_nii.affine, gt_nii.affine)
                or nib.aff2axcodes(gt_nii.affine) != ('L', 'P', 'S')
                or not np.allclose(gt_nii.affine[:3, :3], np.diag(np.diag(gt_nii.affine[:3, :3])))
                or gt_nii.header.get_xyzt_units()[0] != 'mm'):
            raise ValueError(f'Examples require aligned axial LPS CT/GT in mm: {patient}')
        ct = ct_nii.get_fdata(dtype=np.float32)
        mask = np.asanyarray(gt_nii.dataobj) == k
        centre = np.array([float(row[f'centroid_{a}']) for a in 'ijk'])
        indices = np.rint(centre).astype(int)
        spacing = np.abs(np.diag(gt_nii.affine[:3, :3]))
        occupied = [np.flatnonzero(np.any(mask, axis=tuple(j for j in range(3) if j != a))) for a in range(3)]
        bounds = [(max(-.5, v[0] - 45 / spacing[a]), min(mask.shape[a] - .5, v[-1] + 45 / spacing[a]))
                  for a, v in enumerate(occupied)]
        fig = frame(plt, f"{NAMES[k]} — {patient} — {row['category'].capitalize()} volume",
            f"Volume: {float(row['volume_ml']):,.2f} mL  |  SI extent: {float(row['si_extent_mm']):.2f} mm  |  Normalized SI centroid: {float(row['normalized_si_centroid']):.3f}",
            'Original CT + selected-organ ground truth only. White cross: projected 3D centroid; nearest voxel-centre planes.\n'
            'Colored bracket: whole-organ SI span, including gaps (not just this plane). Coronal/sagittal views retain full scan SI coverage.\n'
            'CT window: −160 to 240 HU (level 40, width 400). Transverse crops add 45 mm context; physical aspect is preserved.')
        axes = fig.subplots(1, 3)
        fig.subplots_adjust(left=.045, right=.955, bottom=.19, top=.79, wspace=.18)
        # LPS index axes: i toward left, j posterior, k superior.
        for ax, (axis, horizontal, vertical, title, directions) in zip(axes, (
                (2, 0, 1, 'Axial', ('R', 'L', 'A', 'P')),
                (1, 0, 2, 'Coronal', ('R', 'L', 'I', 'S')),
                (0, 1, 2, 'Sagittal', ('A', 'P', 'I', 'S')))):
            plane = orthogonal_plane(ct, axis, indices[axis])
            organ = orthogonal_plane(mask, axis, indices[axis])
            if not organ.any():
                raise ValueError(f'Centroid plane misses organ: {patient}, {k}, {title}')
            extent = [-.5 * spacing[horizontal], (mask.shape[horizontal] - .5) * spacing[horizontal],
                      -.5 * spacing[vertical], (mask.shape[vertical] - .5) * spacing[vertical]]
            ax.imshow(plane, cmap='gray', vmin=-160, vmax=240, origin='lower', extent=extent, interpolation='nearest')
            rgba = np.zeros((*organ.shape, 4))
            rgba[organ] = to_rgba(COLORS[k], .40)
            ax.imshow(rgba, origin='lower', extent=extent, interpolation='nearest')
            ax.contour(np.arange(organ.shape[1]) * spacing[horizontal],
                       np.arange(organ.shape[0]) * spacing[vertical], organ.astype(float),
                       levels=[.5], colors=[COLORS[k]], linewidths=.8)
            ax.plot(centre[horizontal] * spacing[horizontal], centre[vertical] * spacing[vertical],
                    '+', color='white', markersize=10, markeredgewidth=1.3)
            ax.set_xlim(np.array(bounds[horizontal]) * spacing[horizontal])
            ax.set_ylim((np.array(bounds[vertical]) * spacing[vertical]) if axis == 2 else extent[2:])
            ax.set_title(title, pad=18)
            ax.set_xticks([]); ax.set_yticks([])
            for label, x, y in zip(directions, [.025, .975, .5, .5], [.5, .5, .02, .98]):
                ax.text(x, y, label, transform=ax.transAxes, color='white', fontsize=11,
                        weight='bold', ha='center', va='center', bbox=dict(facecolor='black', alpha=.45, pad=1, edgecolor='none'))
            if title == 'Coronal':
                transform = blended_transform_factory(ax.transAxes, ax.transData)
                lo, hi = (occupied[2][0] - .5) * spacing[2], (occupied[2][-1] + .5) * spacing[2]
                ax.plot([.94, .94], [lo, hi], color=COLORS[k], linewidth=2, transform=transform)
                for z in (lo, hi):
                    ax.plot([.915, .965], [z, z], color=COLORS[k], linewidth=2, transform=transform)
        fig.savefig(output / 'examples' / row['filename'], dpi=220)
        plt.close(fig)
        records.append(dict(row, axial_index=indices[2], coronal_index=indices[1], sagittal_index=indices[0],
                            ct_window_min_hu=-160, ct_window_max_hu=240, organ_color=COLORS[k]))
    write_csv(output / 'tables/shape_example_selection.csv', records)


def cohort(inventory):
    return (f"{len(inventory)} patients: "
            f"{sum(r['split'] == 'train' for r in inventory)} training + "
            f"{sum(r['split'] == 'val' for r in inventory)} validation")


def class_distribution(output, frequency, inventory):
    """Two explicit denominators prevent background from hiding foreground differences."""
    plt = pyplot(output)
    rows = {r['class_id']: r for r in frequency if r['split'] == 'all'}
    total = sum(r['voxel_count'] for r in rows.values())
    foreground = sum(rows[k]['voxel_count'] for k in NAMES)
    fg_percent = 100 * foreground / total
    fig = frame(plt, "Class distribution: background and annotated organs",
        f"Original NIfTI ground truth  |  {cohort(inventory)}  |  Counts pooled across scans",
        "Background means label 0, including unannotated anatomy. Aorta annotation is intentionally excluded.\n"
        "The two panels use different denominators; counts are original voxels, not physical-volume totals.")
    fig.text(.10, .81, f"{total:,} total scan voxels", fontsize=20, weight='bold')
    fig.text(.10, .765, f"Background: {100 - fg_percent:.3f}%", color=BACKGROUND, fontsize=15, weight='bold')
    fig.text(.57, .765, f"Annotated foreground: {fg_percent:.3f}%", fontsize=15, weight='bold')
    ax = fig.add_axes([.10, .655, .81, .08])
    ax.barh(0, 100 - fg_percent, color=BACKGROUND, height=.65)
    ax.barh(0, fg_percent, left=100 - fg_percent, color='#202020', height=.65)
    ax.set(xlim=(0, 100), yticks=[], xlabel="Share of all scan voxels (%)")
    ax.spines['left'].set_visible(False)
    ax.tick_params(length=0)
    fig.text(.10, .535, f"{foreground:,} annotated foreground voxels", fontsize=22, weight='bold')
    ax = fig.add_axes([.23, .205, .68, .285])
    for k, y in zip(NAMES, (2, 1, 0)):
        fraction = 100 * rows[k]['voxel_count'] / foreground
        ax.barh(y, fraction, color=COLORS[k], height=.58)
        ax.text(fraction + 1.7, y, f"{fraction:.2f}%\n{rows[k]['voxel_count']:,} voxels",
                va='center', fontsize=14, weight='bold', linespacing=1.3)
    ax.set(xlim=(0, 108), ylim=(-.55, 2.55), yticks=[2, 1, 0],
           yticklabels=list(NAMES.values()), xlabel="Share of annotated foreground voxels (%)",
           ylabel="Annotated organ", xticks=[0, 25, 50, 75, 100])
    ax.spines['left'].set_visible(False)
    ax.tick_params(length=0, pad=9)
    save(fig, output, 'class_distribution.png', plt)


def shape_scatter(output, shapes, inventory):
    """One physical size/extent/location view; each point is a patient–organ pair."""
    plt = pyplot(output)
    fig = frame(plt, "3D Shape Descriptor Space of Annotated SegTHOR Structures",
        f"Original NIfTI ground truth  |  {cohort(inventory)}  |  One point = one patient × organ",
        "SI = superior–inferior. Normalized centroid: 0 = inferior scan limit; 1 = superior scan limit.\n"
        "Location is relative to each scan, not anatomical registration. Apparent separation is descriptive, not a clustering result.")
    ax = fig.add_axes([.04, .20, .88, .65], projection='3d')
    for k in NAMES:
        rs = [r for r in shapes if r['class_id'] == k and r['voxel_count']]
        ax.scatter([r['volume_ml'] for r in rs], [r['si_extent_mm'] for r in rs],
                   [r['normalized_si_centroid'] for r in rs], color=COLORS[k], s=65,
                   edgecolors='white', linewidths=.6, alpha=.9, depthshade=False,
                   label=f"{NAMES[k]} ({len(rs)} points)")
    largest = max(shapes, key=lambda r: r['volume_ml'])
    ax.text(largest['volume_ml'], largest['si_extent_mm'], largest['normalized_si_centroid'] + .055,
            f"{largest['patient_id']}\nlargest volume", fontsize=11, weight='bold')
    ax.set_xlabel("Organ volume (mL)", labelpad=16)
    ax.set_ylabel("Superior–inferior extent (mm)", labelpad=16)
    ax.set_zlabel("Normalized SI centroid (0–1)", labelpad=14)
    ax.set_zlim(0, 1)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_box_aspect((1.4, 1, 1), zoom=1.0)
    ax.view_init(elev=23, azim=-57)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor('#eeeeee')
        axis._axinfo['grid']['color'] = (.88, .88, .88, .45)
        axis._axinfo['grid']['linewidth'] = .6
    ax.tick_params(labelsize=11, pad=2)
    fig.legend(*ax.get_legend_handles_labels(), loc='upper right', bbox_to_anchor=(.955, .84),
               fontsize=12, labelspacing=.8)
    save(fig, output, 'shape_descriptors_3d.png', plt)


def shape_summary(output, shapes, inventory):
    """Three coordinated distributions make the 3D view quantitatively readable."""
    plt = pyplot(output)
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    fig = frame(plt, "Size, longitudinal extent and scan location differ by organ",
        f"Original NIfTI ground truth  |  {cohort(inventory)}  |  Each dot is one patient",
        "Boxes: median and middle 50% of patients; whiskers: 1.5 × interquartile range. All patient points are shown.\n"
        "Volume uses a log scale. Normalized centroid runs from inferior (0) to superior (1) within each scan.")
    fields = [('volume_ml', 'Physical organ volume (mL)', 'Volume'),
              ('si_extent_mm', 'Superior–inferior extent (mm)', 'Extent'),
              ('normalized_si_centroid', 'Normalized SI centroid (0–1)', 'Location')]
    for j, (field, label, title) in enumerate(fields):
        ax = fig.add_subplot(1, 3, j + 1)
        for k in NAMES:
            rs = sorted([r for r in shapes if r['class_id'] == k and r['voxel_count']], key=lambda r:r['patient_id'])
            values = [r[field] for r in rs]
            box = ax.boxplot([values], positions=[k], widths=.5, showfliers=False, patch_artist=True,
                             medianprops={'color': COLORS[k], 'linewidth': 2.8},
                             whiskerprops={'color':'#777777'}, capprops={'color':'#777777'})
            box['boxes'][0].set(facecolor=COLORS[k], alpha=.15, edgecolor=COLORS[k])
            ax.scatter(k + np.linspace(-.16, .16, len(rs)), values, color=COLORS[k],
                       s=30, alpha=.8, edgecolors='white', linewidths=.4, zorder=3)
        ax.set_xticks(list(NAMES), list(NAMES.values()), rotation=20, ha='right')
        ax.set(title=title, xlabel='Annotated organ', ylabel=label)
        if field == 'volume_ml':
            ax.set_yscale('log')
            ax.set_yticks([20, 50, 100, 200, 500, 1000, 2000])
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
        elif field == 'normalized_si_centroid':
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(bottom=0)
        clean_axis(ax)
    save(fig, output, 'shape_descriptors_summary.png', plt)


def area_through_scan(output, trends, inventory):
    """Patient-weighted processed areas: zero slices included, one explicit percent unit."""
    plt = pyplot(output)
    fig = frame(plt, "The 2D model sees different target sizes along the scan",
        f"Processed 256 × 256 ground-truth masks  |  {cohort(inventory)}  |  Empty slices included",
        "Within each of 10 scan-position bins, average slices per patient, then take the median across patients.\n"
        "Shading: middle 50% of patient means, not confidence intervals. Scan position does not align anatomy between patients.")
    axes = fig.subplots(1, 3, sharex=True)
    for ax, k in zip(axes, NAMES):
        rs = [r for r in trends if r['split'] == 'all' and r['class_id'] == k]
        x = [r['z_midpoint'] for r in rs]
        ax.plot(x, [100*r['median'] for r in rs], 'o-', color=COLORS[k], markersize=5)
        ax.fill_between(x, [100*r['p25'] for r in rs], [100*r['p75'] for r in rs], color=COLORS[k], alpha=.18)
        ax.set(title=NAMES[k], xlim=(0,1), ylim=(0, None),
               xlabel='Normalized scan position (0–1)', ylabel='Target area (% of slice)')
        ax.set_xticks([0,.25,.5,.75,1], ['0','.25','.50','.75','1'])
        clean_axis(ax)
    fig.text(.50, .135, "0 = first / inferior slice     →     1 = last / superior slice     •     Vertical scales differ by organ",
             ha='center', fontsize=12, weight='bold')
    save(fig, output, 'target_area_through_scan.png', plt)


def baseline_dice(output, patients):
    """All five patient scores and equal-patient class means on the original 3D grid."""
    plt = pyplot(output)
    n = len({r['patient_id'] for r in patients})
    fig = frame(plt, "Baseline volumetric segmentation performance by organ",
        f"Original ENet baseline  |  {n} validation patients  |  Reconstructed predictions vs original NIfTI ground truth",
        "Each labeled dot is one patient; the thick horizontal mark is the equal-patient mean for that organ.\n"
        "Dice = 2 × overlap / (ground-truth + predicted volume). These are full-volume scores, not averages of slice Dice.")
    ax = fig.add_subplot(111)
    for k in NAMES:
        rs = sorted([r for r in patients if r['class_id'] == k], key=lambda r:r['patient_id'])
        vals = np.array([r['dice_3d'] for r in rs], float)
        if not np.isfinite(vals).all():
            raise ValueError('The primary baseline figure requires defined 3D Dice for every patient-organ pair')
        offsets = np.linspace(-.25,.25,len(rs))
        for r, x in zip(rs, k+offsets):
            ax.scatter(x, r['dice_3d'], s=90, color=COLORS[k], edgecolor='white', linewidth=1, zorder=4)
            ax.annotate(r['patient_id'].replace('Patient_', 'P'), (x,r['dice_3d']),
                        xytext=(0, -16 if 0 < vals.mean() - r['dice_3d'] < .08 else 9),
                        textcoords='offset points', ha='center', fontsize=11, weight='bold')
        mean = vals.mean()
        ax.hlines(mean,k-.36,k+.36,color=COLORS[k],linewidth=4,zorder=3)
        ax.text(k,1.035,f"Mean {mean:.3f}",color=COLORS[k],ha='center',fontsize=17,weight='bold')
    ax.set(xlim=(.45,3.55),ylim=(-.04,1.12),ylabel='Patient-level 3D Dice (0–1)',xlabel='Annotated organ')
    ax.set_xticks(list(NAMES), list(NAMES.values()))
    ax.set_yticks(np.arange(0,1.01,.2))
    clean_axis(ax)
    save(fig, output, 'baseline_3d_dice_by_class.png', plt)


def baseline_size(output, bins, patients):
    """Class-specific area quintiles; patient mean/IQR, no raw-slice pseudo-replication."""
    plt = pyplot(output)
    from matplotlib.ticker import NullFormatter
    n = len({r['patient_id'] for r in patients})
    fig = frame(plt, "Baseline Dice and target size: three different patterns",
        f"Original ENet baseline  |  {n} validation patients  |  Processed 256 × 256 slices with the organ present in ground truth",
        "Five area-quantile bins per organ. Average Dice within each patient/bin, then average those patient means equally.\n"
        "Shading: middle 50% of patient means. n = contributing patients; composition differs by bin. Association is not causation.")
    axes = fig.subplots(1,3,sharey=True)
    for ax,k in zip(axes,NAMES):
        rs = [r for r in bins if r['class_id']==k and r['bin_type']=='area_quantile']
        x = [r['median_gt_area'] for r in rs]
        ax.plot(x,[r['patient_mean_dice_mean'] for r in rs], 'o-', color=COLORS[k], markersize=7)
        ax.fill_between(x,[r['patient_mean_dice_p25'] for r in rs],
                        [r['patient_mean_dice_p75'] for r in rs],color=COLORS[k],alpha=.18)
        for xx,r in zip(x,rs):
            ax.annotate(f"n={r['num_patients']}",(xx,r['patient_mean_dice_mean']),
                        xytext=(0,11),textcoords='offset points',ha='center',fontsize=10)
        ax.set(xscale='log',title=NAMES[k],ylim=(-.03,1.13),
               xlabel='')
        ax.set_xticks(x,[f'{v:,.0f}' for v in x],rotation=40,ha='right')
        ax.xaxis.set_minor_formatter(NullFormatter())
        clean_axis(ax)
    axes[0].set_ylabel('Mean GT-positive slice Dice (0–1)')
    axes[0].set_yticks(np.arange(0,1.01,.2))
    fig.text(.50,.155,'Ground-truth area in processed 256 × 256 slice (pixels; logarithmic scale)',ha='center',fontsize=14,weight='bold')
    fig.text(.50,.115,'Horizontal coordinate = median target area within each bin; area ranges differ by organ.',ha='center',fontsize=11)
    save(fig,output,'baseline_dice_vs_target_size.png',plt)


def baseline_position(output,bins,patients):
    """Beginning/middle/end with clear labels and equal patient weights."""
    plt=pyplot(output)
    n=len({r['patient_id'] for r in patients})
    fig=frame(plt,'Baseline Dice across the annotated organ extent',
        f'Original ENet baseline  |  {n} validation patients  |  Processed slices with the organ present in ground truth',
        'Position runs from the first to last organ-positive slice in each patient. Average slices per patient/region, then patients equally.\n'
        'Shading: middle 50% of patient means. Size and position vary together; these descriptive differences are not causal effects.')
    axes=fig.subplots(1,3,sharey=True)
    for ax,k in zip(axes,NAMES):
        rs=[r for r in bins if r['class_id']==k and r['bin_type']=='organ_z']
        x=[r['bin_id'] for r in rs]
        y=[r['patient_mean_dice_mean'] for r in rs]
        ax.plot(x,y,'o-',color=COLORS[k],markersize=8)
        ax.fill_between(x,[r['patient_mean_dice_p25'] for r in rs],
                        [r['patient_mean_dice_p75'] for r in rs],color=COLORS[k],alpha=.18)
        for xx,yy in zip(x,y):
            ax.annotate(f'{yy:.2f}',(xx,yy),xytext=(0,12),textcoords='offset points',
                        ha='center',weight='bold',fontsize=14,color=COLORS[k])
        ax.set(title=NAMES[k],ylim=(-.03,1.04),xlim=(-.2,2.2),
               xlabel='')
        ax.set_xticks([0,1,2],['Beginning\nFirst 20%','Middle\nMiddle 60%','End\nLast 20%'])
        ax.tick_params(axis='x',labelsize=11)
        clean_axis(ax)
    axes[0].set_ylabel('Mean GT-positive slice Dice (0–1)')
    axes[0].set_yticks(np.arange(0,1.01,.2))
    fig.text(.50,.155,'Position within annotated organ extent',ha='center',fontsize=14,weight='bold')
    fig.text(.50,.115,'Normalized organ position: [0, 0.2) beginning   •   [0.2, 0.8) middle   •   [0.8, 1] end',ha='center',fontsize=11)
    save(fig,output,'baseline_dice_by_organ_position.png',plt)


def dataset_figures(output,frequency,shapes,trends,inventory):
    class_distribution(output,frequency,inventory)
    shape_scatter(output,shapes,inventory)
    shape_summary(output,shapes,inventory)
    area_through_scan(output,trends,inventory)


def baseline_figures(output,bins,patients):
    baseline_dice(output,patients)
    baseline_size(output,bins,patients)
    baseline_position(output,bins,patients)
