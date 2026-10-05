"""Loss comparison on the CORRECTED SEGTHOR labels: Dice, Dice + CE (CE over all / foreground classes) and CE
(foreground) against the all-class CE baseline, from the runs copied into metrics/.

    python dataset_analysis/compare_losses.py [--seeds 0 1 2 3 4] [--out results/loss_comparison]

Groups are compared on the seeds they all share (or --seeds), so every group has the same n. A group with no finished
runs is skipped with a warning. Undefined HD95/ASSD (empty prediction) is never scored as 0 or dropped silently:
it is marked in the plots and counted in the tables.

Writes 6 annotated figures (.png + .pdf; the overview is slide-sized), 2 tables (.csv + .md) and a manifest with the seeds and configs used.
"""
import argparse
import csv
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from PIL import Image

from utils import CLASSES, COLORS, INK, NAMES, REPO, clean_axis, decorate, pyplot, write_csv
from plot_style import EARTH, PALETTE, legend_below

METRICS, RUNS, DATA = REPO / "metrics", REPO / "runs", REPO / "data/SEGTHOR_corrected"
BASELINE = "ce_all"
# (key, label, experiment): one per loss variant, plus the existing all-class CE baseline on the corrected data.
SPECS = (("dice_fg", "Dice (fg)", "segthor_enet_dice_fg_corrected"),
         ("dice_all", "Dice (all)", "segthor_enet_dice_all_corrected"),
         ("dice_ce_all", "CE (all) + Dice (fg)", "segthor_enet_dice_ce_all_corrected"),
         ("dice_ce_fg", "CE (fg) + Dice (fg)", "segthor_enet_dice_ce_fg_corrected"),
         ("ce_fg", "CE (fg)", "segthor_enet_ce_fg_corrected"),
         (BASELINE, "CE (all), baseline", "segthor_enet_ce_repro_corrected_fixed"))
COLOR = {"dice_fg": PALETTE[0], "dice_all": PALETTE[2], "dice_ce_all": PALETTE[1], "dice_ce_fg": PALETTE[4],
         "ce_fg": PALETTE[5], BASELINE: "#6B6B6B"}
PANELS = [(CLASSES[k], NAMES[k]) for k in CLASSES] + [("fg", "Foreground mean")]  # (summary key, title)


@dataclass
class Group:
    key: str
    label: str
    experiment: str
    runs: list = field(default_factory=list)

    @property
    def color(self):
        return COLOR[self.key]

    def scores(self, metric: str, key: str) -> np.ndarray:
        """One value per seed: eval mean over the validation patients. NaN where undefined (empty prediction)."""
        return np.array([np.nan if r["eval"][f"val_{metric}_{key}"] is None else float(r["eval"][f"val_{metric}_{key}"])
                         for r in self.runs])


def load_run(experiment: str, seed: int) -> dict:
    d = METRICS / experiment / f"seed{seed}"
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    if Path(cfg["data"]["root"]).name != "SEGTHOR_corrected":
        raise ValueError(f"{d}: trained on {cfg['data']['root']}, not the corrected labels")
    summary = json.loads((d / "summary.json").read_text())
    for metric in ("dice", "hd95", "assd"):
        for key in [CLASSES[k] for k in CLASSES] + ["fg"]:
            if f"val_{metric}_{key}" not in summary["eval"]:
                raise KeyError(f"{d}/summary.json has no eval.val_{metric}_{key}: not all organs were evaluated")
    with (d / "epochs.csv").open() as f:
        curve = np.array([float(r["val_dice_fg"]) for r in csv.DictReader(f)])
    return {"seed": seed, "cfg": cfg, "eval": summary["eval"], "best_epoch": summary["best_epoch"], "curve": curve}


def finished_seeds(experiment: str) -> list[int]:
    return sorted(int(d.name.removeprefix("seed")) for d in (METRICS / experiment).glob("seed*")
                  if (d / "summary.json").exists() and "eval" in json.loads((d / "summary.json").read_text()))


def load_groups(seeds: list[int] | None) -> list[Group]:
    available = {key: finished_seeds(exp) for key, _, exp in SPECS if (METRICS / exp).is_dir()}
    for key, label, exp in SPECS:
        if not available.get(key):
            print(f"WARNING: no finished runs for {label} ({exp}); skipped")
    available = {k: v for k, v in available.items() if v}
    if BASELINE not in available:
        raise FileNotFoundError("the CE (all) baseline runs are missing; nothing to compare against")
    seeds = seeds or sorted(set.intersection(*map(set, available.values())))
    if not seeds:
        raise ValueError(f"the groups share no seed: {available}")
    groups = []
    for key, label, exp in SPECS:
        if key in available:
            missing = set(seeds) - set(available[key])
            if missing:
                raise FileNotFoundError(f"{label} ({exp}) has no finished run for seeds {sorted(missing)}")
            groups.append(Group(key, label, exp, [load_run(exp, s) for s in seeds]))
    return groups


def mean_std(v: np.ndarray) -> tuple[float, float, int]:
    """Mean and sample std over the finite seeds, and how many are finite."""
    f = v[np.isfinite(v)]
    return (float(f.mean()) if len(f) else np.nan, float(f.std(ddof=1)) if len(f) > 1 else np.nan, len(f))


def collapsed(g: Group, key: str, below: float) -> list[int]:
    return [r["seed"] for r, d in zip(g.runs, g.scores("dice", key)) if d < below]


def save(plt, fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def n_patients(groups) -> int:
    v = groups[0].runs[0]["eval"]["val_patients"]
    return len(v) if isinstance(v, list) else int(v)


def seed_note(groups) -> str:
    seeds = [r["seed"] for r in groups[0].runs]
    return f"{len(seeds)} seeds per loss (seeds {seeds}), {n_patients(groups)} validation patients"


def strip_figure(plt, groups, out, metric, title, unit, log, below):
    """Figures 1 and 2: one panel per organ + foreground mean, one dot per seed, black bar = mean over seeds."""
    fig, axes = plt.subplots(1, len(PANELS), figsize=(3.3 * len(PANELS) + .6, 5.8), sharey=True)
    base = next(g for g in groups if g.key == BASELINE)
    allv = np.concatenate([g.scores(metric, k) for g in groups for k, _ in PANELS])
    finite = allv[np.isfinite(allv)]
    for ax in axes:
        if log:
            ax.set_yscale("log")
            ax.set_ylim(finite.min() / 1.8, finite.max() * 1.8)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        else:
            ax.set_ylim(-.08, 1.03)
            ax.set_yticks(np.arange(0, 1.01, .2))
    for ax, (key, name) in zip(axes, PANELS):
        clean_axis(ax, "y")
        ax.set_title(name, pad=26, fontsize=13)
        ax.set_xlim(-.6, len(groups) - .4)
        ticks = []
        for gi, g in enumerate(groups):
            v = g.scores(metric, key)
            ok = np.isfinite(v)
            offs = np.linspace(-.22, .22, len(v)) if len(v) > 1 else np.zeros(1)
            ax.scatter(gi + offs[ok], v[ok], s=70, color=g.color, edgecolor="white", linewidth=1, zorder=3,
                       label=g.label if key == PANELS[0][0] else None)
            m, _, k = mean_std(v)
            if k:
                ax.hlines(m, gi - .33, gi + .33, color=INK, linewidth=2.6, zorder=4)
                ax.text(gi, 1.012, f"{m:.2f}" if metric == "dice" else f"{m:.0f}", transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=11, weight="bold", color=INK)
            ticks.append(f"n={k}" if k == len(v) else f"n={k}/{len(v)}")
            for i, (o, s, seed) in enumerate(zip(offs, v, [r["seed"] for r in g.runs])):
                if metric == "dice" and key != "fg" and s < below:  # ring + seed number on collapsed seeds
                    ax.scatter(gi + o, s, s=230, facecolor="none", edgecolor=INK, linewidth=1.4, zorder=5)
                    ax.annotate(f"s{seed}", (gi + o, s), xytext=(0, 16 + 11 * (i % 2)), textcoords="offset points",
                                ha="center", fontsize=8, color=INK)  # above the ring (below hits the n= tick labels); alternate heights
                if not np.isfinite(s):  # undefined distance: x at the axis foot, labelled with its seed
                    ax.scatter(gi + o, .03, marker="x", s=55, color=g.color, linewidth=2, zorder=5,
                               transform=ax.get_xaxis_transform(), clip_on=False)
                    ax.annotate(f"s{seed}", (gi + o, .03), xycoords=ax.get_xaxis_transform(), xytext=(0, 8),
                                textcoords="offset points", ha="center", fontsize=8, color=INK)
        bm = mean_std(base.scores(metric, key))[0]
        if np.isfinite(bm):
            ax.axhline(bm, color=COLOR[BASELINE], linestyle="--", linewidth=1.1, zorder=2,
                       label=f"{base.label} mean" if key == PANELS[0][0] else None)
        ax.set_xticks(range(len(groups)), ticks, fontsize=9)
    legend_below(axes[0], ncol=3)
    fg = {g.label: mean_std(g.scores(metric, "fg"))[0] for g in groups}
    best = (max if metric == "dice" else min)((k for k in fg if np.isfinite(fg[k])), key=fg.get)
    headline = f"{best} has the {'highest' if metric == 'dice' else 'lowest'} mean {title}: {fg[best]:.{2 if metric == 'dice' else 0}f}{unit}"
    note = ("Dot = one seed (mean over the validation patients); black bar and number = mean over seeds; dashed line = "
            f"{base.label} mean. ")
    note += ("Ringed dots (labelled with the seed) = collapsed organ, Dice < %.2f." % below if metric == "dice" else
             "x at the foot of a panel = HD95 undefined for that seed (organ never predicted); n=k/N counts defined seeds. "
             "Log scale.")
    decorate(fig, headline, subtitle=f"{title} per organ, by loss. {seed_note(groups)}.", footnote_text=note)
    save(plt, fig, out, f"loss_comparison_{metric}_by_organ")


def sees_background(loss: dict) -> bool:
    """Does any term of the loss include class 0? Defaults follow src/losses: CE covers all classes, Dice the foreground."""
    kw = loss["kwargs"]
    terms = {"cross_entropy": {"idk": True}, "soft_dice": {"idk": False},
             "dice_ce": {"ce_idk": True, "dice_idk": False}}[loss["name"]]
    return any(0 in kw[k] if k in kw else default for k, default in terms.items())


def overview_figure(plt, groups, out, below):
    """Figure 6 (slide-sized): one row per loss, best on top, coloured by whether the loss sees background."""
    fg = [mean_std(g.scores("dice", "fg"))[0] for g in groups]
    ordered = [groups[i] for i in np.argsort(fg)[::-1]]
    organs = [k for k, _ in PANELS[:-1]]
    total = len(groups[0].runs) * len(organs)  # organ-seeds per loss
    sees = {g.key: sees_background(g.runs[0]["cfg"]["loss"]) for g in groups}
    color = {True: EARTH[1], False: "#9A9A9A"}

    def hits(flag: bool) -> tuple[int, int]:
        members = [g for g in groups if sees[g.key] == flag]
        return sum(len(collapsed(g, o, below)) for g in members for o in organs), total * len(members)

    fig, ax = plt.subplots(figsize=(13, 6.0))
    clean_axis(ax, "x")
    hi = min(1.0, np.ceil((np.nanmax([g.scores("dice", "fg") for g in groups]) + .05) * 10) / 10)
    ax.set_xlim(0, hi)
    ax.set_ylim(len(ordered) - .5, -.5)  # inverted: best on top
    base = next(g for g in groups if g.key == BASELINE)
    ax.axvline(mean_std(base.scores("dice", "fg"))[0], color=COLOR[BASELINE], linestyle="--", linewidth=1.2, zorder=1,
               label=f"{base.label} mean")
    head = dict(transform=ax.get_yaxis_transform(), clip_on=False, fontsize=13, color=INK, weight="bold", va="bottom")
    ax.text(1.03, -.62, "Collapsed\norgan-seeds", ha="left", **head)
    ax.text(1.24, -.62, "HD95\n(mm)", ha="left", **head)
    for y, g in enumerate(ordered):
        v = g.scores("dice", "fg")
        offs = np.linspace(-.16, .16, len(v)) if len(v) > 1 else np.zeros(1)
        ax.scatter(v, y + offs, s=130, color=color[sees[g.key]], edgecolor="white", linewidth=1.2, zorder=3)
        m = mean_std(v)[0]
        ax.vlines(m, y - .32, y + .32, color=INK, linewidth=3, zorder=4)
        ax.text(np.nanmax(v) + .012, y, f"{m:.2f}", va="center", fontsize=15, weight="bold", color=INK)
        k = sum(len(collapsed(g, o, below)) for o in organs)
        ax.text(1.03, y, f"{k} / {total}", va="center", ha="left", fontsize=15, transform=ax.get_yaxis_transform(),
                clip_on=False, color="#B5533C" if k else INK, weight="bold" if k else "normal")
        ax.text(1.24, y, f"{mean_std(g.scores('hd95', 'fg'))[0]:.0f}", va="center", ha="left", fontsize=15,
                transform=ax.get_yaxis_transform(), clip_on=False, color=INK)
    ax.set_yticks(range(len(ordered)), [g.label for g in ordered], fontsize=15)
    ax.tick_params(axis="x", labelsize=13)
    ax.set_xlabel("3D Dice, mean over the four organs", fontsize=14)
    ax.scatter([], [], s=130, color=color[True], label="Loss sees background")
    ax.scatter([], [], s=130, color=color[False], label="Loss ignores background")
    legend_below(ax, ncol=3)
    (a, na), (b, nb) = hits(True), hits(False)
    decorate(fig, f"Collapsed organs: {a}/{na} when the loss sees background, {b}/{nb} when it ignores it",
             subtitle=f"Mean 3D Dice by loss, best on top. {seed_note(groups)}.",
             footnote_text="Dot = one seed; black bar and number = mean over seeds; dashed line = CE (all) baseline mean. "
                           f"Collapsed = an organ's 3D Dice below {below:g} in a seed, counted over organs x seeds. "
                           "HD95 = mean over seeds of the foreground-mean 3D HD95. Sees background = a class list "
                           "in the loss includes class 0 (from each run's config).")
    save(plt, fig, out, "loss_comparison_overview")


def curves_figure(plt, groups, out):
    """Figure 3: validation Dice (2D, patient-level, the checkpoint-selection metric) per epoch."""
    fig, ax = plt.subplots(figsize=(11, 6.2))
    clean_axis(ax, "y")
    epochs = {len(r["curve"]) for g in groups for r in g.runs}
    if len(epochs) != 1:
        raise ValueError(f"runs have different epoch counts: {sorted(epochs)}")
    x = np.arange(1, epochs.pop() + 1)
    ends = []
    for g in groups:
        a = np.stack([r["curve"] for r in g.runs])
        ax.fill_between(x, a.min(0), a.max(0), color=g.color, alpha=.16, linewidth=0)
        ax.plot(x, a.mean(0), color=g.color, linewidth=2.4, label=g.label)
        best = np.mean([r["best_epoch"] for r in g.runs]) + 1  # summary best_epoch is 0-based
        ax.scatter(best, np.interp(best, x, a.mean(0)), s=95, color=g.color, edgecolor=INK, linewidth=1.4, zorder=5)
        ends.append(a.mean(0)[-1])
    order, y = np.argsort(ends), np.array(ends)
    gap = .035 * max(1e-9, max(ends))
    for i in range(1, len(order)):  # nudge end labels apart
        y[order[i]] = max(y[order[i]], y[order[i - 1]] + gap)
    for g, yy, e in zip(groups, y, ends):
        ax.text(x[-1] + .4, yy, f"{e:.2f}", color=g.color, fontsize=11, weight="bold", va="center")
    ax.set_xlim(1, x[-1] + 3)
    ax.set_ylim(0, None)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Dice, foreground mean")
    legend_below(ax, ncol=3)
    decorate(fig, "Validation Dice over training, by loss",
             subtitle=f"Mean over seeds with min-max band. {seed_note(groups)}.",
             footnote_text="Number at the line end = mean value at the last epoch; ringed dot = mean best epoch "
                           "(checkpoint used for the 3D evaluation). 2D patient-level Dice at 256x256, the metric that "
                           "selects the checkpoint. Train loss is not shown: its scale differs between losses.")
    save(plt, fig, out, "loss_comparison_val_dice_curves")


def heatmap_figure(plt, groups, out, below):
    """Figure 4: number of seeds where an organ collapsed (3D Dice < below)."""
    fig, ax = plt.subplots(figsize=(9.4, 1.1 * len(groups) + 2.6))
    n = len(groups[0].runs)
    cmap = LinearSegmentedColormap.from_list("collapse", ["#FFFFFF", "#B5533C"])
    counts = np.array([[len(collapsed(g, k, below)) for k, _ in PANELS[:-1]] for g in groups])
    ax.imshow(counts, cmap=cmap, vmin=0, vmax=n, aspect="auto")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(range(counts.shape[1]), [name for _, name in PANELS[:-1]])
    ax.set_yticks(range(len(groups)), [g.label for g in groups])
    ax.tick_params(length=0)
    for i, g in enumerate(groups):
        for j, (key, _) in enumerate(PANELS[:-1]):
            seeds = collapsed(g, key, below)
            ax.text(j, i - (.12 if seeds else 0), f"{len(seeds)}/{n}", ha="center", va="center", fontsize=14, weight="bold",
                    color="white" if counts[i, j] > .6 * n else INK)
            if seeds:
                ax.text(j, i + .25, "s" + ",".join(map(str, seeds)), ha="center", va="center", fontsize=8,
                        color="white" if counts[i, j] > .6 * n else INK)
    decorate(fig, f"{int(counts.sum())} organ collapses across {len(groups)} losses",
             subtitle=f"Seeds where an organ's 3D Dice is below {below:g}. {seed_note(groups)}.",
             footnote_text="Cell = collapsed seeds / seeds, with the seed numbers underneath. Darker = more collapses.")
    save(plt, fig, out, "loss_comparison_collapse_heatmap")


def pick_slice(patient: str | None, scale: int) -> tuple[str, int]:
    """Validation slice where all organs are visible and the smallest organ is largest (independent of any model)."""
    best = None
    for p in sorted((DATA / "val/gt").glob("Patient_*.png")):
        pat, idx = p.stem.rsplit("_", 1)
        if patient and pat != patient:
            continue
        gt = np.array(Image.open(p)) // scale
        score = min(int((gt == k).sum()) for k in CLASSES)
        if score and (best is None or score > best[0]):
            best = (score, pat, int(idx))
    if best is None:
        raise ValueError(f"no validation slice with all organs found (patient={patient})")
    return best[1], best[2]


def patient_dice(experiment: str, seed: int, patient: str) -> dict:
    with (METRICS / experiment / f"seed{seed}/metrics_3d.csv").open() as f:
        return {r["class_name"]: float(r["dice"]) for r in csv.DictReader(f) if r["patient"] == patient}


def overlay_figure(plt, groups, out, patient, scale):
    """Figure 5: one validation slice; ground truth + the median seed (by 3D Dice fg) of every loss."""
    pat, idx = pick_slice(patient, scale)
    name = f"{pat}_{idx:04d}.png"
    ct = np.array(Image.open(DATA / "val/img" / name), float)
    gt = np.array(Image.open(DATA / "val/gt" / name)) // scale
    ys, xs = np.nonzero(gt)
    half = max(np.ptp(ys), np.ptp(xs)) // 2 + 48  # 48 px margin around the ground-truth organs
    cy, cx = int(ys.mean()), int(xs.mean())
    y0, x0 = max(0, min(cy - half, ct.shape[0] - 2 * half)), max(0, min(cx - half, ct.shape[1] - 2 * half))
    crop = (slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))

    def draw(ax, labels, title, sub=None):
        ax.imshow(ct[crop], cmap="gray", vmin=0, vmax=np.percentile(ct[crop], 99.5))
        rgba = np.zeros(labels.shape + (4,))
        for k in CLASSES:
            rgba[labels == k] = (*[int(COLORS[k][i:i + 2], 16) / 255 for i in (1, 3, 5)], .6)
        ax.imshow(rgba[crop])
        if title != "Ground truth":
            ax.contour((labels != gt)[crop].astype(float), levels=[.5], colors="black", linewidths=1.0)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel(sub or "", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)

    ncols = 3 if len(groups) + 1 <= 6 else 4
    nrows = -(-(len(groups) + 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 5.3 * nrows))
    axes = axes.ravel()
    draw(axes[0], gt, "Ground truth", f"{pat}, slice {idx}")
    picks = {}
    for ax, g in zip(axes[1:], groups):
        ranked = sorted(g.runs, key=lambda r: r["eval"]["val_dice_fg"])
        run = ranked[(len(ranked) - 1) // 2]  # median seed by 3D Dice fg (lower median for even n)
        pred = np.array(Image.open(RUNS / g.experiment / f"seed{run['seed']}/predictions/val" / name)) // scale
        d = patient_dice(g.experiment, run["seed"], pat)
        picks[g.key] = run["seed"]
        draw(ax, pred, f"{g.label} (seed {run['seed']})", "  ".join(f"{NAMES[k][:3]} {d[CLASSES[k]]:.2f}" for k in CLASSES))
    for ax in axes[len(groups) + 1:]:
        ax.axis("off")
    for k in CLASSES:
        axes[0].scatter([], [], marker="s", s=80, color=COLORS[k], label=NAMES[k])
    axes[0].plot([], [], color="black", linewidth=1.2, label="prediction != ground truth")
    legend_below(axes[0], ncol=5)
    decorate(fig, f"Predictions of each loss on {pat}, slice {idx}",
             subtitle="Median seed of every loss, ranked by 3D Dice (foreground mean) over the validation patients.",
             footnote_text="Text under a panel = 3D Dice of that organ for this patient (whole volume, not this slice). "
                           "Black outline = pixels where the prediction differs from the ground truth. Same crop in every panel; "
                           "predictions outside the crop are not shown.")
    fig.subplots_adjust(hspace=.32)  # room for the per-panel Dice caption above the next row's title
    save(plt, fig, out, "loss_comparison_overlay")
    return {"patient": pat, "slice": idx, "seed_shown": picks}


def md_table(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(r) + " |" for r in rows]) + "\n"


def cell(v: np.ndarray, digits: int) -> str:
    m, s, k = mean_std(v)
    if not k:
        return "undefined"
    text = f"{m:.{digits}f}" + (f" ± {s:.{digits}f}" if np.isfinite(s) else "")
    return text + (f" ({len(v) - k} undef.)" if k < len(v) else "")


def write_tables(groups, out, below):
    keys = [k for k, _ in PANELS]
    titles = [t for _, t in PANELS]
    n = len(groups[0].runs)
    dice_rows, dist_rows = [], []
    for g in groups:
        loss = f"{g.runs[0]['cfg']['loss']['name']} {json.dumps(g.runs[0]['cfg']['loss']['kwargs'])}"
        drow, trow = {"loss": g.label, "loss_config": loss, "n_seeds": n}, {"loss": g.label, "n_seeds": n}
        for key in keys:
            drow[f"dice_{key}_mean"], drow[f"dice_{key}_std"], _ = mean_std(g.scores("dice", key))
            for metric in ("hd95", "assd"):
                v = g.scores(metric, key)
                trow[f"{metric}_{key}_mean"], trow[f"{metric}_{key}_std"], k = mean_std(v)
                trow[f"{metric}_{key}_undefined_seeds"] = len(v) - k
        for key in keys[:-1]:
            trow[f"collapsed_{key}"] = len(collapsed(g, key, below))
        dice_rows.append(drow), dist_rows.append(trow)
    write_csv(out / "loss_comparison_dice.csv", dice_rows)
    write_csv(out / "loss_comparison_distance.csv", dist_rows)
    md = ["# Loss comparison, corrected SEGTHOR labels\n", f"{seed_note(groups)}. Mean ± sample std over seeds.\n",
          "## 3D Dice (higher is better)\n",
          md_table(["Loss", "Loss config"] + titles,
                   [[g.label, f"`{r['loss_config']}`"] + [cell(g.scores("dice", k), 3) for k in keys]
                    for g, r in zip(groups, dice_rows)])]
    for metric, name, digits in (("hd95", "HD95 (mm, lower is better)", 1), ("assd", "ASSD (mm, lower is better)", 2)):
        md += [f"## {name}\n", "\"undef.\" = seeds where the organ was never predicted, so the distance is undefined "
               "(these are not counted as 0).\n",
               md_table(["Loss"] + titles, [[g.label] + [cell(g.scores(metric, k), digits) for k in keys] for g in groups])]
    md += [f"## Collapsed seeds (organ 3D Dice < {below:g})\n",
           md_table(["Loss"] + titles[:-1], [[g.label] + [f"{len(collapsed(g, k, below))}/{n}" for k in keys[:-1]]
                                             for g in groups])]
    (out / "loss_comparison_tables.md").write_text("\n".join(md))


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        return f"unknown ({e})"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", help="default: seeds finished by every group")
    p.add_argument("--out", type=Path, default=REPO / "results/loss_comparison")
    p.add_argument("--collapse-below", type=float, default=.1, help="organ Dice below this counts as collapsed")
    p.add_argument("--patient", help="validation patient for the overlay (default: best slice over all patients)")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    plt = pyplot()

    groups = load_groups(a.seeds)
    print(f"{len(groups)} groups, seeds {[r['seed'] for r in groups[0].runs]}: " + ", ".join(g.label for g in groups))
    write_tables(groups, a.out, a.collapse_below)
    overview_figure(plt, groups, a.out, a.collapse_below)
    strip_figure(plt, groups, a.out, "dice", "3D Dice", "", False, a.collapse_below)
    strip_figure(plt, groups, a.out, "hd95", "3D HD95", " mm", True, a.collapse_below)
    curves_figure(plt, groups, a.out)
    heatmap_figure(plt, groups, a.out, a.collapse_below)
    overlay = None
    try:
        overlay = overlay_figure(plt, groups, a.out, a.patient, groups[0].runs[0]["cfg"]["data"]["label_scale"])
    except FileNotFoundError as e:  # scratch runs/ predictions are wiped after 14 days without access
        print(f"ERROR: overlay figure skipped, prediction PNGs not found: {e}")
    manifest = {"git_commit": git_commit(), "args": {k: str(v) for k, v in vars(a).items()}, "overlay": overlay,
                "groups": {g.key: {"label": g.label, "experiment": g.experiment, "seeds": [r["seed"] for r in g.runs],
                                   "loss": g.runs[0]["cfg"]["loss"]} for g in groups}}
    (a.out / "loss_comparison_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {a.out}")
    if overlay is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
