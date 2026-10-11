"""
Per-patient tables for the full 40-patient SegTHOR training set, compared with the
20 patients of the midterm. Every table is written per patient (no pooling); the
group comparison is computed from those tables afterwards.

Groups: "original_20" are the patient ids found in --midterm-dir, "added_20" are
the other ids of --data-dir. The midterm volumes are profiled too, only to check
that the first 20 patients did not change between the two folders.

Writes to --out-dir: patient_geometry.csv, patient_intensity.csv, patient_labels.csv,
patient_adjacency.csv, patient_wide.csv, fold_preprocessing.csv, comparison.csv,
midterm_vs_full_first20.csv, inputs.csv, run_log.json and README.md.

Usage:
    python tools/dataset_40_tables.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.append(str(Path(__file__).resolve().parent))
from dataset_profile import LABELS, hist_stats, profile_patient

REPO = Path(__file__).resolve().parents[1]
ORIGINAL, ADDED, ALL = "original_20", "added_20", "all_40"
ORGANS = {1: "esophagus", 2: "heart", 3: "trachea", 4: "aorta"}
REGIONS = {"scan": "all voxels", **{f"label {k}": ORGANS[k] for k in LABELS}}
HU_STATS = ["p0.5", "p99.5", "mean", "median", "std"]
FOLD_ROOTS = [
    "12749947",
    "d27e8eea",
    "6f81632c",
    "e5985fec",
]  # full_cv4_enet_dice_ce_fold0..3
MIDTERM_ROOT = "a2842ac2"  # 20-patient, 15/5 split, full preprocessing
MIDTERM_SLIDE = {"crop": 288, "spacing": (0.98, 0.98, 2.5), "window": (-993, 259)}
CACHE = REPO / "results" / "dataset_40_cache"


def input_files(patient_dir: Path) -> list[Path]:
    return [patient_dir / f"{patient_dir.name}.nii.gz", patient_dir / "GT.nii.gz"]


def stat_key(files: list[Path]) -> list[list[int]]:
    return [[f.stat().st_size, f.stat().st_mtime_ns] for f in files]


def process(patient_dir: Path) -> dict:
    """Reduced profile of one patient; cached by input size and mtime so a restart resumes."""
    key = stat_key(input_files(patient_dir))
    cache = CACHE / f"{patient_dir.parent.parent.name}_{patient_dir.name}.pkl"
    if cache.exists():
        stored = pickle.loads(cache.read_bytes())
        if stored["key"] == key:
            return stored["result"]
    full = profile_patient(patient_dir)
    slices = pd.DataFrame(full["slices"])
    two_plus = (
        slices.assign(two_plus=slices["pieces_sharing_an_edge"] >= 2)
        .groupby("label")["two_plus"]
        .mean()
        .to_dict()
    )
    result = {
        "patient": full["patient"],
        "labels": [
            {
                **r,
                "share_slices_2plus_components": 100 * two_plus.get(r["label"], np.nan),
            }
            for r in full["labels"]
        ],
        "pairs": full["pairs"],
        "hists": {g: full["hists"][g] for g in REGIONS},
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps({"key": key, "result": result}))
    tmp.replace(cache)
    return result


def safe_process(patient_dir: Path) -> tuple[str, dict | None]:
    try:
        return patient_dir.name, process(patient_dir)
    except Exception:  # noqa: BLE001  # log which patient failed, keep going, fail at the end
        print(f"FAILED {patient_dir}:\n{traceback.format_exc()}", flush=True)
        return patient_dir.name, None


def profile_all(data_dir: Path, workers: int) -> dict[str, dict]:
    dirs = sorted(
        p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("Patient_")
    )
    with Pool(workers) as pool:
        done = dict(pool.imap(safe_process, dirs))
    failed = [k for k, v in done.items() if v is None]
    if failed:
        raise SystemExit(f"{data_dir}: profiling failed for {failed}")
    print(f"profiled {len(done)} patients in {data_dir}", flush=True)
    return done


def build_tables(
    profiles: dict[str, dict], original_ids: set[str]
) -> dict[str, pd.DataFrame]:
    geometry, intensity, labels, adjacency = [], [], [], []
    for pid, res in sorted(profiles.items()):
        group = ORIGINAL if pid in original_ids else ADDED
        p = res["patient"]
        geometry.append(
            {
                "patient": pid,
                "group": group,
                "pixel_size_left_right_mm": p["x_spacing_mm"],
                "pixel_size_anterior_posterior_mm": p["y_spacing_mm"],
                "slice_spacing_superior_inferior_mm": p["z_spacing_mm"],
                "matrix_left_right": p["x_voxels"],
                "matrix_anterior_posterior": p["y_voxels"],
                "n_slices": p["z_voxels"],
                "scan_length_mm": p["z_extent_mm"],
                "voxels": p["voxels"],
                "axis_codes": p["image_axis_codes"],
                "label_axis_codes": p["label_axis_codes"],
                "label_values": p["label_values_found"],
            }
        )
        for g, region in REGIONS.items():
            counts, start = res["hists"][g]
            s = hist_stats(counts, start) if len(counts) else None
            row = {"patient": pid, "group": group, "region": region}
            row |= {k: (s[k] if s else np.nan) for k in HU_STATS} | {
                "n_voxels": s["n"] if s else 0
            }
            intensity.append(row)
        total_fg = sum(r["voxels"] for r in res["labels"])
        for r in res["labels"]:
            present = r["voxels"] > 0
            labels.append(
                {
                    "patient": pid,
                    "group": group,
                    "label": r["label"],
                    "organ": ORGANS[r["label"]],
                    "voxels": r["voxels"],
                    "volume_ml": r["volume_ml"],
                    "present": int(present),
                    "share_of_scan_pct": r["pct_of_scan"],
                    "share_of_foreground_pct": 100 * r["voxels"] / total_fg
                    if total_fg
                    else np.nan,
                    "components_face": r["pieces_sharing_a_face"] if present else 0,
                    "components_face_edge_corner": r[
                        "pieces_sharing_face_edge_or_corner"
                    ]
                    if present
                    else 0,
                    "slices_with_organ": r.get("slices_with_label", 0),
                    "share_slices_2plus_components_pct": r[
                        "share_slices_2plus_components"
                    ],
                }
            )
        for r in res["pairs"]:
            adjacency.append(
                {
                    "patient": pid,
                    "group": group,
                    "pair": f"{ORGANS[r['label_a']]}-{ORGANS[r['label_b']]}",
                    "shared_face_area_mm2": r["shared_face_area_mm2"],
                    "touching": int(r["shared_face_area_mm2"] > 0),
                    "closest_voxel_centers_mm": r["closest_voxel_centers_mm"],
                }
            )
    return {
        "patient_geometry": pd.DataFrame(geometry),
        "patient_intensity": pd.DataFrame(intensity),
        "patient_labels": pd.DataFrame(labels),
        "patient_adjacency": pd.DataFrame(adjacency),
    }


def wide_table(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per patient, one column per quantity (what the group comparison summarises)."""
    wide = t["patient_geometry"].drop(
        columns=["axis_codes", "label_axis_codes", "label_values"]
    )
    hu = t["patient_intensity"].drop(columns="group")
    hu = hu.pivot(index="patient", columns="region", values=HU_STATS)
    hu.columns = [
        f"hu_{region.replace(' ', '_')}_{stat}" for stat, region in hu.columns
    ]
    lab = t["patient_labels"].drop(columns=["group", "label"])
    value_cols = [c for c in lab.columns if c not in ("patient", "organ")]
    lab = lab.pivot(index="patient", columns="organ", values=value_cols)
    lab.columns = [f"{organ}_{col}" for col, organ in lab.columns]
    adj = t["patient_adjacency"].drop(columns="group")
    adj = adj.pivot(
        index="patient", columns="pair", values=["shared_face_area_mm2", "touching"]
    )
    adj.columns = [f"adjacency_{pair}_{col}" for col, pair in adj.columns]
    return (
        wide.merge(hu, on="patient").merge(lab, on="patient").merge(adj, on="patient")
    )


def compare_groups(wide: pd.DataFrame) -> pd.DataFrame:
    """min/median/max per group for every quantity, with the original-vs-added difference."""
    rows = []
    quantities = [c for c in wide.columns if c not in ("patient", "group")]
    for q in quantities:
        by = {
            ORIGINAL: wide.loc[wide.group == ORIGINAL, q].dropna(),
            ADDED: wide.loc[wide.group == ADDED, q].dropna(),
            ALL: wide[q].dropna(),
        }
        row = {"quantity": q}
        for name, v in by.items():
            row |= {
                f"{name}_min": v.min(),
                f"{name}_median": v.median(),
                f"{name}_max": v.max(),
            }
        a, b = by[ORIGINAL], by[ADDED]
        row["median_added_minus_original"] = b.median() - a.median()
        row["added_outside_original_range"] = int(((b < a.min()) | (b > a.max())).sum())
        constant = a.nunique() <= 1 and b.nunique() <= 1
        if constant:
            row["mannwhitney_p"] = np.nan
            row["differs"] = (
                bool(a.iloc[0] != b.iloc[0]) if len(a) and len(b) else False
            )
            row["test"] = "constant in both groups"
        else:
            row["mannwhitney_p"] = float(
                mannwhitneyu(a, b, alternative="two-sided").pvalue
            )
            row["differs"] = row["mannwhitney_p"] < 0.05
            row["test"] = "Mann-Whitney U, two-sided, p < 0.05"
        rows.append(row)
    return pd.DataFrame(rows)


def fold_preprocessing(data_dir: Path) -> pd.DataFrame:
    rows = []
    for name, root in [(f"fold{k}", r) for k, r in enumerate(FOLD_ROOTS)] + [
        ("midterm", MIDTERM_ROOT)
    ]:
        d = data_dir / f"sliced_{root}"
        roi = json.loads((d / "roi_crop.json").read_text())
        norm = json.loads((d / "ct_norm_stats.json").read_text())
        required = roi["required_train"]
        worst = max(required, key=required.get)
        rows.append(
            {
                "fold": name,
                "dataset": d.name,
                "n_train_patients": len(norm["train_patients"]),
                "roi_window_px": roi["size"],
                "roi_margin_px": roi["margin"],
                "roi_multiple_px": roi["multiple"],
                "largest_required_px": required[worst],
                "largest_required_patient": worst,
                "target_spacing_left_right_mm": roi["target_spacing"][0],
                "target_spacing_anterior_posterior_mm": roi["target_spacing"][1],
                "target_spacing_superior_inferior_mm": roi["target_spacing"][2],
                "window_lo_hu": norm["lo"],
                "window_hi_hu": norm["hi"],
                "zscore_mean_hu": norm["mean"],
                "zscore_std_hu": norm["std"],
                "n_foreground_voxels": norm["n_foreground_voxels"],
            }
        )
    return pd.DataFrame(rows)


def midterm_check(full: dict, mid: dict) -> pd.DataFrame:
    """Per patient: is the midterm copy identical to the same id in the full set?"""
    rows = []
    for pid in sorted(mid):
        a, b = full[pid], mid[pid]
        geometry = {k: a["patient"][k] for k in a["patient"] if k != "patient"}
        geometry_mid = {k: b["patient"][k] for k in b["patient"] if k != "patient"}
        hist_equal = all(
            np.array_equal(a["hists"][g][0], b["hists"][g][0])
            and a["hists"][g][1] == b["hists"][g][1]
            for g in REGIONS
        )
        rows.append(
            {
                "patient": pid,
                "geometry_equal": geometry == geometry_mid,
                "label_voxels_equal": [r["voxels"] for r in a["labels"]]
                == [r["voxels"] for r in b["labels"]],
                "components_equal": [r["pieces_sharing_a_face"] for r in a["labels"]]
                == [r["pieces_sharing_a_face"] for r in b["labels"]],
                "histograms_equal": hist_equal,
            }
            | {
                f"{ORGANS[ra['label']]}_voxel_diff": ra["voxels"] - rb["voxels"]
                for ra, rb in zip(a["labels"], b["labels"])
            }
        )
    return pd.DataFrame(rows)


def group_stat(df: pd.DataFrame, group: str, col: str, how: str) -> float:
    v = df[col] if group == ALL else df.loc[df.group == group, col]
    return float(getattr(v, how)())


def fmt(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".rstrip("0").rstrip(".") if abs(x) < 1e5 else f"{x:.0f}"


def triple(df: pd.DataFrame, group: str, col: str, digits: int = 2) -> str:
    return " / ".join(
        fmt(group_stat(df, group, col, h), digits) for h in ("min", "median", "max")
    )


def pooled_share(labels: pd.DataFrame, group: str, label: int, column: str) -> float:
    sub = labels if group == ALL else labels[labels.group == group]
    return 100 * sub.loc[sub.label == label, "voxels"].sum() / sub[column].sum()


def write_readme(
    out: Path,
    t: dict,
    wide: pd.DataFrame,
    cmp: pd.DataFrame,
    folds: pd.DataFrame,
    old: dict[str, pd.DataFrame],
    check: pd.DataFrame,
) -> None:
    lines = ["# Full 40-patient SegTHOR training set versus the midterm 20", ""]
    lines += [
        "Source: `data/segthor_train_full/train` (corrected labels). `original_20` are the patient ids in",
        "`data/segthor_part1_corrected/train` (the midterm set), `added_20` are the other 20 ids, `all_40` is both.",
        "Everything is computed per patient first (`patient_*.csv`, `patient_wide.csv`); the tables below summarise those.",
        'Old values in "Slide numbers that change" are recomputed with the same code from the midterm folder itself, not read off the slides.',
        "",
    ]
    same = check[
        ["geometry_equal", "label_voxels_equal", "components_equal", "histograms_equal"]
    ].all()
    lines += [
        "## Are the first 20 patients the same files as at midterm?",
        "",
        f"Compared `{len(check)}` patients between the two folders (geometry, per-organ voxel counts, "
        f"3D component counts, exact 1 HU histograms of the whole scan and each organ): "
        + ", ".join(f"{c} = {int(check[c].sum())}/{len(check)}" for c in same.index)
        + ".",
        "",
        "Organ voxel counts of the same patient id, full set minus midterm folder (patients that differ, total voxels): "
        + "; ".join(
            f"{o}: {int((check[f'{o}_voxel_diff'] != 0).sum())}/{len(check)} patients, "
            f"{int(check[f'{o}_voxel_diff'].sum()):+d} voxels"
            for o in ORGANS.values()
        )
        + ". The CT volumes are compared through the geometry and histogram columns of `midterm_vs_full_first20.csv`.",
        "",
    ]

    lines += ["## Group comparison (min / median / max over patients)", ""]
    lines += [
        "`differs` is True when the two groups are constant and unequal, or when a two-sided Mann-Whitney U test of",
        (
            "original_20 against added_20 gives p < 0.05. This is one test per quantity, "
            f"{len(cmp)} quantities, so about {0.05 * len(cmp):.0f} rows are expected to flag by chance alone."
        ),
        "`outside` is how many of the 20 added patients lie outside the original 20's [min, max].",
        "",
        "| quantity | original_20 | added_20 | all_40 | median added - original | outside | p | differs |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in cmp.iterrows():
        cells = [
            " / ".join(fmt(r[f"{g}_{h}"]) for h in ("min", "median", "max"))
            for g in (ORIGINAL, ADDED, ALL)
        ]
        p = "" if np.isnan(r["mannwhitney_p"]) else f"{r['mannwhitney_p']:.3g}"
        lines.append(
            f"| {r['quantity']} | {cells[0]} | {cells[1]} | {cells[2]} | {fmt(r['median_added_minus_original'])} "
            f"| {r['added_outside_original_range']} | {p} | {r['differs']} |"
        )
    differ = cmp[cmp.differs]["quantity"].tolist()
    same_q = cmp[~cmp.differs]["quantity"].tolist()
    lines += [
        "",
        f"**Differ ({len(differ)}):** "
        + (", ".join(f"`{q}`" for q in differ) or "none"),
        "",
        f"**Do not differ ({len(same_q)}):** "
        + (", ".join(f"`{q}`" for q in same_q) or "none"),
        "",
    ]

    lab, geo = t["patient_labels"], t["patient_geometry"]
    old_lab, old_geo, old_adj = (
        old["patient_labels"],
        old["patient_geometry"],
        old["patient_adjacency"],
    )
    lines += ["## Preprocessing the cross-validation actually used", ""]
    lines += [
        "| | roi window (px) | largest required (px, patient) | target spacing (mm) | window lo..hi (HU) | train patients |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in folds.iterrows():
        label = "midterm dataset" if r["fold"] == "midterm" else r["fold"]
        lines.append(
            f"| {label} | {r['roi_window_px']} | {fmt(r['largest_required_px'])} ({r['largest_required_patient']}) "
            f"| {r['target_spacing_left_right_mm']:.4f} x {r['target_spacing_anterior_posterior_mm']:.4f} x "
            f"{r['target_spacing_superior_inferior_mm']:.1f} | {r['window_lo_hu']:.1f} .. {r['window_hi_hu']:.1f} "
            f"| {r['n_train_patients']} |"
        )
    lines += [
        "",
        (
            f"Midterm slide values: window {MIDTERM_SLIDE['crop']} x {MIDTERM_SLIDE['crop']}, "
            f"{MIDTERM_SLIDE['spacing'][0]} x {MIDTERM_SLIDE['spacing'][1]} x {MIDTERM_SLIDE['spacing'][2]} mm, "
            f"{MIDTERM_SLIDE['window'][0]} to {MIDTERM_SLIDE['window'][1]} HU."
        ),
        "",
    ]

    lines += [
        "## Slide numbers that change",
        "",
        "Old = the 20 patients of the midterm folder, new = all 40 of the full folder. Medians are of per-patient values.",
        "",
    ]
    lines += ["| number | old | new |", "|---|---|---|"]
    mid_fold = folds[folds.fold == "midterm"].iloc[0]
    cv = folds[folds.fold != "midterm"]
    lines.append(
        f"| patients with aorta voxels | {int(old_lab[old_lab.label == 4].present.sum())} of 20 "
        f"| {int(lab[lab.label == 4].present.sum())} of 40 |"
    )
    old_aorta = old_lab[old_lab.label == 4].voxels.sum()
    lines.append(
        f"| aorta voxels, summed over patients | {old_aorta} | {lab[lab.label == 4].voxels.sum()} |"
    )
    lines.append(
        f"| in-plane pixel size, median (mm) | {fmt(group_stat(old_geo, ALL, 'pixel_size_left_right_mm', 'median'), 4)} "
        f"x {fmt(group_stat(old_geo, ALL, 'pixel_size_anterior_posterior_mm', 'median'), 4)} "
        f"| {fmt(group_stat(geo, ALL, 'pixel_size_left_right_mm', 'median'), 4)} "
        f"x {fmt(group_stat(geo, ALL, 'pixel_size_anterior_posterior_mm', 'median'), 4)} |"
    )
    lines.append(
        f"| slice spacing, min / median / max (mm) | {triple(old_geo, ALL, 'slice_spacing_superior_inferior_mm')} "
        f"| {triple(geo, ALL, 'slice_spacing_superior_inferior_mm')} |"
    )
    lines.append(
        f"| number of slices, min / median / max | {triple(old_geo, ALL, 'n_slices')} | {triple(geo, ALL, 'n_slices')} |"
    )
    lines.append(
        f"| scan length, min / median / max (mm) | {triple(old_geo, ALL, 'scan_length_mm', 0)} "
        f"| {triple(geo, ALL, 'scan_length_mm', 0)} |"
    )
    lines.append(
        f"| target spacing (mm) | {mid_fold.target_spacing_left_right_mm:.4f} x "
        f"{mid_fold.target_spacing_anterior_posterior_mm:.4f} x {mid_fold.target_spacing_superior_inferior_mm:.1f} "
        f"| per fold: "
        + "; ".join(
            f"{r.target_spacing_left_right_mm:.4f} x {r.target_spacing_anterior_posterior_mm:.4f} x "
            f"{r.target_spacing_superior_inferior_mm:.1f}"
            for r in cv.itertuples()
        )
        + " |"
    )
    lines.append(
        f"| crop window (px) | {mid_fold.roi_window_px} | per fold: "
        + ", ".join(str(v) for v in cv.roi_window_px)
        + " |"
    )
    lines.append(
        f"| HU window (lo .. hi) | {mid_fold.window_lo_hu:.1f} .. {mid_fold.window_hi_hu:.1f} | per fold: "
        + "; ".join(
            f"{r.window_lo_hu:.1f} .. {r.window_hi_hu:.1f}" for r in cv.itertuples()
        )
        + " |"
    )
    for k, organ in ORGANS.items():
        sub = lab[lab.label == k]
        old, new = old_lab[old_lab.label == k], sub
        lines.append(
            f"| {organ} volume, min / median / max (mL) | {' / '.join(fmt(getattr(old.volume_ml, h)(), 1) for h in ('min', 'median', 'max'))} "
            f"| {' / '.join(fmt(getattr(new.volume_ml, h)(), 1) for h in ('min', 'median', 'max'))} |"
        )
    for k, organ in ORGANS.items():
        lines.append(
            f"| {organ} voxel share of foreground, pooled over patients (%) "
            f"| {pooled_share(old_lab, ALL, k, 'voxels'):.2f} | {pooled_share(lab, ALL, k, 'voxels'):.2f} |"
        )
    for k, organ in ORGANS.items():
        sub = lab[lab.label == k]
        lines.append(
            f"| {organ} voxel share of scan, median (%) | {fmt(old_lab[old_lab.label == k].share_of_scan_pct.median(), 3)} "
            f"| {fmt(sub.share_of_scan_pct.median(), 3)} |"
        )
    for k, organ in ORGANS.items():
        sub = lab[lab.label == k]
        old = old_lab[old_lab.label == k]
        lines.append(
            f"| {organ} 3D components (face connectivity), min / median / max "
            f"| {' / '.join(fmt(getattr(old.components_face, h)()) for h in ('min', 'median', 'max'))} "
            f"| {' / '.join(fmt(getattr(sub.components_face, h)()) for h in ('min', 'median', 'max'))} |"
        )
    for k, organ in ORGANS.items():
        sub = lab[lab.label == k]
        old = old_lab[old_lab.label == k]
        lines.append(
            f"| {organ} share of axial slices with 2+ components, median (%) "
            f"| {fmt(old.share_slices_2plus_components_pct.median(), 1)} | {fmt(sub.share_slices_2plus_components_pct.median(), 1)} |"
        )
    adj = t["patient_adjacency"]
    for pair in sorted(adj.pair.unique()):
        sub = adj[adj.pair == pair]
        lines.append(
            f"| patients where {pair} share a voxel face | {int(old_adj[old_adj.pair == pair].touching.sum())} of 20 "
            f"| {int(sub.touching.sum())} of 40 |"
        )
    lines += [
        "",
        (
            "Components are 6-connected in 3D (`components_face`; 26-connected counts are in `patient_labels.csv`). "
            "The 2+ components share uses 4-connected pieces per axial slice, over the slices that contain the organ."
        ),
        "",
        "## Figures",
        "",
        (
            "Rows are patients (Patient_01 at the top), circles are original_20 and squares are added_20. "
            "Axes are named left-right, anterior-posterior, superior-inferior."
        ),
        "",
        "- `geometry_per_patient.png`: pixel size (left-right, anterior-posterior), slice spacing, matrix size, number of slices and scan length (superior-inferior) per patient.",
        "- `hu_per_patient.png`: HU median (marker) with p0.5 to p99.5 (line) and mean (tick) per patient, for all voxels and inside each organ.",
        "- `organ_volume_per_patient.png`: organ volume in mL per patient, one panel per organ.",
        "- `esophagus_components_per_patient.png`: number of 3D connected components of the esophagus, and share of its axial slices with 2+ components, per patient.",
        "",
        "## Files",
        "",
        (
            "`patient_geometry.csv`, `patient_intensity.csv` (long: patient x region), `patient_labels.csv` (long: patient x organ), "
            "`patient_adjacency.csv` (long: patient x organ pair), `patient_wide.csv`, `comparison.csv`, `fold_preprocessing.csv`, "
            "`midterm_vs_full_first20.csv`, `inputs.csv` (size and mtime of every input), `run_log.json`. "
            "Regenerate with `jobsAndOutputs/dataset_40/jobs/dataset_40.job`; check with `tools/validate_dataset_40.py`."
        ),
        "",
    ]
    (out / "README.md").write_text("\n".join(lines))


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data-dir", type=Path, default=REPO / "data/segthor_train_full/train"
    )
    ap.add_argument(
        "--midterm-dir", type=Path, default=REPO / "data/segthor_part1_corrected/train"
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=REPO / "data",
        help="folder holding the sliced_<hash> fold datasets",
    )
    ap.add_argument("--out-dir", type=Path, default=REPO / "figures/dataset_40")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    inputs = [
        f
        for d in (args.data_dir, args.midterm_dir)
        for p in sorted(d.glob("Patient_*"))
        for f in input_files(p)
    ]
    inputs += [
        args.data_root / f"sliced_{r}" / n
        for r in FOLD_ROOTS + [MIDTERM_ROOT]
        for n in ("roi_crop.json", "ct_norm_stats.json")
    ]
    pd.DataFrame(
        {
            "path": [str(f) for f in inputs],
            "size_bytes": [f.stat().st_size for f in inputs],
            "mtime_ns": [f.stat().st_mtime_ns for f in inputs],
        }
    ).to_csv(args.out_dir / "inputs.csv", index=False)

    full = profile_all(args.data_dir, args.workers)
    mid = profile_all(args.midterm_dir, args.workers)
    original_ids = set(mid)
    assert original_ids <= set(full), (
        f"midterm ids missing from the full set: {sorted(original_ids - set(full))}"
    )

    tables = build_tables(full, original_ids)
    wide = wide_table(tables)
    cmp = compare_groups(wide)
    folds = fold_preprocessing(args.data_root)
    check = midterm_check(full, mid)
    for name, df in tables.items():
        df.to_csv(args.out_dir / f"{name}.csv", index=False)
    wide.to_csv(args.out_dir / "patient_wide.csv", index=False)
    cmp.to_csv(args.out_dir / "comparison.csv", index=False)
    folds.to_csv(args.out_dir / "fold_preprocessing.csv", index=False)
    check.to_csv(args.out_dir / "midterm_vs_full_first20.csv", index=False)
    write_readme(
        args.out_dir, tables, wide, cmp, folds, build_tables(mid, original_ids), check
    )
    (args.out_dir / "run_log.json").write_text(
        json.dumps(
            {
                "git_commit": git_commit(),
                "started": started,
                "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "args": {k: str(v) for k, v in vars(args).items()},
                "n_patients": len(full),
                "n_original": len(original_ids),
                "seed": "none (the nnU-Net sample inside dataset_profile uses its fixed seed 1234)",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote tables and README to {args.out_dir}")


if __name__ == "__main__":
    main()
