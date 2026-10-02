#!/usr/bin/env python3
"""Regenerate the dataset figures used in the mid-project slides from raw NIfTI data.

Every label figure comes from --data-dir (the corrected four-label release);
--before-data-dir (the original release) is only used by the before/after figures.
Steps, in order:
  1. tools/dataset_profile.py          -> <out-dir>/profile/*.csv + histograms
  2. scan_geometry, organ_position_3d, label_example
  3. with --before-data-dir: profile of the original data, then label_hu_distribution,
     connected_components, label_makeup_rings and label_correction_example,
     written to --comparison-dir

Usage:
    python tools/run_all_figures.py --data-dir data/segthor_part1_corrected/train --out-dir figures \\
        --before-data-dir data/segthor_part1/train --names "3 labels (aorta merged)" "4 labels (aorta separate)"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIGURES = REPO / "dataset_analysis/profile_figures"

Step = tuple[str, list[str]]  # (label, argv)


def figure(name: str, *args: object) -> list[str]:
    """Command line running one figure script.

    Args:
        name: Script name inside dataset_analysis/profile_figures, without .py.
        *args: Arguments passed to the script.

    Returns:
        The argv list.
    """
    return [sys.executable, "-B", str(FIGURES / f"{name}.py"), *map(str, args)]


def profile(data_dir: Path, out_dir: Path, max_patients: int | None) -> list[str]:
    """Command line writing the profile tables of one dataset.

    Args:
        data_dir: Raw NIfTI dataset directory.
        out_dir: Directory the tables go to.
        max_patients: Deterministic subset size, or None for every patient.

    Returns:
        The argv list.
    """
    cmd = [
        sys.executable,
        "-B",
        "tools/dataset_profile.py",
        "--data-dir",
        str(data_dir),
        "--out-dir",
        str(out_dir),
    ]
    return cmd + (["--max-patients", str(max_patients)] if max_patients else [])


def build_plan(args: argparse.Namespace) -> list[Step]:
    """Every step to run, in order.

    Args:
        args: Parsed command line.

    Returns:
        Ordered (label, argv) steps.
    """
    profile_dir = args.out_dir / "profile"
    steps: list[Step] = [
        ("dataset_profile", profile(args.data_dir, profile_dir, args.max_patients)),
        (
            "scan_geometry",
            figure(
                "scan_geometry",
                "--profile-dir",
                profile_dir,
                "--data-dir",
                args.data_dir,
            ),
        ),
        (
            "organ_position_3d",
            figure("organ_position_3d", "--profile-dir", profile_dir),
        ),
        (
            "label_example",
            figure(
                "label_example",
                "--data-dir",
                args.data_dir,
                "--out-dir",
                args.comparison_dir,
            ),
        ),
    ]
    if args.before_data_dir:
        before_profile = args.out_dir / "before" / "profile"
        names = ["--names", *args.names]
        steps += [
            (
                "before/dataset_profile",
                profile(args.before_data_dir, before_profile, args.max_patients),
            ),
            (
                "label_hu_distribution",
                figure(
                    "label_hu_distribution",
                    "--profile-dir",
                    before_profile,
                    profile_dir,
                    *names,
                    "--out-dir",
                    args.comparison_dir,
                ),
            ),
            (
                "connected_components",
                figure(
                    "connected_components",
                    "--data-dir",
                    args.data_dir,
                    "--before-data-dir",
                    args.before_data_dir,
                    *names,
                    "--out-dir",
                    args.comparison_dir,
                ),
            ),
            (
                "label_makeup_rings",
                figure(
                    "label_makeup_rings",
                    "--profile-dir",
                    before_profile,
                    profile_dir,
                    *names,
                    "--out-dir",
                    args.comparison_dir,
                ),
            ),
            (
                "label_correction_example",
                figure(
                    "label_correction_example",
                    "--data-dir",
                    args.data_dir,
                    "--before-data-dir",
                    args.before_data_dir,
                    "--out-dir",
                    args.comparison_dir,
                ),
            ),
        ]
    return steps


def run(steps: list[Step]) -> None:
    """Run the steps in order, printing each one's wall time; a nonzero exit aborts the driver.

    Args:
        steps: (label, argv) pairs.
    """
    for label, cmd in steps:
        print(f"\n=== {label} ===")
        print(" ".join(cmd))
        start = time.monotonic()
        subprocess.run(cmd, cwd=REPO, check=True)
        print(f"--- {label} done in {time.monotonic() - start:.1f}s ---")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Arguments, default sys.argv[1:].

    Returns:
        The parsed namespace, with the comparison folder default filled in.
    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/segthor_part1_corrected/train"),
        help="Raw NIfTI dataset directory, one Patient_XX/ folder per patient.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures"),
        help="Root for the profile tables and figures.",
    )
    ap.add_argument(
        "--before-data-dir",
        type=Path,
        default=None,
        help="Original dataset the before/after figures compare --data-dir against.",
    )
    ap.add_argument(
        "--comparison-dir",
        type=Path,
        default=None,
        help="Where the label and before/after figures go; default <out-dir>/comparison.",
    )
    ap.add_argument(
        "--names",
        nargs=2,
        default=["before", "after"],
        metavar=("BEFORE", "AFTER"),
        help="Names of --before-data-dir and --data-dir in the comparison figures.",
    )
    ap.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Deterministic subset, for a fast smoke run over every step.",
    )
    args = ap.parse_args(argv)
    if args.comparison_dir is None:
        args.comparison_dir = args.out_dir / "comparison"
    return args


def main() -> None:
    """Run the whole pipeline."""
    args = parse_args()
    steps = build_plan(args)
    run(steps)
    print(f"\nAll {len(steps)} steps complete. Figures: {args.out_dir}")


if __name__ == "__main__":
    main()
