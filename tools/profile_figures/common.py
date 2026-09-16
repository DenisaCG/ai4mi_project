"""
Shared helpers for the SegTHOR dataset-profile figure scripts in this folder.

Every script here reads one or more tables written by tools/dataset_profile.py
and writes one PNG (or a pair of PNGs) into its own numbered subfolder under
figures/profile/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Fixed color per organ label, shared by every figure in this package.
LABEL_COLORS = {1: "#2A9D8F", 2: "#264653", 3: "#E9A23B"}


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Argument parser shared by the profile-figure scripts.

    Args:
        description: Script description shown in --help (usually __doc__).

    Returns:
        Parser with --profile-dir already added (default figures/profile).
    """
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile-dir", type=Path, default=Path("figures/profile"))
    return ap


def out_subdir(profile_dir: Path, name: str) -> Path:
    """Create (if needed) and return a numbered output subfolder under profile_dir.

    Args:
        profile_dir: Root profile directory, e.g. figures/profile.
        name: Subfolder name, e.g. "01-03_scan_geometry".

    Returns:
        The subfolder path.
    """
    out_dir = profile_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def header(fig, title: str, subtitle: str, subtitle_y: float) -> None:
    """Big bold left-aligned figure title with a lighter subtitle line below it.

    Args:
        fig: Figure to decorate.
        title: Title text, drawn at 22pt bold.
        subtitle: One-line subtitle text, drawn at 13.5pt in grey.
        subtitle_y: Figure-fraction y position of the subtitle, since each
            figure's own layout leaves a different amount of headroom.
    """
    fig.suptitle(title, fontsize=22, fontweight="bold", x=0.02, ha="left", y=0.995)
    fig.text(0.02, subtitle_y, subtitle, fontsize=13.5, color="#444444", ha="left")
