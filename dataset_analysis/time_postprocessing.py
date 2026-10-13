"""Time of each post-processing step on one validation volume, written to postprocessing_cost.csv.

    python dataset_analysis/time_postprocessing.py [--out figures]

The steps run on the 512 x 512 x 171 heart labels of the three seeds of Patient 30 (deck/snellius_inventory/p30_seeds),
on whatever CPU this runs on. The softmax average has no saved probabilities to read, so it averages random float16
maps of the shape the network writes, (171, 5, 256, 256) per seed, and takes the argmax. Each time is the median of
five runs. The seed ensembles also need three trained models; their training and network time is in
figures/all_results_table.csv and the timing folder of the results bundle.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from utils import REPO, write_csv

sys.path.insert(0, str(REPO / "tools"))
from postprocess_filters import heart_hull, majority_vote, remove_small_components  # noqa: E402

SEEDS = REPO / "deck/snellius_inventory/p30_seeds"
HEART, TRACHEA = 2, 3
REPEATS = 5


def median_seconds(step) -> float:
    """Median wall-clock seconds of REPEATS calls of `step`."""
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        step()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def largest_component(volume: np.ndarray, organs: list[int]) -> np.ndarray:
    """Keeps the largest 6-connected piece of each organ in `organs`."""
    for organ in organs:
        volume = remove_small_components(volume, organ, rel=1.0)
    return volume


def softmax_average(maps: list[np.ndarray]) -> np.ndarray:
    """Labels of the mean of the seeds' (Z, K, H, W) probability maps."""
    return np.mean([m.astype(np.float32) for m in maps], axis=0).argmax(axis=1)


def main() -> None:
    """Times the steps and writes the table."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=REPO / "figures")
    args = parser.parse_args()

    seeds = [
        np.asarray(nib.load(SEEDS / f"seed{s}.nii.gz").dataobj).astype(np.uint8)
        for s in range(3)
    ]
    rng = np.random.default_rng(0)
    maps = [
        rng.random((171, 5, 256, 256), dtype=np.float32).astype(np.float16)
        for _ in range(3)
    ]
    steps = {
        "largest_component_all_organs": lambda: largest_component(
            seeds[0], [1, 2, 3, 4]
        ),
        "largest_component_heart": lambda: largest_component(seeds[0], [HEART]),
        "largest_component_heart_trachea": lambda: largest_component(
            seeds[0], [HEART, TRACHEA]
        ),
        "heart_hull_after_largest_component": lambda: heart_hull(
            largest_component(seeds[0], [HEART]), HEART
        ),
        "majority_vote_3_seeds": lambda: majority_vote(seeds),
        "softmax_average_3_seeds": lambda: softmax_average(maps),
    }
    rows = [
        {"step": name, "seconds": round(median_seconds(step), 2)}
        for name, step in steps.items()
    ]
    write_csv(args.out / "postprocessing_cost.csv", rows)
    for row in rows:
        print(f"{row['step']:34s}{row['seconds']:6.2f} s")


if __name__ == "__main__":
    main()
