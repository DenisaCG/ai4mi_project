"""score_volumes must rebuild the metrics_3d.csv of a finished run exactly from its saved 2D predictions."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import REPO, read_yaml
from src.evaluate import score_volumes
from src.metrics_3d import METRICS

RUN = REPO / "runs" / "full_cv4_resenc_ds_unet_dice_ce_fold0" / "seed0"


@unittest.skipUnless((RUN / "eval" / ".done").exists(), f"{RUN} is not available")
class ScoreVolumesReproTest(unittest.TestCase):
    def test_rows_match_the_run_metrics(self):
        cfg = read_yaml((RUN / "config.yaml").read_text())
        scale = cfg["data"]["label_scale"]
        preds = {
            p.stem: (np.array(Image.open(p)) // scale).astype(np.uint8)
            for p in sorted((RUN / "predictions" / "val").glob("*.png"))
        }
        with tempfile.TemporaryDirectory() as tmp:
            rows, _ = score_volumes(
                preds,
                "val",
                cfg["data"]["source_pattern"],
                Path(tmp),
                cfg,
                {},
                _Log(),
            )
        with (RUN / "eval" / "metrics_3d.csv").open(newline="") as f:
            expected = list(csv.DictReader(f))
        self.assertEqual(len(rows), len(expected))
        for mine, theirs in zip(rows, expected):
            self.assertEqual(
                (mine["patient"], str(mine["class_idx"])),
                (theirs["patient"], theirs["class_idx"]),
            )
            for m in METRICS:
                a, b = float(theirs[m]), mine[m]
                self.assertTrue(
                    np.isnan(a) == np.isnan(b) and (np.isnan(a) or a == b),
                    (theirs["patient"], m, a, b),
                )
            self.assertEqual(int(theirs["pred_voxels"]), mine["pred_voxels"])


class _Log:
    def info(self, *args):
        pass

    def warning(self, *args):
        pass


if __name__ == "__main__":
    unittest.main()
