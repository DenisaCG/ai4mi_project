import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.config import load_config
from src.data import build_dataset, keep_patients

REGEX = r"(Patient_\d+)_\d{4}"


def slices(n_patients: int, per_patient: int = 3) -> list[tuple[Path, Path]]:
    return [
        (Path(f"Patient_{p:02d}_{s:04d}.png"), Path(f"Patient_{p:02d}_{s:04d}.png"))
        for p in range(1, n_patients + 1)
        for s in range(per_patient)
    ]


def patients_of(files) -> set[str]:
    return {img.stem[:10] for img, _ in files}


class KeepPatientsTests(unittest.TestCase):
    def test_keeps_all_slices_of_n_patients(self):
        kept = keep_patients(slices(30), REGEX, 15)
        self.assertEqual(len(patients_of(kept)), 15)
        self.assertEqual(len(kept), 15 * 3)

    def test_subsets_are_nested(self):
        files = slices(30)
        p15, p22, p30 = (
            patients_of(keep_patients(files, REGEX, n)) for n in (15, 22, 30)
        )
        self.assertLess(p15, p22)
        self.assertLess(p22, p30)
        self.assertEqual(p30, patients_of(files))

    def test_choice_is_fixed_and_independent_of_the_input_order(self):
        files = slices(30)
        self.assertEqual(
            keep_patients(files, REGEX, 15), keep_patients(files, REGEX, 15)
        )
        self.assertEqual(
            patients_of(keep_patients(files[::-1], REGEX, 15)),
            patients_of(keep_patients(files, REGEX, 15)),
        )

    def test_more_patients_than_available_raises(self):
        with self.assertRaisesRegex(ValueError, "only 30 patients"):
            keep_patients(slices(30), REGEX, 31)


class TrainPatientsConfigTests(unittest.TestCase):
    def test_default_is_all_patients(self):
        self.assertEqual(
            load_config(Path("configs/segthor_enet_ce.yaml"))["train"][
                "train_patients"
            ],
            0,
        )

    def test_only_the_train_split_is_subsampled(self):
        with tempfile.TemporaryDirectory() as root:
            for split in ("train", "val"):
                for kind in ("img", "gt"):
                    (Path(root) / split / kind).mkdir(parents=True)
                    for p in range(1, 7):
                        Image.new("L", (8, 8)).save(
                            Path(root) / split / kind / f"Patient_{p:02d}_0000.png"
                        )
            cfg = {
                "seed": 0,
                "train": {"debug_samples": 0, "train_patients": 4},
                "data": {
                    "root": root,
                    "preprocess": None,
                    "num_classes": 5,
                    "label_scale": 63,
                    "augment": [],
                    "patient_regex": REGEX,
                },
            }
            self.assertEqual(len(build_dataset(cfg, "train")), 4)
            self.assertEqual(len(build_dataset(cfg, "val")), 6)


if __name__ == "__main__":
    unittest.main()
