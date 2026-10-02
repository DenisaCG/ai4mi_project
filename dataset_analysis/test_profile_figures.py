"""Tests for the dataset profile tables and the slide figures built from them."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from dataset_profile import (
    hist_stats,
    identical_neighbor_slices,
    label_row,
    occupied_box,
    pair_row,
    slice_rows,
)
from profile_figures.connected_components import (
    column_slots,
    count_pieces,
    draw_split_arrow,
    piece_counts,
    piece_shares,
    unchanged,
)
from profile_figures.label_correction_example import example_slice, ranked_pieces
from profile_figures.label_correction_example import main as draw_correction_example
from profile_figures.label_example import main as draw_label_example
from profile_figures.label_example import organ_slice
from profile_figures.label_hu_distribution import (
    bar_style,
    bin_values,
    binned_counts,
    draw_order,
    foreground_stats,
    pool_histograms,
    share_outside,
    tick_positions,
    tissue_high,
    warp,
)
from profile_figures.label_makeup_rings import (
    background_share,
    organ_voxels,
    wedge_angles,
)
from profile_figures.label_makeup_rings import main as draw_label_makeup_rings
from profile_figures.organ_position_3d import (
    axis_angle,
    box_edges,
    label_boxes,
    organ_centers,
)
from profile_figures.organ_position_3d import main as draw_organ_position_3d
from profile_figures.scan_geometry import dot_stacks, fov_groups
from run_all_figures import build_plan, parse_args


class DatasetProfileTests(unittest.TestCase):
    def test_hist_stats_match_numpy(self):
        vals = np.random.RandomState(0).randint(-1000, 3000, size=5001)
        counts = np.bincount(vals - vals.min())
        stats = hist_stats(counts, int(vals.min()))
        for q, key in ((0.5, "p0.5"), (50, "median"), (99.5, "p99.5")):
            self.assertAlmostEqual(stats[key], np.percentile(vals, q))
        self.assertAlmostEqual(stats["mean"], vals.mean())
        self.assertAlmostEqual(stats["std"], vals.std())
        self.assertEqual(
            (stats["min"], stats["max"], stats["n"]),
            (vals.min(), vals.max(), vals.size),
        )
        self.assertIsNone(hist_stats(np.zeros(3), 0))

    def test_label_row_extent_and_pieces(self):
        mask = np.zeros((10, 10, 6), dtype=bool)
        mask[2:4, 2:4, 1:3] = True
        mask[4, 4, 3] = True
        mask[2:4, 2:4, 5] = True
        row = label_row(mask, np.array([1.0, 2.0, 2.5]))
        self.assertEqual(row["voxels"], 13)
        self.assertEqual(row["z_size_mm"], 12.5)
        self.assertEqual(row["y_size_mm"], 6.0)
        self.assertEqual(row["empty_slices_inside_range"], 1)
        self.assertEqual(row["pieces_sharing_a_face"], 3)
        self.assertEqual(row["pieces_sharing_face_edge_or_corner"], 2)
        self.assertEqual(label_row(np.zeros_like(mask), np.ones(3))["voxels"], 0)

    def test_slice_rows_sizes(self):
        mask = np.zeros((10, 10, 3), dtype=bool)
        mask[1:4, 2:7, 1] = True
        mask[8, 8, 1] = True
        (row,) = slice_rows(mask, np.array([0.5, 2.0, 3.0]))
        self.assertEqual((row["slice"], row["z_mm"]), (1, 3.0))
        self.assertEqual(row["area_mm2"], 16)
        self.assertEqual((row["x_size_mm"], row["y_size_mm"]), (4.0, 14.0))
        self.assertEqual(row["pieces_sharing_an_edge"], 2)

    def test_occupied_box_and_identical_slices(self):
        ct = np.full((8, 6, 5), -1000)
        ct[2:5, 1:3, 1:4] = 40
        ct[:, :, 2] = ct[:, :, 1]
        box = occupied_box(ct, np.array([1.0, 2.0, 2.5]))
        self.assertEqual(
            (
                box["occupied_x_size_mm"],
                box["occupied_y_size_mm"],
                box["occupied_z_size_mm"],
            ),
            (3, 4, 7.5),
        )
        self.assertEqual(
            (box["outside_box_x_low_voxels"], box["outside_box_x_high_voxels"]), (2, 3)
        )
        self.assertAlmostEqual(box["occupied_pct_of_image"], 100 * 18 / 240)
        self.assertEqual(identical_neighbor_slices(ct), 2)

    def test_pair_row_touching_and_apart(self):
        a = np.zeros((6, 6, 6), dtype=bool)
        b = np.zeros_like(a)
        a[1, 1:3, 1:3] = True
        b[2, 1:3, 1:3] = True
        zooms = np.array([2.0, 1.0, 1.0])
        touching = pair_row(a, b, zooms)
        self.assertEqual(touching["closest_voxel_centers_mm"], 2.0)
        self.assertEqual(touching["shared_face_area_mm2"], 4.0)
        b[:] = False
        b[5, 1, 1] = True
        apart = pair_row(a, b, zooms)
        self.assertEqual(
            (apart["closest_voxel_centers_mm"], apart["shared_face_area_mm2"]),
            (8.0, 0.0),
        )
        self.assertTrue(
            np.isnan(pair_row(a, np.zeros_like(a), zooms)["closest_voxel_centers_mm"])
        )

    def test_dot_stacks_one_dot_per_value(self):
        x, y = dot_stacks(pd.Series([0.98, 0.976, 1.37, 2.0]), 0.01)
        self.assertEqual(len(x), 4)
        self.assertEqual(
            sorted(zip(np.round(x, 2), y)),
            [(0.98, 0.5), (0.98, 1.5), (1.37, 0.5), (2.0, 0.5)],
        )

    def test_fov_groups_counts_each_size_largest_first(self):
        scans = pd.DataFrame(
            {
                "patient": list("abcd"),
                "x_extent_mm": [499.9998, 500.0, 459.0, 700.0002],
                "x_spacing_mm": [0.9765625, 0.9765625, 0.896484, 1.367188],
            }
        )
        g = fov_groups(scans)
        self.assertEqual(g.index.tolist(), [700, 500, 459])
        self.assertEqual(g.n.tolist(), [1, 2, 1])
        self.assertAlmostEqual(g.loc[500, "pixel_mm"], 0.9765625)

    def test_share_outside_windows(self):
        counts = np.array([1, 1, 1, 1])
        self.assertEqual(share_outside(counts, 0, [(0, 2)]), 50)
        self.assertEqual(share_outside(counts, 0, [(0, 2), (2, 4)]), 0)
        self.assertEqual(share_outside(counts, -4, [(0, 2)]), 100)

    def test_tissue_window_grows_for_bright_labels(self):
        self.assertEqual(tissue_high(120), 300)
        self.assertEqual(tissue_high(410), 500)

    def test_bin_values_sum_all_patients_in_thousands(self):
        counts = np.zeros(2000)
        counts[50:60] = 100  # 1000 voxels at -1050 + 50 .. -1050 + 59
        data = {"pooled": {2: (counts, -1050)}, "patients": ["a", "b"]}
        x, y = bin_values(data, 2, 300)
        self.assertEqual(len(x), (300 + 1050) // 10)
        self.assertEqual(x[0], -1045)
        self.assertEqual(
            y.sum(), 1.0
        )  # 1000 voxels, not divided by the 2 patients, in thousands
        self.assertEqual(np.flatnonzero(y).tolist(), [5])

    def test_warp_squeezes_only_the_stretch_between_air_and_tissue(self):
        np.testing.assert_allclose(
            warp([-1000, -900, -500, -200, 0, 300]),
            [-1000, -900, -1000 * 0 - 2500 / 3, -2350 / 3, -1750 / 3, -850 / 3],
        )
        self.assertTrue(np.all(np.diff(warp(np.arange(-1050, 350, 10))) > 0))

    def test_draw_order_puts_the_smaller_bar_of_each_bin_in_front(self):
        heights = np.array([[10, 3, 0], [12, 5, 0], [1, 9, 0]])
        rank = draw_order(heights)
        np.testing.assert_array_equal(
            rank[:, 0], [1, 0, 2]
        )  # 12 at the back, 1 in front
        np.testing.assert_array_equal(rank[:, 1], [2, 1, 0])
        np.testing.assert_array_equal(
            np.sort(rank, axis=0), [[0] * 3, [1] * 3, [2] * 3]
        )

    def test_bar_style_fades_the_heart_and_strengthens_the_other_labels(self):
        heart, aorta = bar_style(2, "#B5533C"), bar_style(4, "#5E9142")
        self.assertGreater(
            min(heart["facecolor"]), min(bar_style(4, "#B5533C")["facecolor"])
        )  # lighter than a strong label
        self.assertEqual(
            aorta["edgecolor"], "#5E9142"
        )  # a strong label keeps its full color outline
        self.assertGreater(aorta["linewidth"], heart["linewidth"])

    def test_tick_positions_are_denser_outside_the_squeezed_stretch(self):
        major, minor = tick_positions(300)
        self.assertEqual(major, [-1000, -900, -600, -200, -100, 0, 100, 200, 300])
        self.assertFalse(set(major) & set(minor))
        self.assertIn(50, minor)
        self.assertIn(-500, minor)  # every 100 HU inside the squeezed stretch
        self.assertNotIn(-550, minor)

    def test_foreground_stats_weight_every_patient_equally(self):
        small = (np.array([1]), 0)  # one labeled voxel at HU 0
        large = (np.array([3]), 10)  # three labeled voxels at HU 10
        stats = foreground_stats([small, large])
        self.assertAlmostEqual(stats["mean"], 5.0)  # plain pooling would give 7.5
        self.assertEqual(stats["p0.5"], 0)
        self.assertEqual(stats["p99.5"], 10)

    def test_count_pieces_reads_every_patient_of_a_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, two_pieces in (("Patient_01", True), ("Patient_02", False)):
                gt = np.zeros((6, 6, 3), dtype=np.uint8)
                gt[0:2, 0:2, :] = 1
                gt[4:6, 4:6, :] = 1 if two_pieces else 0
                (Path(tmp) / name).mkdir()
                nib.save(
                    nib.Nifti1Image(gt, np.eye(4)), str(Path(tmp) / name / "GT.nii.gz")
                )
            rows = count_pieces(Path(tmp))
        whole = rows[(rows.label == 1) & (rows.rule == "3D · 6")]
        self.assertEqual(
            sorted(whole.pieces), [1, 2]
        )  # one patient has two separate blocks
        self.assertEqual(set(rows.label), {1})  # labels without voxels leave no rows

    def test_binned_counts_crops_and_sums(self):
        x, c = binned_counts(np.array([1, 2, 3, 4, 5]), start=-2, lo=0, hi=4, width=2)
        self.assertEqual(x.tolist(), [1.0, 3.0])
        self.assertEqual(c.tolist(), [7.0, 5.0])
        self.assertEqual(
            binned_counts(np.array([9]), start=50, lo=0, hi=4, width=2)[1].tolist(),
            [0.0, 0.0],
        )

    def test_label_boxes_outer_faces_and_reference_shift(self):
        rows = []
        for patient, (lo, hi, center) in {
            "P2": (4.0, 10.0, 7.0),
            "P1": (2.0, 6.0, 4.0),
        }.items():
            for label, shift in ((1, 0.0), (2, 1.0)):
                row = {"patient": patient, "label": label}
                for a in "xyz":
                    row |= {
                        f"{a}_min_mm": lo + shift,
                        f"{a}_max_mm": hi + shift,
                        f"{a}_center_mm": center + shift,
                    }
                    row[f"{a}_size_mm"] = hi - lo + 2.0
                rows.append(row)
        labels = pd.DataFrame(rows)
        start, size = label_boxes(labels, 1)
        self.assertEqual(start[:, 0].tolist(), [1.0, 3.0])
        self.assertEqual(size[:, 0].tolist(), [6.0, 8.0])
        shifted, _ = label_boxes(labels, 1, reference=2)
        self.assertEqual(shifted[:, 2].tolist(), [-4.0, -5.0])

    def test_box_edges_twelve_axis_aligned(self):
        edges = box_edges(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
        self.assertEqual(len(edges), 12)
        lengths = sorted(float(np.abs(b - a).sum()) for a, b in edges)
        self.assertEqual(lengths, [4.0] * 4 + [5.0] * 4 + [6.0] * 4)
        self.assertTrue(all(np.count_nonzero(b != a) == 1 for a, b in edges))


class HistogramTests(unittest.TestCase):
    def test_pool_histograms_aligns_starts(self):
        counts, start = pool_histograms(
            [(np.array([1, 2]), -3), (np.array([5]), -2), (np.array([7]), 1)]
        )
        self.assertEqual(start, -3)
        self.assertEqual(counts.tolist(), [1, 7, 0, 0, 7])


class ConnectedComponentTests(unittest.TestCase):
    def test_piece_counts_differ_by_connectivity(self):
        mask = np.zeros((6, 6, 4), dtype=bool)
        mask[1, 1, 0] = mask[2, 2, 0] = True  # corner-touching pixels in one slice
        mask[4, 4, 1] = True  # touches mask[3, 3, 2] only through a 3D corner
        mask[3, 3, 2] = True
        counts = piece_counts(mask)
        self.assertEqual(counts["2D · 4"].tolist(), [2, 1, 1])
        self.assertEqual(counts["2D · 8"].tolist(), [1, 1, 1])
        self.assertEqual(counts["3D · 6"].tolist(), [4])
        self.assertEqual(counts["3D · 18"].tolist(), [3])
        self.assertEqual(counts["3D · 26"].tolist(), [2])
        self.assertEqual(piece_counts(np.zeros_like(mask))["3D · 6"].size, 0)

    def test_piece_shares_caps_and_normalizes(self):
        counts = pd.DataFrame(
            {"label": [1, 1, 1, 1], "rule": ["r"] * 4, "pieces": [1, 1, 2, 5]}
        )
        row = piece_shares(counts).loc[(1, "r")]
        self.assertEqual(row.tolist(), [50.0, 25.0, 25.0])

    def test_unchanged_labels_have_equal_share_tables_in_both_datasets(self):
        rows = {"label": [1, 1, 2, 2], "rule": ["r"] * 4, "pieces": [1, 2, 1, 1]}
        before = piece_shares(pd.DataFrame(rows))
        after = piece_shares(pd.DataFrame({**rows, "pieces": [1, 1, 1, 1]}))
        self.assertTrue(unchanged(before, after, 2))
        self.assertFalse(unchanged(before, after, 1))
        self.assertFalse(unchanged(before, after, 4))

    def test_column_slots_keep_shared_labels_in_place_and_fill_free_columns(self):
        self.assertEqual(column_slots([1, 2, 3], [1, 4]), {1: 0, 4: 1})
        self.assertEqual(column_slots([1, 2, 3], [2, 4]), {2: 1, 4: 0})
        self.assertEqual(
            column_slots([1, 2, 3], [1, 2, 3, 4]), {1: 0, 2: 1, 3: 2, 4: 3}
        )

    def test_split_arrow_runs_from_the_source_edge_into_the_target(self):
        fig, (left, right) = plt.subplots(1, 2)
        fig.subplots_adjust(wspace=0.5)
        draw_split_arrow(
            fig, left, right, "green", "aorta was part of the esophagus label"
        )
        line = next(a for a in fig.artists if hasattr(a, "get_xdata"))
        x = line.get_xdata()
        self.assertEqual(x[0], left.get_position().x1)
        self.assertTrue(left.get_position().x1 < x[1] < right.get_position().x0)
        self.assertEqual(len(fig.texts), 1)
        plt.close(fig)


class LabelCorrectionTests(unittest.TestCase):
    def test_ranked_pieces_number_the_largest_first(self):
        mask = np.zeros((6, 6, 3), dtype=bool)
        mask[4, 4, 0] = True  # a single voxel far from the block
        mask[0:2, 0:2, :] = True
        pieces, n = ranked_pieces(mask)
        self.assertEqual(n, 2)
        self.assertTrue((pieces[0:2, 0:2, :] == 1).all())
        self.assertEqual(pieces[4, 4, 0], 2)

    def test_correction_example_writes_the_pair_and_each_half(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            before, after = (
                np.zeros((160, 160, 8), dtype=np.uint8),
                np.zeros((160, 160, 8), dtype=np.uint8),
            )
            before[70:90, 70:90, 1:7] = 1  # aorta merged into the esophagus label ...
            before[70:76, 50:56, 2:7] = 1  # ... and the esophagus proper, apart from it
            after[70:90, 70:90, 1:7] = 4
            after[70:76, 50:56, 2:7] = 1
            ct = (
                np.random.default_rng(0)
                .integers(-100, 200, before.shape)
                .astype(np.int16)
            )
            for root, gt in (("old", before), ("new", after)):
                (tmp / root / "Patient_01").mkdir(parents=True)
                nib.save(
                    nib.Nifti1Image(gt, np.eye(4)),
                    str(tmp / root / "Patient_01" / "GT.nii.gz"),
                )
                nib.save(
                    nib.Nifti1Image(ct, np.eye(4)),
                    str(tmp / root / "Patient_01" / "Patient_01.nii.gz"),
                )
            argv = [
                "x",
                "--data-dir",
                str(tmp / "new"),
                "--before-data-dir",
                str(tmp / "old"),
                "--patient",
                "Patient_01",
                "--out-dir",
                str(tmp / "out"),
            ]
            with mock.patch.object(sys, "argv", argv):
                draw_correction_example()
            written = sorted(p.name for p in (tmp / "out").iterdir())
        self.assertEqual(
            written,
            [
                "label_correction_example.png",
                "label_correction_example_after.png",
                "label_correction_example_before.png",
            ],
        )

    def test_label_example_writes_the_slice_and_the_surfaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gt = np.zeros((300, 300, 8), dtype=np.uint8)
            for k, (x, y) in enumerate(
                ((140, 140), (160, 140), (140, 160), (160, 160)), start=1
            ):
                gt[x : x + 10, y : y + 10, 1:7] = k
            (tmp / "new" / "Patient_01").mkdir(parents=True)
            nib.save(
                nib.Nifti1Image(gt, np.eye(4)),
                str(tmp / "new" / "Patient_01" / "GT.nii.gz"),
            )
            ct = np.random.default_rng(0).integers(-100, 200, gt.shape).astype(np.int16)
            nib.save(
                nib.Nifti1Image(ct, np.eye(4)),
                str(tmp / "new" / "Patient_01" / "Patient_01.nii.gz"),
            )
            argv = [
                "x",
                "--data-dir",
                str(tmp / "new"),
                "--patient",
                "Patient_01",
                "--out-dir",
                str(tmp / "out"),
            ]
            with mock.patch.object(sys, "argv", argv):
                draw_label_example()
            written = sorted(p.name for p in (tmp / "out").iterdir())
        self.assertEqual(written, ["label_example_2D.png", "label_example_3D.png"])

    def test_example_slice_needs_both_esophagus_and_aorta(self):
        gt = np.zeros((4, 4, 5), dtype=np.uint8)
        gt[0:3, 0, 1] = 1  # large esophagus but only one aorta pixel on this slice
        gt[0, 1, 1] = 4
        gt[0:2, 0, 3] = 1
        gt[0:2, 1, 3] = 4
        gt[0:4, 2, 4] = 1  # esophagus alone
        self.assertEqual(example_slice(gt), 3)

    def test_organ_slice_needs_all_four_organs(self):
        gt = np.zeros((4, 4, 5), dtype=np.uint8)
        gt[0:4, 0, 4] = 1  # the biggest slice has no aorta
        gt[0:4, 1, 4] = 2
        gt[0:4, 2, 4] = 3
        for k in (1, 2, 3, 4):
            gt[k - 1, 3, 2] = k
        self.assertEqual(organ_slice(gt), 2)

    def test_background_share_is_the_unlabeled_part_of_all_voxels(self):
        patients = pd.DataFrame({"voxels": [60, 40]})
        labels = pd.DataFrame({"voxels": [4, 6]})
        self.assertEqual(background_share(patients, labels), 90)


class LabelMakeupRingsTests(unittest.TestCase):
    def test_wedges_follow_each_other_clockwise_from_twelve_oclock(self):
        angles = wedge_angles([0.25, 0.75])
        self.assertEqual(angles, [(0.0, 90.0), (-270.0, 0.0)])
        self.assertEqual(wedge_angles([1.0], start=0.0), [(-360.0, 0.0)])

    def test_organ_voxels_pool_patients_and_drop_absent_organs(self):
        labels = pd.DataFrame(
            {
                "patient": ["a", "a", "b", "b", "b"],
                "label": [3, 1, 3, 1, 2],
                "voxels": [5, 10, 7, 20, 40],
            }
        )
        voxels = organ_voxels(labels)
        self.assertEqual(list(voxels.index), [1, 2, 3])
        self.assertEqual(list(voxels), [30, 40, 12])
        labels.loc[labels.label == 2, "voxels"] = 0
        self.assertEqual(list(organ_voxels(labels).index), [1, 3])

    def test_makeup_rings_write_the_figure_from_two_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for folder, rows in (
                ("before", [(1, 30), (2, 60), (3, 10)]),
                ("after", [(1, 5), (2, 60), (3, 10), (4, 25)]),
            ):
                (tmp / folder).mkdir()
                pd.DataFrame(
                    [
                        {"patient": "Patient_01", "label": k, "voxels": v}
                        for k, v in rows
                    ]
                ).to_csv(tmp / folder / "labels.csv", index=False)
                pd.DataFrame({"patient": ["Patient_01"], "voxels": [10000]}).to_csv(
                    tmp / folder / "patients.csv", index=False
                )
            argv = [
                "x",
                "--profile-dir",
                str(tmp / "before"),
                str(tmp / "after"),
                "--out-dir",
                str(tmp / "out"),
            ]
            with mock.patch.object(sys, "argv", argv):
                draw_label_makeup_rings()
            written = sorted(p.name for p in (tmp / "out").iterdir())
        self.assertEqual(written, ["label_makeup_rings.png"])


def position_labels() -> pd.DataFrame:
    """A labels.csv for two patients with a heart (2), an esophagus (1) and an aorta (4)."""
    rows = []
    for patient, shift in (("Patient_01", 0.0), ("Patient_02", 100.0)):
        for label, offset in ((2, 0.0), (1, 20.0), (4, -30.0)):
            row = {"patient": patient, "label": label, "voxels": 10}
            for i, a in enumerate("xyz"):
                center = shift + offset * (i + 1)
                row |= {
                    f"{a}_center_mm": center,
                    f"{a}_min_mm": center - 9.0,
                    f"{a}_max_mm": center + 9.0,
                    f"{a}_size_mm": 20.0,
                }
            rows.append(row)
    return pd.DataFrame(rows)


class OrganPosition3DTests(unittest.TestCase):
    def test_organ_centers_are_relative_to_each_patients_heart(self):
        centers = organ_centers(position_labels(), 1)
        self.assertEqual(centers.tolist(), [[20.0, 40.0, 60.0]] * 2)
        self.assertEqual(
            organ_centers(position_labels(), 2).tolist(), [[0.0, 0.0, 0.0]] * 2
        )

    def test_axis_angle_is_never_upside_down(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.view_init(elev=14, azim=-52)
        fig.canvas.draw()
        for start, end in (
            ((0, 0, 0), (1, 0, 0)),
            ((1, 0, 0), (0, 0, 0)),
            ((1, 0, 0), (1, 1, 0)),
        ):
            self.assertTrue(-90 <= axis_angle(ax, start, end) <= 90)
        self.assertAlmostEqual(
            axis_angle(ax, (0, 0, 0), (1, 0, 0)), axis_angle(ax, (1, 0, 0), (0, 0, 0))
        )
        plt.close(fig)

    def test_position_figure_is_written_to_its_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "profile"
            profile.mkdir()
            position_labels().to_csv(profile / "labels.csv", index=False)
            with mock.patch.object(sys, "argv", ["x", "--profile-dir", str(profile)]):
                draw_organ_position_3d()
            self.assertTrue(
                (profile / "organ_position_3d" / "organ_position_3d.png").is_file()
            )


class RunAllFiguresTests(unittest.TestCase):
    def test_default_plan_profiles_first_and_has_no_comparison(self):
        labels = [label for label, _ in build_plan(parse_args(["--out-dir", "out"]))]
        self.assertEqual(
            labels,
            ["dataset_profile", "scan_geometry", "organ_position_3d", "label_example"],
        )

    def test_before_dataset_is_only_profiled_and_compared(self):
        args = parse_args(
            [
                "--data-dir",
                "new/train",
                "--out-dir",
                "figs",
                "--before-data-dir",
                "old/train",
                "--names",
                "a",
                "b",
            ]
        )
        plan = dict(build_plan(args))
        before = plan["before/dataset_profile"]
        self.assertEqual(before[before.index("--data-dir") + 1], "old/train")
        cmd = plan["label_hu_distribution"]
        self.assertEqual(
            cmd[cmd.index("--profile-dir") + 1 : cmd.index("--names")],
            ["figs/before/profile", "figs/profile"],
        )
        self.assertEqual(cmd[cmd.index("--out-dir") + 1], "figs/comparison")
        cmd = plan["connected_components"]
        self.assertEqual(cmd[cmd.index("--data-dir") + 1], "new/train")
        self.assertEqual(cmd[cmd.index("--before-data-dir") + 1], "old/train")
        self.assertEqual(plan["label_correction_example"][-1], "figs/comparison")


if __name__ == "__main__":
    unittest.main()
