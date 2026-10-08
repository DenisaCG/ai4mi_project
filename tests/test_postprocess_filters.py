"""Ground-truth-free filters on tiny volumes."""

import unittest

import numpy as np

from tools.postprocess_filters import (
    adjacent_gate,
    area_gate,
    contiguous_z,
    heart_hull,
    majority_vote,
    min_run_length,
    remove_small_components,
    slice_components,
    take_organs,
    threshold_vote,
    trachea_anchor,
    z_extent,
)


class TakeOrgansTest(unittest.TestCase):
    def test_only_listed_organs_come_from_source(self):
        base = np.array([[1, 1, 2, 0]], dtype=np.uint8)
        source = np.array([[1, 0, 0, 2]], dtype=np.uint8)
        np.testing.assert_array_equal(take_organs(base, source, [1]), [[1, 0, 2, 0]])
        np.testing.assert_array_equal(base, [[1, 1, 2, 0]])


class RemoveSmallComponentsTest(unittest.TestCase):
    def setUp(self):
        self.pred = np.zeros((10, 10, 10), dtype=np.uint8)
        self.pred[0:5, 0:5, 0:4] = 1  # 100 voxels
        self.pred[8, 8, 8:10] = 1  # 2 voxels
        self.pred[8, 0, 0:8] = 1  # 8 voxels
        self.pred[9, 9, 0] = 2  # another organ

    def test_relative_threshold(self):
        out = remove_small_components(self.pred, 1, rel=0.05)  # below 5 voxels goes
        self.assertEqual(int((out == 1).sum()), 108)
        out = remove_small_components(self.pred, 1, rel=0.2)  # below 20 voxels goes
        self.assertEqual(int((out == 1).sum()), 100)

    def test_voxel_threshold_and_other_organs_untouched(self):
        out = remove_small_components(self.pred, 1, min_voxels=9)
        self.assertEqual(int((out == 1).sum()), 100)
        self.assertEqual(out[9, 9, 0], 2)

    def test_largest_is_kept_even_below_the_threshold(self):
        out = remove_small_components(self.pred, 1, min_voxels=500)
        self.assertEqual(int((out == 1).sum()), 100)

    def test_input_unchanged_and_missing_organ(self):
        before = self.pred.copy()
        remove_small_components(self.pred, 1, rel=0.5)
        np.testing.assert_array_equal(self.pred, before)
        np.testing.assert_array_equal(remove_small_components(self.pred, 3, 0.5), before)


class HeartHullTest(unittest.TestCase):
    def test_fills_concavity_without_overwriting_other_organs(self):
        pred = np.zeros((9, 9, 9), dtype=np.uint8)
        pred[1:8, 1:8, 1:8] = 2
        pred[4, 4, 1:8] = 0  # a channel through the cube
        pred[4, 4, 4] = 3  # another organ inside it
        out = heart_hull(pred, 2)
        self.assertEqual(out[4, 4, 2], 2)
        self.assertEqual(out[4, 4, 4], 3)
        self.assertEqual(int((out == 2).sum()), 7**3 - 1)
        self.assertEqual(pred[4, 4, 2], 0)  # input unchanged

    def test_hull_of_scattered_points_stays_in_the_volume(self):
        pred = np.zeros((6, 6, 6), dtype=np.uint8)
        pred[0, 0, 0] = pred[5, 0, 0] = pred[0, 5, 0] = pred[0, 0, 5] = 2  # a tetrahedron
        out = heart_hull(pred, 2)
        self.assertEqual(out.shape, pred.shape)
        self.assertGreater(int((out == 2).sum()), 4)

    def test_missing_organ(self):
        pred = np.zeros((3, 3, 3), dtype=np.uint8)
        np.testing.assert_array_equal(heart_hull(pred, 2), pred)


class AreaGateTest(unittest.TestCase):
    def test_small_slices_removed_per_organ(self):
        pred = np.zeros((4, 4, 3), dtype=np.uint8)
        pred[0, 0, 0] = 1  # 1 voxel on z = 0
        pred[:, :2, 1] = 1  # 8 voxels on z = 1
        pred[0, 0, 2] = 2  # organ 2: 1 voxel
        out = area_gate(pred, [1], min_voxels=5)
        self.assertEqual(int((out[:, :, 0] == 1).sum()), 0)
        self.assertEqual(int((out[:, :, 1] == 1).sum()), 8)
        self.assertEqual(out[0, 0, 2], 2)  # organ 2 is not in `classes`

    def test_threshold_is_exclusive_and_input_unchanged(self):
        pred = np.zeros((2, 3, 1), dtype=np.uint8)
        pred[0, :, 0] = 1  # 3 voxels
        np.testing.assert_array_equal(area_gate(pred, [1], 3), pred)
        self.assertEqual(int(area_gate(pred, [1], 3.5).sum()), 0)
        self.assertEqual(int(pred.sum()), 3)


class AdjacentGateTest(unittest.TestCase):
    def test_isolated_slices_removed_runs_kept(self):
        pred = np.zeros((2, 2, 7), dtype=np.uint8)
        pred[0, 0, [0, 1, 3, 6]] = 1  # z = 0, 1 adjacent; 3 and 6 isolated
        out = adjacent_gate(pred, [1])
        self.assertEqual(np.flatnonzero(out[0, 0] == 1).tolist(), [0, 1])

    def test_judged_on_the_input_not_iteratively(self):
        pred = np.zeros((1, 1, 3), dtype=np.uint8)
        pred[0, 0, :2] = [1, 2]  # organs on different slices do not support each other
        out = adjacent_gate(pred, [1, 2])
        self.assertEqual(int(out.sum()), 0)

    def test_other_organ_untouched(self):
        pred = np.zeros((1, 1, 3), dtype=np.uint8)
        pred[0, 0, 0] = 3
        np.testing.assert_array_equal(adjacent_gate(pred, [1]), pred)


class MinRunLengthTest(unittest.TestCase):
    def test_short_runs_removed(self):
        pred = np.zeros((1, 1, 9), dtype=np.uint8)
        pred[0, 0, [0, 1, 3, 4, 5, 7]] = 1  # runs of 2, 3 and 1
        out = min_run_length(pred, [1], 3)
        self.assertEqual(np.flatnonzero(out[0, 0]).tolist(), [3, 4, 5])
        self.assertEqual(np.flatnonzero(min_run_length(pred, [1], 2)[0, 0]).tolist(), [0, 1, 3, 4, 5])
        self.assertEqual(int(pred.sum()), 6)

    def test_other_organs_untouched_and_absent_organ(self):
        pred = np.zeros((1, 1, 3), dtype=np.uint8)
        pred[0, 0, 0] = 2
        np.testing.assert_array_equal(min_run_length(pred, [1], 5), pred)


class ZExtentTest(unittest.TestCase):
    def test_slices_outside_the_largest_component_span_removed(self):
        pred = np.zeros((4, 4, 12), dtype=np.uint8)
        pred[:, :, 4:8] = 1  # main component, z 4..7
        pred[0, 0, 11] = 1  # isolated, far away
        pred[0, 0, 2] = 1  # isolated, 2 slices below
        self.assertEqual(np.flatnonzero(z_extent(pred, [1], 0)[0, 0]).tolist(), [4, 5, 6, 7])
        self.assertEqual(np.flatnonzero(z_extent(pred, [1], 2)[0, 0]).tolist(), [2, 4, 5, 6, 7])

    def test_margin_clipped_to_the_scan_and_missing_organ(self):
        pred = np.zeros((2, 2, 4), dtype=np.uint8)
        pred[:, :, 0:2] = 1
        np.testing.assert_array_equal(z_extent(pred, [1, 2], 9), pred)


class SliceComponentsTest(unittest.TestCase):
    def test_small_piece_on_a_slice_removed(self):
        pred = np.zeros((10, 10, 2), dtype=np.uint8)
        pred[0:4, 0:4, 0] = 1  # 16 px
        pred[8, 8, 0] = 1  # 1 px
        pred[8, 8, 1] = 1  # alone on its slice: kept
        out = slice_components(pred, [1], 0.25)
        self.assertEqual(int((out[:, :, 0] == 1).sum()), 16)
        self.assertEqual(out[8, 8, 1], 1)

    def test_similar_pieces_kept_and_input_unchanged(self):
        pred = np.zeros((10, 10, 1), dtype=np.uint8)
        pred[0:4, 0:4, 0] = 1  # 16 px
        pred[6:9, 6:9, 0] = 1  # 9 px
        before = pred.copy()
        self.assertEqual(int(slice_components(pred, [1], 0.25).sum()), 25)
        self.assertEqual(int(slice_components(pred, [1], 0.7).sum()), 16)
        np.testing.assert_array_equal(pred, before)


class MajorityVoteTest(unittest.TestCase):
    def test_majority_label_wins(self):
        a = np.array([[0, 1, 2, 3]], dtype=np.uint8)
        b = np.array([[0, 1, 1, 3]], dtype=np.uint8)
        c = np.array([[1, 0, 1, 2]], dtype=np.uint8)
        np.testing.assert_array_equal(majority_vote([a, b, c]), [[0, 1, 1, 3]])

    def test_three_way_tie_takes_the_first_volume(self):
        vols = [np.array([[2]], np.uint8), np.array([[0]], np.uint8), np.array([[3]], np.uint8)]
        np.testing.assert_array_equal(majority_vote(vols), [[2]])

    def test_a_false_positive_slice_in_one_volume_is_voted_away(self):
        clean = np.zeros((2, 2, 4), dtype=np.uint8)
        clean[:, :, 1:3] = 1
        noisy = clean.copy()
        noisy[0, 0, 3] = 1
        out = majority_vote([noisy, clean, clean])
        np.testing.assert_array_equal(out, clean)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(noisy[0, 0, 3], 1)  # inputs unchanged

    def test_single_volume_is_returned_as_is(self):
        v = np.array([[0, 4]], dtype=np.uint8)
        np.testing.assert_array_equal(majority_vote([v]), v)


class ThresholdVoteTest(unittest.TestCase):
    def setUp(self):
        # one voxel row: seeds disagree on whether the organ is there
        self.vols = [np.array([[1, 1, 1, 0]], np.uint8), np.array([[1, 1, 0, 0]], np.uint8), np.array([[1, 0, 0, 2]], np.uint8)]

    def test_union_majority_intersection(self):
        for t, expected in ((1, [[1, 1, 1, 2]]), (2, [[1, 1, 0, 0]]), (3, [[1, 0, 0, 0]])):
            np.testing.assert_array_equal(
                threshold_vote(self.vols, [1, 2], {1: t, 2: t}), expected, f"t={t}"
            )

    def test_threshold_is_per_organ(self):
        out = threshold_vote(self.vols, [1, 2], {1: 3, 2: 1})
        np.testing.assert_array_equal(out, [[1, 0, 0, 2]])

    def test_most_votes_wins_among_eligible_organs_and_ties_go_to_the_first_volume(self):
        vols = [np.array([[2, 1, 2]], np.uint8), np.array([[1, 1, 1]], np.uint8), np.array([[1, 2, 0]], np.uint8)]
        out = threshold_vote(vols, [1, 2], {1: 1, 2: 1})
        np.testing.assert_array_equal(out, [[1, 1, 2]])  # voxel 2: organs 1 and 2 tie on one vote -> first volume says 2

    def test_equals_the_majority_vote_for_three_volumes_without_three_way_ties(self):
        vols = [np.array([[0, 1, 1, 2]], np.uint8), np.array([[0, 1, 0, 2]], np.uint8), np.array([[1, 0, 0, 1]], np.uint8)]
        np.testing.assert_array_equal(threshold_vote(vols, [1, 2], {1: 2, 2: 2}), majority_vote(vols))


class ContiguousZTest(unittest.TestCase):
    def setUp(self):
        self.pred = np.zeros((1, 1, 20), dtype=np.uint8)
        self.pred[0, 0, [0, 1, 2]] = 1  # run of 3
        self.pred[0, 0, [5, 6]] = 1  # 2 empty slices after the first run
        self.pred[0, 0, [12, 13, 14, 15]] = 1  # run of 4, far away
        self.pred[0, 0, 18] = 2

    def kept(self, gap):
        return np.flatnonzero(contiguous_z(self.pred, [1, 2], gap)[0, 0] == 1).tolist()

    def test_gap_joins_runs_and_the_longest_is_kept(self):
        self.assertEqual(self.kept(0), [12, 13, 14, 15])
        self.assertEqual(self.kept(2), [0, 1, 2, 5, 6])  # 5 predicted slices beat 4
        self.assertEqual(self.kept(5), [0, 1, 2, 5, 6, 12, 13, 14, 15])  # gap of 5 empty slices joins all

    def test_other_organ_single_run_kept_input_unchanged_and_missing_organ(self):
        out = contiguous_z(self.pred, [1, 2, 3], 0)
        self.assertEqual(out[0, 0, 18], 2)
        self.assertEqual(int((self.pred == 1).sum()), 9)


class TracheaAnchorTest(unittest.TestCase):
    def setUp(self):
        self.pred = np.zeros((40, 40, 6), dtype=np.uint8)
        self.pred[18:22, 18:22, :] = 3  # trachea centred at (19.5, 19.5)
        self.pred[24:26, 18:22, 0:6] = 1  # esophagus beside it: ~5.5 voxels away
        self.pred[2:4, 2:4, 0:2] = 1  # a piece far away (~24 voxels)

    def test_components_beyond_the_radius_removed(self):
        out = trachea_anchor(self.pred, 1, 3, radius=10)
        self.assertEqual(int((out == 1).sum()), int((self.pred[24:26] == 1).sum()))
        self.assertEqual(out[2, 2, 0], 0)
        self.assertEqual(int((out == 3).sum()), int((self.pred == 3).sum()))
        self.assertEqual(int((trachea_anchor(self.pred, 1, 3, radius=40) == 1).sum()), int((self.pred == 1).sum()))

    def test_nearest_slice_with_trachea_is_used_and_no_trachea_removes_nothing(self):
        pred = self.pred.copy()
        pred[:, :, 3:] = 0  # trachea only on slices 0-2, esophagus piece at z 4-5
        pred[30:32, 30:32, 4:6] = 1  # far from the trachea
        out = trachea_anchor(pred, 1, 3, radius=10)
        self.assertEqual(int((out[30:32, 30:32, 4:6] == 1).sum()), 0)
        none = self.pred.copy()
        none[none == 3] = 0
        np.testing.assert_array_equal(trachea_anchor(none, 1, 3, radius=1), none)


if __name__ == "__main__":
    unittest.main()
