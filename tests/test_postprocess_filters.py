"""Ground-truth-free filters on tiny volumes."""

import unittest

import numpy as np

from tools.postprocess_filters import (
    adjacent_gate,
    area_gate,
    heart_hull,
    remove_small_components,
    take_organs,
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


if __name__ == "__main__":
    unittest.main()
