"""Oracle variants on tiny volumes: gating by GT slice presence and the largest 6-connected component per organ."""

import unittest

import numpy as np

from tools.gating_oracle import gate, largest_component


class GateTest(unittest.TestCase):
    def test_organ_removed_only_on_slices_without_gt(self):
        gt = np.zeros((4, 4, 5), dtype=np.uint8)
        gt[:, :, 1:3] = 1  # organ 1 present on z = 1, 2
        pred = np.zeros_like(gt)
        pred[0, 0, 0:4] = 1  # z = 0 and 3 are false positives
        pred[1, 1, 2] = 2  # organ 2 has no GT anywhere
        out = gate(pred, gt, [1, 2])
        self.assertEqual(np.flatnonzero(out[0, 0] == 1).tolist(), [1, 2])
        self.assertFalse((out == 2).any())
        self.assertTrue((pred == 2).any())  # the input is not modified

    def test_other_organs_untouched(self):
        gt = np.zeros((2, 2, 3), dtype=np.uint8)
        gt[:, :, 0] = 1
        pred = np.zeros_like(gt)
        pred[0, 0, 2] = 3
        np.testing.assert_array_equal(gate(pred, gt, [1]), pred)


class LargestComponentTest(unittest.TestCase):
    def test_keeps_largest_component_per_organ(self):
        pred = np.zeros((6, 6, 6), dtype=np.uint8)
        pred[0:3, 0:3, 0:3] = 1  # 27 voxels
        pred[5, 5, 5] = 1  # isolated
        pred[0, 5, 0:2] = 2  # organ 2: 2 voxels
        pred[5, 0, 5] = 2  # organ 2: 1 voxel
        out = largest_component(pred, [1, 2])
        self.assertEqual(int((out == 1).sum()), 27)
        self.assertEqual(out[5, 5, 5], 0)
        self.assertEqual(int((out == 2).sum()), 2)

    def test_diagonal_voxels_are_separate_components(self):
        pred = np.zeros((3, 3, 3), dtype=np.uint8)
        pred[0, 0, 0] = pred[1, 1, 1] = pred[1, 1, 2] = (
            1  # the second pair touches by a face, the first voxel only by a corner
        )
        out = largest_component(pred, [1])
        self.assertEqual(int((out == 1).sum()), 2)
        self.assertEqual(out[0, 0, 0], 0)

    def test_empty_organ_and_input_unchanged(self):
        pred = np.zeros((3, 3, 3), dtype=np.uint8)
        pred[0, 0, 0] = pred[2, 2, 2] = 1
        before = pred.copy()
        largest_component(pred, [1, 2])
        np.testing.assert_array_equal(pred, before)


if __name__ == "__main__":
    unittest.main()
