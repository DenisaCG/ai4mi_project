"""Post-processing of the seed volumes on tiny volumes: organ selection, majority and threshold votes."""

import unittest

import numpy as np

from tools.postprocess_filters import majority_vote, take_organs, threshold_vote


class TakeOrgansTest(unittest.TestCase):
    def test_only_listed_organs_come_from_source(self):
        base = np.array([[1, 1, 2, 0]], dtype=np.uint8)
        source = np.array([[1, 0, 0, 2]], dtype=np.uint8)
        np.testing.assert_array_equal(take_organs(base, source, [1]), [[1, 0, 2, 0]])
        np.testing.assert_array_equal(base, [[1, 1, 2, 0]])

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

if __name__ == "__main__":
    unittest.main()
