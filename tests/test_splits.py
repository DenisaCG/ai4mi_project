"""validate_split: the single holdout keeps its old rules, cross-validation needs equal folds covering every patient."""

import unittest

from src.splits import validate_split


class HoldoutTests(unittest.TestCase):
    def test_valid_holdout_indices(self):
        validate_split(20, 5, 0)
        validate_split(20, 5, 3)  # the last of four holdout splits

    def test_holdout_needs_fewer_validation_patients_than_patients(self):
        for retains in (0, -1, 40, 41):
            with self.assertRaises(ValueError, msg=retains):
                validate_split(40, retains, 0)

    def test_holdout_index_must_select_a_full_validation_set(self):
        with self.assertRaises(ValueError):
            validate_split(20, 5, 4)  # patients 20..24 do not exist
        with self.assertRaises(ValueError):
            validate_split(20, 5, -1)

    def test_non_integers_are_rejected(self):
        for retains, fold in ((5.0, 0), (True, 0), (5, 0.0), (5, False)):
            with self.assertRaises(ValueError, msg=(retains, fold)):
                validate_split(20, retains, fold)


class CrossValidationTests(unittest.TestCase):
    def test_four_folds_of_ten_on_forty_patients(self):
        for fold in range(4):
            validate_split(40, 10, fold, 4)

    def test_folds_must_tile_the_patients_exactly(self):
        for retains, num_folds in ((8, 4), (10, 5), (10, 3), (13, 3)):
            with self.assertRaises(ValueError, msg=(retains, num_folds)):
                validate_split(40, retains, 0, num_folds)

    def test_fold_index_must_be_below_num_folds(self):
        with self.assertRaises(ValueError):
            validate_split(40, 10, 4, 4)
        with self.assertRaises(ValueError):
            validate_split(40, 10, -1, 4)

    def test_num_folds_range(self):
        for num_folds in (0, 1, 41):
            with self.assertRaises(ValueError, msg=num_folds):
                validate_split(40, 10, 0, num_folds)


if __name__ == "__main__":
    unittest.main()
