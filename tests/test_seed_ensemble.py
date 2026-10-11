"""Seed-ensemble report helpers on hand-made rows."""

import unittest

from tools.seed_ensemble import choose_thresholds, scaled_count_table


def row(variant, k, fp):
    return {"variant": variant, "class_idx": k, "fp_slices": fp}


class ScaledCountTableTest(unittest.TestCase):
    def test_counts_are_scaled_to_the_ensemble_row_count(self):
        rows = [row("baseline", 1, 2)] * 120 + [row("vote", 1, 1)] * 40
        text = scaled_count_table("t", rows, [1], ["bg", "eso"], lambda r: r["fp_slices"])
        self.assertIn("| baseline | 80.0 | 80.0 |", text)  # 240 slices over 120 rows -> per 40 rows
        self.assertIn("| vote | 40.0 | 40.0 |", text)


def vrow(variant, k, dice, hd95, fold=0):
    return {"variant": variant, "run": f"e_fold{fold}/vote", "class_idx": k, "patient": "p", "dice": dice, "hd95": hd95}


class ChooseThresholdsTest(unittest.TestCase):
    def rows(self):
        out = []
        for fold in (0, 1):  # organ 1: t=1 best Dice, t=3 best HD95 at a Dice loss; organ 2: all equal
            out += [vrow("vote_t1", 1, 0.82, 9.0, fold), vrow("vote_t2", 1, 0.80, 8.0, fold), vrow("vote_t3", 1, 0.799, 5.0, fold)]
            out += [vrow(f"vote_t{t}", 2, 0.9, 4.0, fold) for t in (1, 2, 3)]
        return out

    def test_dice_choice_and_majority_wins_ties(self):
        self.assertEqual(choose_thresholds(self.rows(), [1, 2], 3), {1: 1, 2: 2})

    def test_hd95_choice_needs_a_small_dice_loss(self):
        self.assertEqual(choose_thresholds(self.rows(), [1, 2], 3, "hd95"), {1: 3, 2: 2})
        rows = [r | {"dice": r["dice"] - 0.01} if r["variant"] == "vote_t3" else r for r in self.rows()]
        self.assertEqual(choose_thresholds(rows, [1, 2], 3, "hd95")[1], 2)


if __name__ == "__main__":
    unittest.main()
