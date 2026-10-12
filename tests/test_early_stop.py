"""Early stopping rule."""

import unittest

from src.engine import should_stop


class EarlyStopTests(unittest.TestCase):
    def test_stops_after_patience_epochs_without_gain(self):
        self.assertFalse(should_stop(epoch=14, best_epoch=0, patience=15))
        self.assertTrue(should_stop(epoch=15, best_epoch=0, patience=15))

    def test_patience_zero_never_stops(self):
        self.assertFalse(should_stop(epoch=1000, best_epoch=0, patience=0))


if __name__ == "__main__":
    unittest.main()
