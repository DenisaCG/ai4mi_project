"""src.predict_test predicts with test-time augmentation by default; --no-tta gives the plain forward pass."""
import unittest

from src.predict_test import LCC_CLASSES, build_parser


class DefaultsTests(unittest.TestCase):
    def parse(self, *extra):
        return build_parser().parse_args(["--runs", "runs/final/seed0", "--out", "predictions/x", *extra])

    def test_tta_is_on_by_default(self):
        self.assertTrue(self.parse().tta)

    def test_tta_can_be_switched_off_and_on_explicitly(self):
        self.assertFalse(self.parse("--no-tta").tta)
        self.assertTrue(self.parse("--tta").tta)

    def test_largest_component_is_kept_for_heart_and_trachea_only(self):
        self.assertEqual(LCC_CLASSES, (2, 3))


if __name__ == "__main__":
    unittest.main()
