"""Test-time augmentation: the warps are invertible, the plain variant is the plain prediction."""
import unittest

import torch

from src.eval_tta import TRANSFORMS, VARIANTS, average_probs, predict_variants, theta, unwarp, warp


def blob(size=64, cx=0.0, cy=0.0):
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    return torch.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / 0.05))[None, None]  # (1, 1, H, W)


class WarpTests(unittest.TestCase):
    def test_identity_is_a_no_op(self):
        x = torch.rand(2, 3, 32, 32)
        torch.testing.assert_close(warp(x, theta()), x, atol=1e-5, rtol=0)

    def test_flip_is_its_own_inverse(self):
        x = torch.rand(2, 1, 32, 32)
        flipped = warp(x, TRANSFORMS["flip"])
        torch.testing.assert_close(flipped, x.flip(-1), atol=1e-5, rtol=0)
        back, cover = unwarp(flipped, TRANSFORMS["flip"])
        torch.testing.assert_close(back, x, atol=1e-5, rtol=0)
        self.assertTrue(torch.all(cover > 0.999))

    def test_round_trip_returns_the_content_to_where_it_was(self):
        x = blob(cx=0.1, cy=-0.2)
        for name, th in TRANSFORMS.items():
            back, cover = unwarp(warp(x, th), th)
            where = lambda t: divmod(int(t.flatten().argmax()), t.shape[-1])
            self.assertEqual(where(back * (cover > 0.5)), where(x), name)
            inside = cover > 0.99
            self.assertLess(float((back - x)[inside].abs().max()), 0.08, name)

    def test_coverage_marks_pixels_the_warped_input_did_not_reach(self):
        _, cover = unwarp(warp(torch.ones(1, 1, 64, 64), TRANSFORMS["shift_xp"]), TRANSFORMS["shift_xp"])
        self.assertLess(float(cover.min()), 0.5)
        self.assertAlmostEqual(float(cover.max()), 1.0, places=4)

    def test_fill_value_is_used_outside_the_image(self):
        out = warp(torch.zeros(1, 1, 32, 32), TRANSFORMS["shift_xp"], fill=-5.0)
        self.assertAlmostEqual(float(out.min()), -5.0, places=4)
        self.assertAlmostEqual(float(out.max()), 0.0, places=4)


class VariantTests(unittest.TestCase):
    def test_plain_variant_is_the_plain_prediction_and_all_are_label_maps(self):
        torch.manual_seed(0)
        net = torch.nn.Conv2d(1, 5, 3, padding=1).eval()
        images = torch.rand(7, 1, 32, 32)
        out = predict_variants(net, images, fill=0.0, device=torch.device("cpu"), batch=3)
        self.assertEqual(set(out), set(VARIANTS))
        plain = net(images).argmax(dim=1).numpy()
        self.assertTrue((out["none"] == plain).all())
        for v, labels in out.items():
            self.assertEqual(labels.shape, (7, 32, 32), v)
            self.assertEqual(labels.dtype.name, "uint8")
            self.assertLessEqual(int(labels.max()), 4)

    def test_average_probs_matches_predict_variants_and_is_a_distribution(self):
        torch.manual_seed(1)
        net = torch.nn.Conv2d(1, 5, 3, padding=1).eval()
        images = torch.rand(5, 1, 32, 32)
        plain = average_probs(net, images, [], 0.0)
        torch.testing.assert_close(plain, net(images).softmax(dim=1), atol=1e-6, rtol=0)
        for v in ("rot", "all"):
            probs = average_probs(net, images, VARIANTS[v], -1.0)
            torch.testing.assert_close(probs.sum(dim=1), torch.ones(5, 32, 32), atol=1e-5, rtol=0)
            labels = predict_variants(net, images, -1.0, torch.device("cpu"), batch=2)[v]
            self.assertTrue((probs.argmax(dim=1).numpy() == labels).all(), v)

    def test_a_translation_equivariant_net_is_unchanged_by_shift_tta(self):
        net = torch.nn.Conv2d(1, 5, 1).eval()  # per-pixel classifier: shifting the input just shifts the output
        images = blob(size=32).repeat(2, 1, 1, 1)
        out = predict_variants(net, images, fill=0.0, device=torch.device("cpu"), batch=2)
        agree = float((out["shift"] == out["none"]).mean())
        self.assertGreater(agree, 0.9)


if __name__ == "__main__":
    unittest.main()
