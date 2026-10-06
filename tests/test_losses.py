import unittest

import torch
import torch.nn.functional as F

import src.losses  # noqa: F401  registers losses
from src.registry import available, build
from utils import class2one_hot

K = 5


def batch(seed: int, b: int = 4, hw: int = 16):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(b, K, hw, hw, generator=g, requires_grad=True)
    gt = class2one_hot(torch.randint(0, K, (b, hw, hw), generator=g), K)
    return F.softmax(logits, dim=1), gt, logits


class LossTests(unittest.TestCase):
    def test_registered(self):
        self.assertTrue({"cross_entropy", "soft_dice", "dice_ce"} <= set(available("loss")))

    def test_soft_dice_extremes(self):
        _, gt, _ = batch(0)
        perfect = build("loss", "soft_dice", num_classes=K)(gt.float(), gt)
        self.assertAlmostEqual(perfect.item(), 0.0, places=5)
        wrong = build("loss", "soft_dice", num_classes=K)(1 - gt.float(), gt)
        self.assertGreater(wrong.item(), 0.9)

    def test_soft_dice_default_is_foreground_only(self):
        probs, gt, _ = batch(1)
        default = build("loss", "soft_dice", num_classes=K)(probs, gt)
        explicit = build("loss", "soft_dice", num_classes=K, idk=[1, 2, 3, 4])(probs, gt)
        with_bg = build("loss", "soft_dice", num_classes=K, idk=list(range(K)))(probs, gt)
        self.assertEqual(default.item(), explicit.item())
        self.assertNotEqual(default.item(), with_bg.item())

    def test_absent_class_is_finite_with_finite_gradient(self):
        probs, gt, logits = batch(2)
        gt[:, 4] = 0  # aorta absent from the whole batch, as in the original SEGTHOR labels
        loss = build("loss", "soft_dice", num_classes=K)(probs, gt)
        loss.backward()
        self.assertTrue(torch.isfinite(loss) and torch.isfinite(logits.grad).all())

    def test_dice_ce_is_weighted_sum(self):
        probs, gt, _ = batch(3)
        ce = build("loss", "cross_entropy", num_classes=K)(probs, gt)
        dice = build("loss", "soft_dice", num_classes=K)(probs, gt)
        combined = build("loss", "dice_ce", num_classes=K, ce_weight=0.3, dice_weight=2.0)(probs, gt)
        torch.testing.assert_close(combined, 0.3 * ce + 2.0 * dice)

    def test_foreground_ce_ignores_background_pixels(self):
        """CE with idk=[1..4] is blind to what is predicted on background pixels; CE over all classes is not."""
        probs, gt, _ = batch(5)
        bg = gt[:, 0:1].bool()  # (B, 1, H, W), True on background pixels
        other = F.softmax(torch.randn_like(probs), dim=1)
        changed = torch.where(bg, other, probs)  # different prediction on background pixels only
        fg_ce = build("loss", "cross_entropy", num_classes=K, idk=[1, 2, 3, 4])
        all_ce = build("loss", "cross_entropy", num_classes=K, idk=list(range(K)))
        torch.testing.assert_close(fg_ce(probs, gt), fg_ce(changed, gt))
        self.assertNotAlmostEqual(all_ce(probs, gt).item(), all_ce(changed, gt).item(), places=4)

    def test_dice_ce_foreground_ce_is_weighted_sum(self):
        probs, gt, _ = batch(6)
        fg = [1, 2, 3, 4]
        ce = build("loss", "cross_entropy", num_classes=K, idk=fg)(probs, gt)
        dice = build("loss", "soft_dice", num_classes=K, idk=fg)(probs, gt)
        combined = build("loss", "dice_ce", num_classes=K, ce_idk=fg, dice_idk=fg)(probs, gt)
        torch.testing.assert_close(combined, ce + dice)
        default = build("loss", "dice_ce", num_classes=K)(probs, gt)  # CE over all classes
        self.assertNotAlmostEqual(combined.item(), default.item(), places=4)

    def test_dice_ce_trains(self):
        """A few gradient steps on the logits lower the combined loss."""
        _, gt, logits = batch(4)
        loss_fn = build("loss", "dice_ce", num_classes=K)
        opt = torch.optim.SGD([logits], lr=1.0)
        first = loss_fn(F.softmax(logits, dim=1), gt).item()
        for _ in range(20):
            opt.zero_grad()
            loss_fn(F.softmax(logits, dim=1), gt).backward()
            opt.step()
        self.assertLess(loss_fn(F.softmax(logits, dim=1), gt).item(), first)

    def test_tversky_with_equal_weights_is_dice(self):
        probs, gt, _ = batch(6)
        dice = build("loss", "soft_dice", num_classes=K)(probs, gt)
        tversky = build("loss", "soft_tversky", num_classes=K, alpha=0.5, beta=0.5)(probs, gt)
        torch.testing.assert_close(tversky, dice)

    def test_tversky_with_beta_above_alpha_punishes_missed_foreground(self):
        """Two foreground classes with the same TP; one misses 2 pixels, the other adds 2."""

        def scores(alpha: float, beta: float) -> tuple[float, float]:
            loss_fn = build("loss", "soft_tversky", num_classes=2, alpha=alpha, beta=beta)

            def line(n: int) -> torch.Tensor:  # one-hot map (1, 2, 1, 16) with n foreground pixels
                fg = (torch.arange(16) < n).float().reshape(1, 1, 1, 16)
                return torch.cat([1 - fg, fg], dim=1)

            missed = loss_fn(line(6), line(8)).item()  # TP 6, FN 2, FP 0
            extra = loss_fn(line(8), line(6)).item()  # TP 6, FN 0, FP 2
            return missed, extra

        missed, extra = scores(0.5, 0.5)
        self.assertAlmostEqual(missed, extra, places=5)
        missed, extra = scores(0.3, 0.7)
        self.assertGreater(missed, extra)

    def test_tversky_ce_is_the_sum_of_its_terms(self):
        probs, gt, _ = batch(8)
        ce = build("loss", "cross_entropy", num_classes=K)(probs, gt)
        tversky = build("loss", "soft_tversky", num_classes=K, alpha=0.3, beta=0.7)(probs, gt)
        combined = build("loss", "tversky_ce", num_classes=K, alpha=0.3, beta=0.7)(probs, gt)
        torch.testing.assert_close(combined, ce + tversky)


if __name__ == "__main__":
    unittest.main()
