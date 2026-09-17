import unittest

from src.config import config_hash, load_config, merge, parse_overrides, read_yaml

CFG = "configs/segthor_enet_ce.yaml"


class ConfigTests(unittest.TestCase):
    def test_experiment_overrides_base(self):
        cfg = load_config(CFG)
        self.assertEqual(cfg["experiment"], "segthor_enet_ce")
        self.assertEqual(cfg["model"]["kwargs"], {"kernels": 8, "factor": 2})
        self.assertEqual(cfg["optim"]["kwargs"]["lr"], 0.0005)  # inherited from base

    def test_unknown_key_rejected_but_kwargs_free_form(self):
        with self.assertRaises(KeyError):
            merge({"train": {"epochs": 1}}, {"train": {"epoch": 2}})
        merged = merge({"model": {"kwargs": {}}}, {"model": {"kwargs": {"depth": 4}}})
        self.assertEqual(merged["model"]["kwargs"]["depth"], 4)

    def test_switching_component_drops_the_old_kwargs(self):
        base = {"optim": {"name": "adam", "kwargs": {"lr": 0.0005, "betas": [0.9, 0.999]}}}
        switched = merge(base, {"optim": {"name": "sgd", "kwargs": {"lr": 0.01, "momentum": 0.9}}})
        self.assertEqual(switched["optim"]["kwargs"], {"lr": 0.01, "momentum": 0.9})
        self.assertEqual(merge(switched, {"optim": {"kwargs": {"lr": 0.05}}})["optim"]["kwargs"],
                         {"lr": 0.05, "momentum": 0.9})
        kept = merge(base, {"optim": {"kwargs": {"lr": 0.001}}})
        self.assertEqual(kept["optim"]["kwargs"], {"lr": 0.001, "betas": [0.9, 0.999]})
        self.assertEqual(load_config(CFG, ["optim.name=sgd", "optim.kwargs.lr=0.01"])["optim"]["kwargs"],
                         {"lr": 0.01})

    def test_cli_overrides_are_typed(self):
        self.assertEqual(parse_overrides(["train.epochs=5", "optim.kwargs.lr=1e-3", "wandb.tags=[a, b]"]),
                         {"train": {"epochs": 5}, "optim": {"kwargs": {"lr": 1e-3}}, "wandb": {"tags": ["a", "b"]}})
        self.assertEqual(load_config(CFG, ["train.epochs=3"])["train"]["epochs"], 3)
        # PyYAML alone reads `1e-4` as the string '1e-4' (YAML 1.1); config files must get a float
        self.assertEqual(read_yaml("lr: 1e-4\nb: [0.9, 1.5E+2]\nname: e5"), {"lr": 1e-4, "b": [0.9, 150.0], "name": "e5"})

    def test_unregistered_component_fails_before_the_run_starts(self):
        for override in ("model.name=unet", "loss.name=dice", "optim.name=lamb", "scheduler.name=poly"):
            with self.assertRaises(ValueError, msg=override) as caught:
                load_config(CFG, [override])
            self.assertIn("available:", str(caught.exception))
        load_config(CFG, ["optim.name=sgd", "optim.kwargs.lr=0.01", "scheduler.name=cosine"])

    def test_smoke_layer_then_overrides(self):
        cfg = load_config(CFG, ["wandb.mode=offline"], smoke=True)
        self.assertEqual(cfg["train"]["epochs"], 2)
        self.assertEqual(cfg["wandb"]["mode"], "offline")
        self.assertTrue(cfg["paths"]["run_root"].endswith("_smoke"))

    def test_hash_ignores_settings_that_cannot_change_results(self):
        a = load_config(CFG)
        for override in (["notes=hi", "wandb.mode=disabled"], ["device=cpu"], ["data.num_workers=0"],
                         ["paths.run_root=/tmp/runs", "paths.metrics_dir=/tmp/metrics"]):
            self.assertEqual(config_hash(a), config_hash(load_config(CFG, override)), override)
        for override in (["seed=1"], ["train.epochs=5"], ["data.batch_size=4"], ["optim.kwargs.lr=0.01"]):
            self.assertNotEqual(config_hash(a), config_hash(load_config(CFG, override)), override)


if __name__ == "__main__":
    unittest.main()
