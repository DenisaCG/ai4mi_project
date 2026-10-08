"""Number of trainable parameters of every architecture in the cross-validation, from the configs of the runs.

    python dataset_analysis/model_parameters.py [--out figures]

Builds each model from the config.yaml of fold 0 / seed 0 of its experiment (CPU, no weights loaded) and counts the parameters with
requires_grad. Experiments without a local config are listed and skipped. Writes model_parameters.csv.
"""

import argparse
import sys
from pathlib import Path

import yaml
from utils import REPO, write_csv

sys.path.append(str(REPO))
import src.models  # noqa: E402,F401 -- registers the models
from src.registry import build  # noqa: E402

EXPERIMENTS = [
    "full_cv4_enet_ce",
    "full_cv4_enet_dice_ce",
    "full_cv4_enet_dice_ce_25d_c1",
    "full_cv4_enet_dice_ce_25d_c2",
    "full_cv4_unet_dice_ce",
    "full_cv4_attention_unet_dice_ce",
    "full_cv4_resenc_unet_dice_ce",
    "full_cv4_ds_unet_dice_ce",
    "full_cv4_resenc_ds_unet_dice_ce",
    "full_cv4_dino_unet_dice_ce",
]


def count_parameters(config: dict) -> int:
    """Trainable parameters of the model described by a run config.

    Args:
        config: Parsed config.yaml of a run.

    Returns:
        Number of parameters with requires_grad.
    """
    net = build(
        "model", config["model"]["name"], in_channels=config["data"]["in_channels"],
        num_classes=config["data"]["num_classes"], **config["model"]["kwargs"],
    )  # fmt: skip
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metrics-dir", type=Path, default=REPO / "metrics")
    parser.add_argument("--out", type=Path, default=REPO / "figures")
    args = parser.parse_args(argv)

    rows = []
    for experiment in EXPERIMENTS:
        path = args.metrics_dir / f"{experiment}_fold0" / "seed0" / "config.yaml"
        if not path.exists():
            print(f"skipped (no local config): {experiment}")
            continue
        rows.append(
            {
                "experiment": experiment,
                "trainable_parameters": count_parameters(
                    yaml.safe_load(path.read_text())
                ),
            }
        )
        print(f"{experiment}: {rows[-1]['trainable_parameters']:,}")
    write_csv(args.out / "model_parameters.csv", rows)


if __name__ == "__main__":
    main()
