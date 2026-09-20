"""Config loading: configs/base.yaml <- experiment yaml <- --set overrides."""
import copy
import hashlib
import json
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]  # relative paths in configs resolve against this
BASE_CONFIG = REPO / "configs" / "base.yaml"
FREE_FORM = "kwargs"  # keys below a `kwargs` block are component-specific, not validated
IGNORED_BY_HASH = ("notes", "wandb", "paths", "device")

PREPROCESS_DEFAULTS = {"shape": [256, 256], "retains": 25, "fold": 0, "seed": 0}  # = slice_segthor.py defaults

# --smoke: a few slices, two epochs, separate run dir, no W&B. Applied before --set overrides.
SMOKE = {"train": {"epochs": 2, "debug_samples": 16}, "wandb": {"mode": "disabled"}}


class _Loader(yaml.SafeLoader):
    """PyYAML follows YAML 1.1, where `1e-4` is a *string*; read it as a float (as YAML 1.2 does)."""


_Loader.add_implicit_resolver("tag:yaml.org,2002:float", re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)[eE][-+]?\d+$"),
                              list("-+0123456789."))


def read_yaml(text: str):
    return yaml.load(text, Loader=_Loader)


def merge(base: dict, override: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    if "name" in override and override["name"] != out.get("name") and isinstance(out.get(FREE_FORM), dict):
        out[FREE_FORM] = {}
    for key, value in override.items():
        where = f"{path}{key}"
        if key not in out and FREE_FORM not in where.split("."):
            raise KeyError(f"unknown config key '{where}' (not in {BASE_CONFIG.name})")
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value, f"{where}.")
        else:
            out[key] = value
    return out


def parse_overrides(pairs: list[str]) -> dict:
    """['train.epochs=5', 'optim.kwargs.lr=1e-3'] -> nested dict, values parsed as YAML."""
    out: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"override '{pair}' must look like key.path=value")
        dotted, raw = pair.split("=", 1)
        *parents, leaf = dotted.split(".")
        node = out
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = read_yaml(raw)
    return out


def load_config(path: Path | None, overrides: list[str] = (), smoke: bool = False) -> dict:
    cfg = read_yaml(BASE_CONFIG.read_text())
    if path is not None:
        cfg = merge(cfg, read_yaml(Path(path).read_text()) or {})
    if smoke:
        cfg = merge(cfg, SMOKE)
        cfg["paths"]["run_root"] = f"{cfg['paths']['run_root']}/_smoke"
    cfg = merge(cfg, parse_overrides(list(overrides)))
    resolve_preprocess(cfg)
    validate(cfg)
    return cfg


def resolve_preprocess(cfg: dict) -> None:
    """data.preprocess -> fill defaults and point data.root at data/sliced_<hash of the settings>."""
    p = cfg["data"]["preprocess"]
    if p is None:
        return
    if "source_dir" not in p:
        raise ValueError("data.preprocess needs `source_dir`")
    if p.get("gt_version") not in ("original", "corrected"):
        raise ValueError("data.preprocess.gt_version must be 'original' or 'corrected'")
    p = cfg["data"]["preprocess"] = PREPROCESS_DEFAULTS | p
    cfg["data"]["root"] = f"data/sliced_{hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:8]}"


def validate(cfg: dict) -> None:
    if not cfg["experiment"]:
        raise ValueError("config must set `experiment`")
    if len(cfg["data"]["class_names"]) != cfg["data"]["num_classes"]:
        raise ValueError("data.class_names must have data.num_classes entries")
    bad = [k for k in cfg["eval"]["classes"] if not 0 < k < cfg["data"]["num_classes"]]
    if bad:
        raise ValueError(f"eval.classes {bad} outside 1..num_classes-1")
    if cfg["train"]["epochs"] < 1:
        raise ValueError(f"train.epochs must be >= 1, got {cfg['train']['epochs']}")
    if cfg["train"]["select_metric"] not in ("val_dice_fg", "val_dice_legacy_fg"):
        raise ValueError(f"unknown train.select_metric {cfg['train']['select_metric']}")
    validate_component_names(cfg)


def validate_component_names(cfg: dict) -> None:
    import src.data  # noqa: F401
    import src.losses  # noqa: F401
    import src.models  # noqa: F401
    import src.optim  # noqa: F401
    from src.registry import available

    selected = [(kind, cfg[kind]["name"]) for kind in ("model", "loss", "optim", "scheduler")]
    selected += [("augment", a["name"]) for a in cfg["data"]["augment"]]
    for kind, name in selected:
        if name not in available(kind):
            raise ValueError(f"unknown {kind} '{name}', available: {available(kind)}")


def config_hash(cfg: dict) -> str:
    """Identity of a run's settings. `notes`, `wandb`, `paths`, `device` and `data.num_workers`
    don't change results, so they're excluded and a run stays resumable when they differ."""
    relevant = {k: v for k, v in cfg.items() if k not in IGNORED_BY_HASH}
    relevant["data"] = {k: v for k, v in cfg["data"].items() if k != "num_workers"}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:12]
