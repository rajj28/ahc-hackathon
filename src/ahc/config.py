"""Config loading. Every tunable lives in configs/default.yaml - no magic numbers in code.

    from ahc.config import load
    cfg = load()                    # or load("configs/experiment_a.yaml")
    cfg.segment.hi                  # dotted access
    cfg.classes                     # list of the 11; index 11 is "normal"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

NORMAL_IDX = 11
NORMAL = "normal"


class Config(dict):
    """Recursive mapping with attribute access, while retaining normal dict semantics."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return Config({key: _wrap(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


def load(path: str = "configs/default.yaml", overrides: list[str] | None = None):
    """-> dotted-access config object. Supports overrides via `--set segment.hi=0.6` style
    key paths so the threshold sweep does not need a YAML file per trial.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"configuration {path!r} must contain a mapping")
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"override must be dotted.path=value, got {override!r}")
        dotted_path, raw_value = override.split("=", 1)
        keys = dotted_path.split(".")
        target = data
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                raise KeyError(f"unknown configuration path: {dotted_path}")
            target = target[key]
        if keys[-1] not in target:
            raise KeyError(f"unknown configuration path: {dotted_path}")
        target[keys[-1]] = yaml.safe_load(raw_value)
    return _wrap(data)


def class_index(name: str) -> int:
    """Class name -> index. `normal` -> 11. Unknown name raises: a silent fallback here would
    produce a rejected submission much later.
    """
    if name == NORMAL:
        return NORMAL_IDX
    classes = load().classes
    try:
        return classes.index(name)
    except ValueError as exc:
        raise ValueError(f"unknown AHC class: {name!r}") from exc
