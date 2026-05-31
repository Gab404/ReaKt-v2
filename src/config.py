"""
src/config.py
=============
Lightweight configuration system.

Loads YAML files into a nested Config object that supports both
dict-style (cfg["key"]) and attribute-style (cfg.key) access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import yaml


class Config:
    """
    Thin wrapper around a nested dict that supports:
      - cfg.section.key   (attribute access, arbitrarily deep)
      - cfg["section"]["key"]  (item access)
      - cfg.get("key", default)
      - Config.from_yaml(path)
      - cfg.to_dict()
      - cfg.update(overrides_dict)  (shallow merge at top level)
    """

    def __init__(self, data: dict):
        # Store raw dict; wrap nested dicts recursively
        object.__setattr__(self, "_data", {})
        for k, v in data.items():
            self._data[k] = Config(v) if isinstance(v, dict) else v

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(data)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(d)

    # ── Access ────────────────────────────────────────────────────────────────

    def __getattr__(self, key: str) -> Any:
        data = object.__getattribute__(self, "_data")
        if key not in data:
            raise AttributeError(f"Config has no key '{key}'")
        return data[key]

    def __setattr__(self, key: str, value: Any):
        if key == "_data":
            object.__setattr__(self, "_data", value)
        else:
            self._data[key] = Config(value) if isinstance(value, dict) else value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any):
        self._data[key] = Config(value) if isinstance(value, dict) else value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        out = {}
        for k, v in self._data.items():
            out[k] = v.to_dict() if isinstance(v, Config) else v
        return out

    def update(self, overrides: dict):
        """Shallow-merge overrides at the top level."""
        for k, v in overrides.items():
            self._data[k] = Config(v) if isinstance(v, dict) else v

    def __repr__(self) -> str:
        return f"Config({self.to_dict()!r})"
