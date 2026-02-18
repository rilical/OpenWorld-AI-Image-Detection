"""Configuration helpers used across scripts.

The helpers in this module intentionally stay small and explicit:
- parse YAML into dictionaries,
- merge CLI overrides with dotted keys,
- expand environment variables in resolved config dictionaries.
"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file and return a dictionary.

    Examples
    --------
    >>> cfg = load_yaml('tmp.yaml')
    """
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return cfg


def _parse_scalar(value: str) -> Any:
    """Parse a CLI scalar without external dependencies.

    Supports booleans, ints, floats, JSON-like literals and plain strings.
    """
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False

    # Keep dotted numeric detection simple and deterministic.
    try:
        if "." in value or "e" in lower:
            return float(value)
        return int(value)
    except ValueError:
        pass

    # Fall back to json parser for values like [1,2], {"a":1}, null, etc.
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _set_nested(target: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested dict key from a dotted path in-place."""
    parts = dotted_key.split('.')
    if not parts:
        raise ValueError(f"Invalid dotted key: '{dotted_key}'")

    current: Dict[str, Any] = target
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        child = current[part]
        if not isinstance(child, dict):
            raise ValueError(
                f"Cannot apply override '{dotted_key}': '{part}' is not a mapping"
            )
        current = child
    current[parts[-1]] = value


def parse_overrides(items: Iterable[str]) -> Dict[str, Any]:
    """Parse a list like ``["a=1", "b.c=true"]`` into nested dict.

    Examples
    --------
    >>> parse_overrides(["train.lr=0.01", "head.hidden=4", "enabled=true"])
    {'train': {'lr': 0.01}, 'head': {'hidden': 4}, 'enabled': True}
    """
    parsed: Dict[str, Any] = {}
    for item in items:
        if '=' not in item:
            raise ValueError(f"Invalid override '{item}'. Expected format: key=value")

        key, raw = item.split('=', 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override '{item}'. Key cannot be empty")

        _set_nested(parsed, key, _parse_scalar(raw.strip()))

    return parsed


def deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``updates`` into ``base`` and return a new dict."""
    merged = deepcopy(base)
    for key, value in updates.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = deep_update(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve environment variables and ``~`` in string leaves.

    Examples
    --------
    >>> resolve_config({'path': '${HOME}/tmp'})
    {'path': '/home/user/tmp'}
    """
    import os

    def _resolve(value: Any) -> Any:
        if isinstance(value, str):
            return os.path.expanduser(os.path.expandvars(value))
        if isinstance(value, dict):
            return {k: _resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_resolve(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_resolve(v) for v in value)
        return value

    return _resolve(cfg)


def merge_cli_overrides(cfg_dict: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """Apply CLI overrides to a config dict.

    This keeps old behavior while routing through the new parsing/merge helpers.
    """
    if not overrides:
        return cfg_dict
    return deep_update(cfg_dict, parse_overrides(overrides))


def parse_kwargs_to_dict(items: Iterable[str]) -> Dict[str, Any]:
    """Backward-compatible alias for ``parse_overrides``."""
    return parse_overrides(items)


def to_namespace(cfg_dict: Dict[str, Any]) -> SimpleNamespace:
    """Convert a nested mapping to a recursive :class:`SimpleNamespace`."""

    def _coerce(value: Any) -> Any:
        if isinstance(value, dict):
            return SimpleNamespace(**{k: _coerce(v) for k, v in value.items()})
        return value

    return SimpleNamespace(**{k: _coerce(v) for k, v in cfg_dict.items()})


def namespace_to_dict(ns: Any) -> Dict[str, Any]:
    """Convert a namespace tree back to primitives (dict/list)."""
    if isinstance(ns, dict):
        return {k: namespace_to_dict(v) for k, v in ns.items()}
    if isinstance(ns, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    if isinstance(ns, list):
        return [namespace_to_dict(v) for v in ns]
    return ns


def add_config_args(parser: ArgumentParser) -> ArgumentParser:
    """Register common config CLI flags used across scripts."""
    parser.add_argument('--config', required=True)
    parser.add_argument('--opts', nargs='*', default=[], help='Config overrides as key=value')
    return parser


def write_yaml(path: str, obj: Any) -> None:
    """Write ``obj`` as YAML (legacy helper used by scripts)."""
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(obj, f)

