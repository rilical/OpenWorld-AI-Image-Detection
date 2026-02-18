"""Path and environment helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict


def require_env(name: str) -> str:
    """Return a required environment variable or raise an actionable error."""
    value = os.environ.get(name)
    if value:
        return value
    raise EnvironmentError(
        f"Missing required environment variable '{name}'. "
        f"Set it before running this command, e.g. `export {name}=/absolute/path/to/data`."
    )


def ensure_dir(path: str) -> str:
    """Create directory and return its string path."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def resolve_path(*parts: str) -> str:
    """Join path parts, expand env vars/user and normalize to an absolute path."""
    if not parts:
        raise ValueError("resolve_path requires at least one path component")
    expanded = [os.path.expanduser(os.path.expandvars(str(part))) for part in parts if part is not None]
    return str(Path(os.path.join(*expanded)).resolve())


def make_run_dir(cfg: Dict[str, Any] | Any) -> str:
    """Backward-compatible run-dir helper used by existing CLI scripts.

    Reads ``cfg.output.root`` and ``cfg.output.run_name`` and delegates to
    :func:`owaid.utils.run.make_run_dir`.
    """
    from .run import make_run_dir as _make_run_dir

    if isinstance(cfg, dict):
        output_cfg = cfg.get("output")
    else:
        output_cfg = vars(cfg).get("output")

    if not isinstance(output_cfg, dict):
        output_cfg = {}

    output_root = output_cfg.get("root", "outputs/runs")
    run_name = output_cfg.get("run_name")

    run_dir = _make_run_dir(output_root, run_name)
    output_cfg["run_name"] = output_cfg.get("run_name") or Path(run_dir).name
    output_cfg["run_dir"] = run_dir

    # Keep old object-based behavior for callers that pass mutable namespaces.
    if not isinstance(cfg, dict) and hasattr(cfg, "output"):
        try:
            cfg.output = output_cfg
        except Exception:
            pass
    elif isinstance(cfg, dict):
        cfg["output"] = output_cfg

    return run_dir
