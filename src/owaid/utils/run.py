"""Run directory and experiment metadata helpers.

Public helpers are intentionally minimal and side-effect explicit so script
entrypoints can share the same artifact contract.
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import torch

from .config import resolve_config
from .paths import ensure_dir, resolve_path
from .logging import write_yaml


def _get_git_commit() -> str:
    """Read short git commit hash when repository metadata is available."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except Exception:
        return "unknown"


def make_run_dir(output_root: str, run_name: Optional[str]) -> str:
    """Create and return an absolute run directory.

    Parameters
    ----------
    output_root:
        Base directory for experiments.
    run_name:
        Optional run identifier. If omitted, defaults to
        ``YYYYMMDD_HHMMSS_<short_random>``.
    """
    root = resolve_path(output_root)
    safe_run_name = run_name if run_name else f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
    run_dir = resolve_path(root, safe_run_name)
    ensure_dir(run_dir)
    return run_dir


def write_meta(run_dir: str, cfg: Dict[str, Any]) -> None:
    """Write ``meta.json`` under ``run_dir``.

    Fields are fixed to the repository artifact contract and include a best-effort
    git commit when available.
    """
    meta = {
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "git_commit": _get_git_commit(),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if cfg and isinstance(cfg, dict):
        meta.update(
            {
                "seed": cfg.get("seed"),
                "deterministic": cfg.get("deterministic"),
                "device": cfg.get("device"),
            }
        )
    write_path = Path(run_dir) / "meta.json"
    write_path.parent.mkdir(parents=True, exist_ok=True)
    with write_path.open("w", encoding="utf-8") as f:
        import json

        json.dump(meta, f, indent=2)


def save_resolved_config(run_dir: str, cfg_dict: Dict[str, Any]) -> None:
    """Resolve paths/env vars in config and persist it as ``config.yaml``."""
    resolved = resolve_config(cfg_dict)
    write_yaml(str(Path(run_dir) / "config.yaml"), resolved)


if __name__ == "__main__":
    print(make_run_dir("outputs/runs", None))
