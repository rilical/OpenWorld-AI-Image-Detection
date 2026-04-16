"""Path bootstrap utilities for running scripts from the repository root."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def bootstrap_repo_source() -> None:
    """Insert ``src`` into ``sys.path`` for local runs.

    This keeps every CLI entrypoint runnable as ``python scripts/<name>.py`` from
    the repository root without requiring prior installation.
    """
    root = Path(__file__).resolve().parents[1]
    src_dir = root / "src"
    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)

    # Point HuggingFace cache to repo-local directory so all scripts share
    # the same cache and never fall back to ~/.cache/huggingface.
    hf_cache = root / ".cache" / "huggingface"
    os.environ.setdefault("HF_HOME", str(hf_cache))
