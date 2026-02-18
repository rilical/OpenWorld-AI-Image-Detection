"""JSON artifact helpers used by scripts and training loops."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


def write_json(path: str, obj: Any) -> None:
    """Serialize ``obj`` as JSON to ``path`` and create parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def read_json(path: str) -> Any:
    """Read JSON from ``path`` and return Python object."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_yaml(path: str, obj: Any) -> None:
    """Legacy compatibility helper for YAML artifact writes.

    Kept here because a number of existing scripts still import this symbol.
    """
    import yaml

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8') as f:
        yaml.safe_dump(obj, f)


@dataclass
class JsonlLogger:
    """Append-only JSONL writer for step/epoch logs.

    Examples
    --------
    >>> logger = JsonlLogger('tmp.jsonl')
    >>> logger.log({'step': 1, 'loss': 0.1})
    """
    path: str

    def __post_init__(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        """Append one JSON record as a single line."""
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record))
            f.write('\n')

