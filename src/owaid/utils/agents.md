# src/owaid/utils/agents.md

Scope: config loading, run directory creation, logging, seeding, path/env helpers.

Invariants:
- All CLI scripts call:
  - set_seed(cfg.seed)
  - make_run_dir(cfg) -> run_dir
  - write resolved config.yaml to run_dir
  - write meta.json to run_dir (git hash if available)

Required modules:
- config.py:
  - load_yaml(path) -> dict
  - merge_cli_overrides(cfg_dict, overrides) -> dict
  - to_namespace/dataclass helpers (keep simple)
- paths.py:
  - require_env(name) -> str (throws clear error)
  - resolve_path(base, rel) -> str
- seed.py:
  - set_seed(seed, deterministic: bool)
- logging.py:
  - JsonlLogger(path).log(dict)
  - write_json(path, obj)
  - read_json(path) -> obj

Do NOT:
- introduce heavy dependency frameworks
- hide side effects (run_dir creation must be explicit)
