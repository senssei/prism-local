"""
prism.paths: Where Prism looks for and stores models.

Search order: $PRISM_MODEL_DIRS (os.pathsep-separated), ~/.prism/models, then the
Foundry Local cache directories. `prism pull` writes to the first $PRISM_MODEL_DIRS
entry, or ~/.prism/models when the variable is unset.
"""

import os
from pathlib import Path
from typing import List

ENV_MODEL_DIRS = "PRISM_MODEL_DIRS"


def default_model_dir() -> Path:
    """Destination for `prism pull`."""
    configured = _env_model_dirs()
    return Path(configured[0]) if configured else state_dir() / "models"


def state_dir() -> Path:
    """Directory where runtime state and locks are kept: ~/.prism (or $PRISM_STATE_DIR)."""
    override = os.environ.get("PRISM_STATE_DIR")
    if override:
        return Path(os.path.expanduser(override))
    return Path.home() / ".prism"


def model_search_paths() -> List[str]:
    """Existing directories to scan for ONNX models, most preferred first, de-duplicated."""
    home = Path.home()
    candidates = [
        *_env_model_dirs(),
        str(home / ".prism" / "models"),
        str(home / ".foundry" / "cache" / "models" / "Microsoft"),
        str(home / ".foundry" / "cache" / "models"),
    ]
    seen = set()
    result: List[str] = []
    for path in candidates:
        path = os.path.abspath(os.path.expanduser(path))
        if path not in seen and os.path.isdir(path):
            seen.add(path)
            result.append(path)
    return result


def _env_model_dirs() -> List[str]:
    raw = os.environ.get(ENV_MODEL_DIRS, "")
    return [os.path.expanduser(p) for p in raw.split(os.pathsep) if p.strip()]
