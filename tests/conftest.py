from __future__ import annotations

from pathlib import Path
import sys

import pytest


@pytest.fixture()
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Ensure local src/ and vendor/ are importable without external plugins.
_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src", _ROOT / "vendor"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
