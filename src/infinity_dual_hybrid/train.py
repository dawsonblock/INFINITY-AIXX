"""
train.py (compat)

This file remains for backwards compatibility with older entrypoints.

Use:
    python -m infinity_dual_hybrid.train_cli ...
"""
from __future__ import annotations

from .train_cli import main  # re-export


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
