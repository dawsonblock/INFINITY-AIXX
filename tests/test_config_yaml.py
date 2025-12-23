from __future__ import annotations

from pathlib import Path

import json

import yaml  # type: ignore


def test_example_configs_parse(repo_root: Path) -> None:
    # repo_root fixture is provided by conftest.py
    cfg_dir = repo_root / "configs"
    for name in ["default.yaml", "cartpole_baseline.yaml", "cartpole_miras.yaml", "cartpole_neocortex.yaml"]:
        data = yaml.safe_load((cfg_dir / name).read_text(encoding="utf-8"))
        assert isinstance(data, dict)
