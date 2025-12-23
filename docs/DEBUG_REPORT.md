# Infinity AI V3 Unified — Debug/Fix/Enhance Pass

This pass focuses on making the repo **import-safe**, **script-runnable without editable install**, and **robust when optional deps (gymnasium) are missing**.

## Fixes applied

### 1) Import-safety when `gymnasium` / `gym` is missing
- `src/infinity_dual_hybrid/envs.py`
  - Prevents import-time crashes when gym isn’t installed.
  - `DelayedCueEnv` no longer hard-inherits `gym.Env` when gym is absent.
  - `make_env` / `make_envs` now raise a clear `ImportError` **only when called**.

### 2) `python -m infinity_dual_hybrid.train --test` now works in minimal environments
- `src/infinity_dual_hybrid/train.py`
  - Guards `gymnasium` registration imports.
  - Env registration is skipped cleanly if gymnasium is absent.
  - `--test` path no longer fails due to missing gymnasium.

### 3) Scripts run without `pip install -e .`
- `scripts/train_cartpole_baseline.py`
- `scripts/train_cartpole_miras.py`
- `scripts/train_cartpole_neocortex.py`
- `scripts/auto_tuner_dual_hybrid.py`
  - Adds a small bootstrap that injects `../src` into `sys.path`.

### 4) Smoke tests don’t hang when gymnasium isn’t installed
- `tests/test_v2_smoke.py`
  - Uses `pytest.importorskip("gymnasium")` so minimal CI / containers don’t stall.

### 5) Dependency hygiene
- `requirements.txt`
  - Adds `gymnasium[classic-control]` so CartPole + classic-control work out of the box.
- `pyproject.toml`
  - Adds optional extra: `.[env]`
  - Adds gymnasium to `.[all]`

## Verified in this environment
- `python -m infinity_dual_hybrid.train --test` passes with `PYTHONPATH=src`
- Unit tests:
  - `tests/test_dual_tier_miras_cartpole.py` ✅
  - `tests/test_memory_sanity.py` ✅
  - `tests/test_v2_smoke.py` skipped if gymnasium missing ✅

## Suggested next enhancements (not required for correctness)
- Add a tiny `make test` / `make smoke` wrapper.
- Add a `scripts/setup_colab.sh` that pins torch + gymnasium + optional mamba/faiss.
- Add a `--dry-run` mode to training CLIs that prints resolved config + device + deps.
