## 2.1.3
- Storage-backed rollouts: keep rollout tensors on-device to avoid CPU copies; optional rollouts_on_cpu fallback.
- New PinnedStager to eliminate per-step CUDA tensor allocations for observation staging.
- Torch-based minibatch shuffling on-device.

## 2.1.1
- Fixed unreachable stray `set_task` block in `agent.py`.
- Added `infinity_dual_hybrid.utils` with fast observation tensor conversion + deterministic seeding helpers.
- Optimized rollout obs conversion to reduce per-step allocations.
- Unified seeding and added optional `torch.compile` switch in config/CLI.

## v5 (2025-12-19)
- Added best-checkpoint resolver + ranked reports by best eval metric.
- Added `infinity-best` command.

# Changelog

## Unreleased
- Added reproducible run structure (runs/, checkpoints/)
- Added evaluation entrypoint
- Added tests + CI smoke checks
- Added ruff/black/mypy tooling and a Makefile

## 2.0.1
- Initial Infinity Human release (core RL + cognition modules)
