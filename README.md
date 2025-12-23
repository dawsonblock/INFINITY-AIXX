# Infinity Human (Infinity Dual Hybrid)

This repo is a clean, runnable core build of the Infinity Dual Hybrid RL + cognition stack.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

Optional extras:
- FAISS LTM: `pip install -e ".[faiss]"`
- TensorBoard: included via `tensorboard`
- Mamba vendor code is included under `vendor/` (no external install required).

## Quick start (training)

CartPole baseline:

```bash
python -m infinity_dual_hybrid.train_cli --env CartPole-v1 --total-timesteps 100000
```

With a YAML config:

```bash
python -m infinity_dual_hybrid.train_cli --config configs/cartpole_baseline.yaml
```

## Evaluation

```bash
python -m infinity_dual_hybrid.eval --checkpoint checkpoints/latest.pt --env CartPole-v1 --episodes 25
```

## Runs, logs, checkpoints

- `runs/<run_id>/` contains:
  - `config.json`
  - `meta.json` (commit hash if available, device, seed, etc.)
  - TensorBoard events (if enabled)

- `checkpoints/<run_id>/` contains:
  - `latest.pt`
  - `best.pt` (if enabled)

## Development

```bash
pytest
ruff check .
black .
mypy src
```


## Auto-evaluate best checkpoints across runs

```bash
# Evaluate each run's best checkpoint and rank them
infinity-eval-runs --runs_dir runs --checkpoints_dir checkpoints --env CartPole-v1 --episodes 25

# Re-evaluate even if eval_best.json already exists
infinity-eval-runs --force
```

Outputs:
- `runs/eval_rank.csv`
- `runs/eval_rank.md`
- `runs/<run_id>/eval_best.json`


## Multitask Phase-3

Run classic-control multitask training (task router + transfer metrics):

```bash
infinity-train --config configs/multitask_gym3.yaml
# fresh-eval ranking still works:
infinity-eval-runs --runs_dir runs --checkpoints_dir checkpoints --env CartPole-v1 --episodes 25
# Phase-3 artifacts per run:
#   runs/<run_id>/eval_tasks.csv
#   runs/<run_id>/transfer.json
#   runs/<run_id>/transfer.md
```

## Best build composition (this archive)

This "BEST" package is based on **Infinity-human-FULL-BUILD-Phase6-adapter-ece** as the primary, feature-complete codebase.

It additionally includes:
- `extras/Infinity_opt_reference/`: the older optimized/minimal trainer reference build.
- `extras/Infinity_lite_reference/`: the older lite/demo build.
- `extras/Infinity_legacy_reference/`: the older legacy build (pre-Phase6) for comparison.

### Optional fast trainer

You can enable a lean (experimental) PPO trainer for quick iteration on small Gymnasium tasks:

```yaml
ppo:
  trainer:
    kind: fast
```

Default remains the standard trainer.


## Advanced optimized run (AMP + adapters + calibration)

```bash
pip install -e .
infinity-train --config configs/cartpole_advanced_optimized.yaml
```
