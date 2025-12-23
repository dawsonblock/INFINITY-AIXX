from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import platform as _platform
import subprocess as _subprocess
import sys as _sys

import numpy as np
import torch
from .utils import set_global_seed

from .config import TrainConfig, get_config_for_env
from .envs import make_envs
from .agent import build_agent
from .metrics.plots import save_reliability_diagram, save_metric_timeseries

from .ppo_trainer import PPOTrainer
from .trainers import PPOTrainerFast
from .logger import UnifiedLogger, LoggerConfig
from .run_registry import ensure_dirs, make_run_id, write_run_meta, append_run_index


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst



def _parse_bool_like(s: str):
    sl = s.strip().lower()
    if sl in ("true","1","yes","y","on"): return True
    if sl in ("false","0","no","n","off"): return False
    return None

def _apply_set_overrides(cfg_obj, pairs: list[str]):
    # Supports dot paths into dataclasses (TrainConfig) and nested dataclasses.
    # Example: agent.use_ltm=true
    for p in pairs:
        if "=" not in p:
            continue
        key, val = p.split("=", 1)
        key = key.strip()
        val = val.strip()
        parts = [x for x in key.split(".") if x]
        if not parts:
            continue

        # Coerce value
        b = _parse_bool_like(val)
        if b is not None:
            coerced = b
        else:
            try:
                if val.isdigit() or (val.startswith("-") and val[1:].isdigit()):
                    coerced = int(val)
                else:
                    coerced = float(val)
            except Exception:
                coerced = val

        obj = cfg_obj
        for name in parts[:-1]:
            if not hasattr(obj, name):
                raise AttributeError(f"Unknown config path: {key}")
            obj = getattr(obj, name)
        leaf = parts[-1]
        if not hasattr(obj, leaf):
            raise AttributeError(f"Unknown config path: {key}")
        setattr(obj, leaf, coerced)
    return cfg_obj
def _dc_from_dict(dc_type, data: Dict[str, Any]):
    # Recursively build dataclasses from nested dicts.
    kwargs = {}
    for field in dc_type.__dataclass_fields__.values():  # type: ignore[attr-defined]
        name = field.name
        if name not in data:
            continue
        v = data[name]
        if is_dataclass(field.type) and isinstance(v, dict):
            kwargs[name] = _dc_from_dict(field.type, v)
        else:
            kwargs[name] = v
    return dc_type(**kwargs)


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required for --config. Install with: pip install -e '.[dev]'"
        ) from e
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore
            return "mps"
        return "cpu"
    return device


def _git_rev() -> Optional[str]:
    try:
        out = _subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=_subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _success_threshold(env_id: str) -> Optional[float]:
    """Return task success threshold for time-to-threshold metric, or None."""
    if not env_id:
        return None
    env = env_id.strip()
    # Gym classic-control common thresholds
    if env.lower() == "cartpole-v1":
        return 475.0
    if env.lower() == "cartpole-v0":
        return 195.0
    return None


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Infinity Human training entrypoint")
    ap.add_argument("--env", "--env-id", dest="env_id", default=None)
    ap.add_argument("--config", type=str, default=None, help="YAML config file")
    ap.add_argument("--total-timesteps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--rollout-steps", type=int, default=None)
    ap.add_argument("--runs-dir", type=str, default="runs")
    ap.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    ap.add_argument("--set", dest="sets", action="append", default=[], help="Override config keys, e.g. --set agent.use_ltm=true")
    args = ap.parse_args(argv)

    # Base config
    cfg: TrainConfig = get_config_for_env(args.env_id or "CartPole-v1")

    # YAML override
    yaml_data: Dict[str, Any] = {}
    if args.config:
        yaml_data = _load_yaml(args.config)
        # Start from dataclass -> dict, then deep merge, then rebuild dataclass
        merged = _deep_update(asdict(cfg), yaml_data)
        cfg = _dc_from_dict(TrainConfig, merged)

# CLI overrides
    if args.env_id:
        cfg.env_id = args.env_id
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device:
        cfg.device = args.device
    if args.total_timesteps is not None:
        cfg.ppo.total_timesteps = args.total_timesteps
    if args.n_envs is not None:
        cfg.ppo.n_envs = args.n_envs
    if args.rollout_steps is not None:
        cfg.ppo.rollout_steps = args.rollout_steps
    if args.sets:
        _apply_set_overrides(cfg, args.sets)


    device = _resolve_device(cfg.device)

    # Seeding
    if cfg.seed is not None:
        set_global_seed(cfg.seed, deterministic=getattr(cfg.ppo, 'deterministic', False))


    # run id
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    paths = ensure_dirs(Path("."), run_id)
    # allow overriding root dirs
    if args.runs_dir != "runs" or args.checkpoints_dir != "checkpoints":
        root = Path(".")
        paths = {
            "runs_dir": Path(args.runs_dir),
            "checkpoints_dir": Path(args.checkpoints_dir),
            "run_dir": Path(args.runs_dir) / run_id,
            "ckpt_dir": Path(args.checkpoints_dir) / run_id,
        }
        paths["run_dir"].mkdir(parents=True, exist_ok=True)
        paths["ckpt_dir"].mkdir(parents=True, exist_ok=True)
    run_dir = paths["run_dir"]
    ece_steps = []
    ece_r_series = []
    ece_w_series = []

    ckpt_dir = paths["ckpt_dir"]

    # Persist config + metadata
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    meta = {
        "run_id": run_id,
        "env_id": cfg.env_id,
        "seed": cfg.seed,
        "device": device,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "argv": list(argv or []),
        "python": _sys.version.replace("\n", " "),
        "platform": _platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "numpy": getattr(np, "__version__", None),
        "git_rev": _git_rev(),
        "config_path": str(Path(args.config).resolve()) if args.config else None,
        "set_overrides": list(args.sets or []),
    }
    (run_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Build
    # Optional multitask configuration (Phase-3)
    yaml_mt = {}
    try:
        yaml_mt = dict(yaml_data.get("multitask", {}) if isinstance(yaml_data, dict) else {})
    except Exception:
        yaml_mt = {}

    mt_enabled = bool(yaml_mt.get("enabled", False))
    mt_tasks = list(yaml_mt.get("tasks", []) or [])
    mt_router = str(yaml_mt.get("router", "round_robin"))
    mt_segment_steps = int(yaml_mt.get("segment_steps", 10000))
    mt_eval_every_segments = int(yaml_mt.get("eval_every_segments", 1))
    mt_eval_episodes = int(yaml_mt.get("eval_episodes", 25))
    mt_seeds_eval = list(yaml_mt.get("seeds_eval", [0, 1, 2, 3, 4]) or [0, 1, 2, 3, 4])

    agent = build_agent(cfg.agent).to(device)  # type: ignore[attr-defined]
    # Optional torch.compile for speed (torch>=2.0). Best effort, falls back silently.
    if getattr(cfg.ppo, 'torch_compile', False):
        try:
            agent = torch.compile(agent, mode=getattr(cfg.ppo, 'torch_compile_mode', 'reduce-overhead'), fullgraph=getattr(cfg.ppo, 'torch_compile_fullgraph', False))
        except Exception:
            pass

    trainer_kind = None
    try:
        trainer_kind = (cfg.ppo.get("trainer", {}) or {}).get("kind", None)
    except Exception:
        trainer_kind = None
    if trainer_kind == "fast":
        trainer = PPOTrainerFast(agent, cfg.ppo, device=device, seed=cfg.seed)
    elif trainer_kind == "optimized":
        # Same trainer, but expect cfg.ppo.amp.enabled=true and other advanced toggles
        trainer = PPOTrainer(agent, cfg.ppo, device=device, seed=cfg.seed)
    else:
        trainer = PPOTrainer(agent, cfg.ppo, device=device, seed=cfg.seed)

    logger = UnifiedLogger(LoggerConfig(
        log_dir=str(run_dir),
        experiment_name="train",
        use_console=True,
        use_csv=True,
        use_jsonl=True,
        use_tensorboard=True,
    ))

    steps_per_iter = cfg.ppo.n_envs * cfg.ppo.rollout_steps
    total_timesteps = int(cfg.ppo.total_timesteps)

    # --- helpers ---
    import csv as _csv

    def _append_eval_task_row(task_id: str, global_step: int, return_mean: float, return_std: float, length_mean: float) -> None:
        p = run_dir / "eval_tasks.csv"
        exists = p.exists()
        with p.open("a", encoding="utf-8", newline="") as f:
            w = _csv.writer(f)
            if not exists:
                w.writerow(["global_step", "task_id", "return_mean", "return_std", "len_mean"])
            w.writerow([global_step, task_id, return_mean, return_std, length_mean])

    def _eval_task(env_id: str, *, episodes: int, seeds: list[int]) -> tuple[float, float, float]:
        # Evaluate by running `episodes` episodes on each seed and aggregating.
        rewards: list[float] = []
        lengths: list[int] = []

        # We use a single env instance per seed for determinism best-effort.
        for s in seeds:
            env = make_envs(env_id, num_envs=1, cfg=cfg)[0]
            try:
                # gymnasium: reset(seed=)
                env.reset(seed=int(s))
            except Exception:
                pass

            for _ in range(int(episodes)):
                obs = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
                done = False
                trunc = False
                ep_r = 0.0
                ep_len = 0
                while not (done or trunc):
                    act, _ = agent.get_action(obs, deterministic=True)  # type: ignore[attr-defined]
                    step_out = env.step(act)
                    if len(step_out) == 5:
                        obs, r, done, trunc, _info = step_out
                    else:
                        obs, r, done, _info = step_out
                        trunc = False
                    ep_r += float(r)
                    ep_len += 1
                rewards.append(ep_r)
                lengths.append(ep_len)

            try:
                env.close()
            except Exception:
                pass

        if not rewards:
            return -1e18, 0.0, 0.0
        rm = float(np.mean(rewards))
        rs = float(np.std(rewards))
        lm = float(np.mean(lengths)) if lengths else 0.0
        return rm, rs, lm

    best_eval = -1e18

    if not mt_enabled:
        # ---- single-task (existing behavior) ----
        n_envs = int(getattr(cfg.ppo, 'n_envs', getattr(cfg.ppo, 'num_envs', 1)))
        use_vec = bool(getattr(cfg.ppo, 'vector_env', False))
        if use_vec and n_envs > 1:
            envs = make_vector_env(cfg.env_id, num_envs=n_envs, cfg=cfg, async_vector=bool(getattr(cfg.ppo, 'vector_async', False)), seed=cfg.seed)
        else:
            envs = make_envs(cfg.env_id, num_envs=n_envs, cfg=cfg)

        iters = max(1, int(int(total_timesteps) // int(steps_per_iter)))

        for it in range(iters):
            rollouts = trainer.collect_rollouts(envs, steps=cfg.ppo.rollout_steps)
            metrics = trainer.train_step(rollouts)
            metrics["iteration"] = trainer.iteration
            metrics["timesteps"] = (it + 1) * steps_per_iter            # Calibration artifacts + timeseries
            if isinstance(getattr(trainer, "last_wm_calibration", None), dict):
                calib = trainer.last_wm_calibration
                try:
                    cal_dir = run_dir / "calibration"
                    plot_dir = run_dir / "plots"
                    cal_dir.mkdir(parents=True, exist_ok=True)
                    plot_dir.mkdir(parents=True, exist_ok=True)

                    # write raw bin JSON
                    (cal_dir / "wm_reward_reliability.json").write_text(json.dumps(calib.get("reward", {}), indent=2))
                    (cal_dir / "wm_latent_reliability.json").write_text(json.dumps(calib.get("latent", {}), indent=2))

                    # render reliability diagrams if possible
                    from .metrics.calibration import CalibrationBins
                    import numpy as _np

                    r = calib.get("reward", {})
                    if r.get("edges") is not None:
                        bins_r = CalibrationBins(
                            _np.array(r.get("edges", []), dtype=_np.float64),
                            _np.array(r.get("bin_conf", []), dtype=_np.float64),
                            _np.array(r.get("bin_acc", []), dtype=_np.float64),
                            _np.array(r.get("bin_count", []), dtype=_np.int64),
                        )
                        save_reliability_diagram(
                            str(plot_dir / "wm_reward_reliability.png"),
                            bins_r,
                            title="World Model Reward Reliability",
                        )

                    l = calib.get("latent", {})
                    if l.get("edges") is not None:
                        bins_w = CalibrationBins(
                            _np.array(l.get("edges", []), dtype=_np.float64),
                            _np.array(l.get("bin_conf", []), dtype=_np.float64),
                            _np.array(l.get("bin_acc", []), dtype=_np.float64),
                            _np.array(l.get("bin_count", []), dtype=_np.int64),
                        )
                        save_reliability_diagram(
                            str(plot_dir / "wm_latent_reliability.png"),
                            bins_w,
                            title="World Model Latent Reliability",
                        )
                except Exception:
                    pass

            # ECE timeseries

            if "wm/calib_ece_r" in metrics:
                ece_steps.append(float(metrics.get("timesteps", 0)))
                ece_r_series.append(float(metrics.get("wm/calib_ece_r", 0.0)))
                ece_w_series.append(float(metrics.get("wm/calib_ece_w", 0.0)))

            if (it + 1) % max(1, int(cfg.ppo.eval_interval)) == 0:
                eval_stats = trainer.evaluate(envs, num_episodes=cfg.ppo.eval_episodes, deterministic=True)

                return_mean = float(eval_stats.get("eval_mean_reward", eval_stats.get("return_mean", -1e18)))
                return_std = float(eval_stats.get("eval_std_reward", eval_stats.get("return_std", 0.0)))
                length_mean = float(eval_stats.get("eval_mean_length", eval_stats.get("length_mean", 0.0)))

                metrics["eval/return_mean"] = return_mean
                metrics["eval/return_std"] = return_std
                metrics["eval/length_mean"] = length_mean

                thr = _success_threshold(cfg.env_id)
                if thr is not None and return_mean >= thr and "eval/time_to_threshold" not in metrics:
                    metrics["eval/time_to_threshold"] = int(metrics["timesteps"])

                if return_mean > best_eval:
                    best_eval = return_mean
                    trainer.save(str(ckpt_dir / "best.pt"))

            logger.log(metrics)

            if (it + 1) % cfg.save_interval == 0:
                trainer.save(str(ckpt_dir / "latest.pt"))

        trainer.save(str(ckpt_dir / "latest.pt"))
        logger.close()

        # Save ECE-over-time plot (reward + latent), if available
        try:
            if len(ece_steps) > 1:
                plot_dir = run_dir / "plots"
                plot_dir.mkdir(parents=True, exist_ok=True)
                save_metric_timeseries(
                    str(plot_dir / "wm_ece_over_time.png"),
                    ece_steps,
                    {"wm/calib_ece_r": ece_r_series, "wm/calib_ece_w": ece_w_series},
                    title="World Model Calibration (ECE) Over Time",
                )
        except Exception:
            pass

        for env in envs:
            try:
                env.close()
            except Exception:
                pass

    else:
        # ---- multitask (Phase-3) ----
        from .tasks.registry import build_default_registry
        from .tasks.router import RoundRobinRouter, CompetenceRouter, SpacedRepetitionRouter
        from .tasks.rollout_mux import RolloutMux
        from .metrics.transfer import compute_transfer_metrics, write_transfer_artifacts

        registry = build_default_registry()
        if not mt_tasks:
            mt_tasks = list(registry.keys())

        for tid in mt_tasks:
            if tid not in registry:
                raise ValueError(f"Unknown task_id in config: {tid}. Known: {sorted(registry.keys())}")

        if mt_router == "competence":
            router = CompetenceRouter(mt_tasks)
        elif mt_router in ("spaced", "spaced_repetition"):
            router = SpacedRepetitionRouter(mt_tasks)
        else:
            router = RoundRobinRouter(mt_tasks)

        mux = RolloutMux(registry=registry, n_envs=cfg.ppo.n_envs, cfg=cfg)

        history: Dict[str, dict] = {tid: {"threshold": registry[tid].success_threshold, "recent_return_mean": -1e18} for tid in mt_tasks}
        time_to_threshold: Dict[str, Optional[int]] = {tid: None for tid in mt_tasks}

        global_step = 0
        segments = max(1, int(np.ceil(total_timesteps / float(mt_segment_steps))))

        for seg in range(segments):
            task_id = router.next_task(step=global_step, history=history)
            envs = mux.switch(task_id)

            target_steps = min(mt_segment_steps, total_timesteps - global_step)
            seg_steps = 0

            while seg_steps < target_steps:
                rollouts = trainer.collect_rollouts(envs, steps=cfg.ppo.rollout_steps)
                metrics = trainer.train_step(rollouts)
                trainer.iteration = int(trainer.iteration)  # make type-checkers happy
                global_step += steps_per_iter
                seg_steps += steps_per_iter

                metrics["iteration"] = trainer.iteration
                metrics["timesteps"] = global_step
                metrics["task_id"] = task_id
                logger.log(metrics)

                # always keep latest current
                trainer.save(str(ckpt_dir / "latest.pt"))

            # eval sweep
            if mt_eval_every_segments > 0 and ((seg + 1) % mt_eval_every_segments == 0):
                per_task_returns = []
                for tid in mt_tasks:
                    spec = registry[tid]
                    rm, rs, lm = _eval_task(spec.env_id, episodes=mt_eval_episodes, seeds=[int(s) for s in mt_seeds_eval])
                    _append_eval_task_row(tid, global_step, rm, rs, lm)
                    history[tid]["recent_return_mean"] = rm

                    # time-to-threshold tracking
                    thr = spec.success_threshold
                    if thr is not None and time_to_threshold.get(tid) is None and rm >= float(thr):
                        time_to_threshold[tid] = int(global_step)

                    per_task_returns.append(rm)

                # define overall success as mean return across tasks
                avg_return = float(np.mean(per_task_returns)) if per_task_returns else -1e18
                avg_std = float(np.std(per_task_returns)) if per_task_returns else 0.0
                logger.log({
                    "timesteps": global_step,
                    "segment": seg + 1,
                    "eval/return_mean": avg_return,
                    "eval/return_std": avg_std,
                })

                if avg_return > best_eval:
                    best_eval = avg_return
                    trainer.save(str(ckpt_dir / "best.pt"))

        logger.close()

        # Save ECE-over-time plot (reward + latent), if available
        try:
            if len(ece_steps) > 1:
                plot_dir = run_dir / "plots"
                plot_dir.mkdir(parents=True, exist_ok=True)
                save_metric_timeseries(
                    str(plot_dir / "wm_ece_over_time.png"),
                    ece_steps,
                    {"wm/calib_ece_r": ece_r_series, "wm/calib_ece_w": ece_w_series},
                    title="World Model Calibration (ECE) Over Time",
                )
        except Exception:
            pass

        # write transfer artifacts
        thresholds = {tid: registry[tid].success_threshold for tid in mt_tasks}
        summary = compute_transfer_metrics(run_dir, thresholds=thresholds, scratch_time_to_threshold=None)
        write_transfer_artifacts(run_dir, summary)


    # append a lightweight index record
    append_run_index(Path(args.runs_dir), {
        "run_id": run_id,
        "ended_at": datetime.utcnow().isoformat(),
        "checkpoints": str(ckpt_dir),
    })
    print(f"Done. run_id={run_id} checkpoints={ckpt_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())