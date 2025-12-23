from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Any, Dict

try:
    import gymnasium as gym  # type: ignore
    _GYMNASIUM = True
except Exception:  # pragma: no cover
    try:
        import gym  # type: ignore
        _GYMNASIUM = False
    except Exception:  # pragma: no cover
        gym = None
        _GYMNASIUM = False


@dataclass(frozen=True)
class TaskSpec:
    """
    A task specification for multitask training.

    task_id: stable identifier used in logs/configs.
    env_id: gym/gymnasium environment id.
    success_threshold: return threshold used for time-to-threshold.
    max_steps: optional cap for episode length (best-effort; depends on env).
    """
    task_id: str
    env_id: str
    success_threshold: Optional[float] = None
    max_steps: Optional[int] = None
    eval_episodes: int = 25

    def make_env(self, seed: Optional[int] = None) -> Any:
        if gym is None:
            raise ModuleNotFoundError(
                "Neither 'gymnasium' nor 'gym' is installed. Install one of them, e.g. `pip install gymnasium`."
            )
        env = gym.make(self.env_id)
        # best-effort seeding
        try:
            if _GYMNASIUM:
                env.reset(seed=seed)
            else:
                if seed is not None:
                    env.seed(seed)
        except Exception:
            pass
        return env
