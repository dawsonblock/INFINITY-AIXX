from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .spec import TaskSpec
from ..envs import make_envs


@dataclass
class RolloutMux:
    """
    Minimal env multiplexer for Phase-3.

    It rebuilds env instances when switching tasks. This avoids mixing task
    distributions within a PPO rollout window.
    """
    registry: Dict[str, TaskSpec]
    n_envs: int
    cfg: object

    _current_task: Optional[str] = None
    _envs: Optional[List] = None

    def switch(self, task_id: str) -> List:
        if task_id not in self.registry:
            raise KeyError(f"Unknown task_id: {task_id}")
        # close old envs
        if self._envs:
            for e in self._envs:
                try:
                    e.close()
                except Exception:
                    pass
        spec = self.registry[task_id]
        self._envs = make_envs(spec.env_id, num_envs=self.n_envs, cfg=self.cfg)
        self._current_task = task_id
        return self._envs

    @property
    def current_task(self) -> Optional[str]:
        return self._current_task

    @property
    def envs(self) -> List:
        if self._envs is None:
            raise RuntimeError("RolloutMux not initialized. Call switch(task_id) first.")
        return self._envs
