from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .spec import TaskSpec


class TaskRouter:
    """Base interface."""
    def next_task(self, *, step: int, history: Dict[str, dict]) -> str:
        raise NotImplementedError


@dataclass
class RoundRobinRouter(TaskRouter):
    tasks: Sequence[str]
    _i: int = 0

    def next_task(self, *, step: int, history: Dict[str, dict]) -> str:
        if not self.tasks:
            raise ValueError("No tasks configured.")
        tid = self.tasks[self._i % len(self.tasks)]
        self._i += 1
        return tid


@dataclass
class CompetenceRouter(TaskRouter):
    """
    Picks the task with the lowest recent success rate / normalized performance.

    history[task_id] should include:
      - "recent_return_mean": float
      - "threshold": Optional[float]
    """
    tasks: Sequence[str]
    eps: float = 1e-6

    def next_task(self, *, step: int, history: Dict[str, dict]) -> str:
        if not self.tasks:
            raise ValueError("No tasks configured.")

        scores = []
        for tid in self.tasks:
            h = history.get(tid, {})
            r = float(h.get("recent_return_mean", -np.inf))
            thr = h.get("threshold", None)
            if thr is None or np.isinf(r):
                # No threshold: just use raw return ranking (lower is worse).
                norm = r
            else:
                # normalize by threshold: 1.0 means at/above threshold.
                norm = r / (float(thr) + self.eps)
            scores.append((norm, tid))

        # choose worst-performing task
        scores.sort(key=lambda x: x[0])
        return scores[0][1]


@dataclass
class SpacedRepetitionRouter(TaskRouter):
    """
    Helps fight catastrophic forgetting.

    It alternates between:
      - worst task (by normalized performance)
      - a periodic revisit of the best task every `revisit_every` segments
    """
    tasks: Sequence[str]
    revisit_every: int = 5
    _segments: int = 0

    def next_task(self, *, step: int, history: Dict[str, dict]) -> str:
        if not self.tasks:
            raise ValueError("No tasks configured.")

        self._segments += 1

        # every N segments, revisit the best task
        if self.revisit_every > 0 and (self._segments % self.revisit_every == 0):
            best = None
            for tid in self.tasks:
                h = history.get(tid, {})
                r = float(h.get("recent_return_mean", -np.inf))
                if best is None or r > best[0]:
                    best = (r, tid)
            return best[1] if best else self.tasks[0]

        # otherwise, train the worst
        worst = None
        for tid in self.tasks:
            h = history.get(tid, {})
            r = float(h.get("recent_return_mean", -np.inf))
            thr = h.get("threshold", None)
            norm = r if thr is None else (r / (float(thr) + 1e-6))
            if worst is None or norm < worst[0]:
                worst = (norm, tid)
        return worst[1] if worst else self.tasks[0]
