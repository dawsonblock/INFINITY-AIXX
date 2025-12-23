from __future__ import annotations

from typing import Dict

from .spec import TaskSpec


def build_default_registry() -> Dict[str, TaskSpec]:
    """
    Default Phase-3 multitask set (small, fast, classic-control).
    task_id values are stable and used by configs/logs.
    """
    reg: Dict[str, TaskSpec] = {}

    reg["cartpole_v1"] = TaskSpec(
        task_id="cartpole_v1",
        env_id="CartPole-v1",
        success_threshold=475.0,
        max_steps=None,
        eval_episodes=25,
    )
    reg["cartpole_v0"] = TaskSpec(
        task_id="cartpole_v0",
        env_id="CartPole-v0",
        success_threshold=195.0,
        max_steps=None,
        eval_episodes=25,
    )
    reg["acrobot_v1"] = TaskSpec(
        task_id="acrobot_v1",
        env_id="Acrobot-v1",
        # "solved" conventions vary; pick a stable threshold for comparisons.
        success_threshold=-100.0,
        max_steps=None,
        eval_episodes=25,
    )
    reg["mountaincar_v0"] = TaskSpec(
        task_id="mountaincar_v0",
        env_id="MountainCar-v0",
        success_threshold=-110.0,
        max_steps=None,
        eval_episodes=25,
    )
    return reg
