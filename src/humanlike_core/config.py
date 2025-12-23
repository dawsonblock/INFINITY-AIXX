from __future__ import annotations
from dataclasses import dataclass

@dataclass
class HumanLikeConfig:
    # Latent self-model
    z_dim: int = 128
    e_dim: int = 6

    # Episode persistence / reset behavior
    z_persist_across_episodes: bool = True
    z_reset_on_env_reset: bool = True

    # Router thresholds
    pause_uncertainty_threshold: float = 0.75
    pause_confidence_threshold: float = 0.35
    max_pause_steps: int = 4

    # Counterfactual sampler (cheap lookahead)
    cf_k: int = 6
    cf_horizon: int = 2
    cf_lambda_identity: float = 0.35
    cf_lambda_risk: float = 0.25
    cf_lambda_uncertainty: float = 0.30

    # Dynamic exploration scaling (maps to PPO entropy_coef)
    entropy_base: float = 0.01
    entropy_min: float = 0.001
    entropy_max: float = 0.05

    enabled: bool = True
