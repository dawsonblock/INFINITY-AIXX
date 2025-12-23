from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable

import torch
import torch.nn as nn

from .config import HumanLikeConfig
from .self_model import SelfModel, SelfModelState
from .emotion_controller import EmotionController, EmotionState
from .counterfactual_simulator import CounterfactualSimulator

@dataclass
class HumanLikeStates:
    self_state: SelfModelState
    emo_state: EmotionState

class HumanLikeRouter(nn.Module):
    """A small decision wrapper that can:
    - maintain a self-latent z
    - adjust exploration pressure (entropy coef)
    - optionally do cheap counterfactual action selection

    In RL environments we cannot literally 'pause time', so 'pause' is
    interpreted as: increase deliberation by switching to counterfactual
    selection + lower entropy.
    """

    def __init__(self, cfg: HumanLikeConfig, w_dim: int, r_dim: int, act_dim: int):
        super().__init__()
        self.cfg = cfg
        self.act_dim = act_dim

        signals_dim = 6  # uncertainty, pred_err, td_err, reward, novelty, conflict (each [B,1])
        self.self_model = SelfModel(cfg, w_dim=w_dim, r_dim=r_dim, signals_dim=signals_dim)
        self.emotion = EmotionController(cfg, signals_dim=signals_dim)
        self.cf = CounterfactualSimulator(cfg)

    def init_states(self, batch: int, device: torch.device) -> Dict[str, Any]:
        return {
            "self": self.self_model.init_state(batch, device),
            "emo": self.emotion.init_state(batch, device),
        }


    def reset_on_env_reset(self, states: Dict[str, Any], done: torch.Tensor) -> Dict[str, Any]:
        """Reset per-batch elements when an env signals done.

        Args:
            states: dict from init_states/step containing keys "self" and "emo"
            done: [B,1] or [B] boolean tensor where True indicates episode boundary
        Returns:
            updated states dict
        """
        if not getattr(self.cfg, "z_reset_on_env_reset", True):
            return states
        if getattr(self.cfg, "z_persist_across_episodes", True):
            return states

        d = done
        if d.ndim == 2:
            d = d.view(-1)
        d = d.bool()
        if d.numel() == 0 or not d.any():
            return states

        st = states["self"]
        em = states["emo"]

        # zero selected rows
        st_z = st.z.clone()
        st_z[d] = 0.0
        em_e = em.e.clone()
        em_e[d] = 0.0

        states["self"] = SelfModelState(z=st_z, pause_budget=st.pause_budget.clone())
        states["emo"] = EmotionState(e=em_e, paused_steps=em.paused_steps.clone())
        return states
    def step(
        self,
        states: Dict[str, Any],
        w: torch.Tensor,          # [B, w_dim]
        r: torch.Tensor,          # [B, r_dim]
        act_dim: int,
        logits: torch.Tensor,     # [B, act_dim]
        uncertainty: torch.Tensor,# [B,1]
        confidence: torch.Tensor, # [B,1]
        signals: Dict[str, torch.Tensor],
        proposal_fn: Callable,
        model_step_fn: Callable,
        value_fn: Callable,
        policy_action_fn: Callable,
    ) -> Dict[str, Any]:
        if not self.cfg.enabled:
            a = policy_action_fn()
            return {"action": a, "states": states, "entropy_coef": torch.full_like(uncertainty, float(self.cfg.entropy_base))}

        # Update self-model
        sm_out = self.self_model.step(states["self"], w=w, r=r, signals=signals)
        self_state = sm_out["state"]
        u_hat = sm_out["uncertainty"]
        id_hat = sm_out["identity"]

        # Emotion/control
        emo_out = self.emotion.step(states["emo"], signals=signals, uncertainty=u_hat, identity=id_hat)
        emo_state = emo_out["state"]
        entropy_coef = emo_out["entropy_coef"]  # [B,1]

        # Decide whether to engage lookahead (deliberation)
        do_cf = (u_hat >= float(self.cfg.pause_uncertainty_threshold)) | (confidence <= float(self.cfg.pause_confidence_threshold))

        # Default action = policy sample
        a = policy_action_fn()

        if do_cf.any():
            # Self-model modulates how hard we think:
            # - higher identity (competence proxy) -> longer horizon, lower uncertainty penalty
            # - lower identity -> shorter horizon, higher penalty
            comp = torch.sigmoid(id_hat).mean().clamp(0.05, 0.95).item()
            base_h = int(self.cfg.cf_horizon)
            base_l = float(self.cfg.cf_lambda_uncertainty)

            eff_horizon = max(1, int(round(base_h * (0.5 + comp))))      # 0.55x .. 1.45x
            eff_lambda = max(0.0, base_l * (1.25 - comp))               # higher penalty when low confidence

            # Plan in fused workspace latent (world-model-compatible)
            latent = w.detach()
            cf_res = self.cf.choose(
                latent=latent,
                proposal_fn=proposal_fn,
                model_step_fn=model_step_fn,
                value_fn=value_fn,
                w=w.detach(),
                z=self_state.z.detach(),
                cf_horizon=eff_horizon,
                cf_lambda_uncertainty=eff_lambda,
            )
            # Blend: for states that want deliberation, take cf action, else keep a
            a_cf = cf_res.action.view(-1).to(a.device)
            a = torch.where(do_cf.view(-1), a_cf, a.view(-1))

            # When deliberating, reduce entropy_coef
            entropy_coef = torch.where(do_cf, torch.maximum(entropy_coef * 0.5, torch.full_like(entropy_coef, float(self.cfg.entropy_min))), entropy_coef)

        return {"action": a, "states": {"self": self_state, "emo": emo_state}, "entropy_coef": entropy_coef, "do_cf": do_cf}