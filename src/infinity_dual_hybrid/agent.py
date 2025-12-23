"""
agent.py

Unified InfinityV3DualHybridAgent.

This is the canonical agent combining:
- Hybrid SSM/Attention backbone for encoding
- Dual-Tier Miras for parametric working memory
- FAISS LTM for episodic long-term memory
- Policy and value heads for RL

Data Flow:
    obs [B, obs_dim]
        |
        v
    ObservationEncoder -> [B, d_model]
        |
        v
    HybridSSMAttentionBackbone -> encoded [B, d_model]
        |
        +---> Miras.read() -> miras_v [B, d_model]
        +---> LTM.retrieve() -> ltm_v [B, d_model]
        |
        v
    MemoryFusion [encoded, miras_v, ltm_v] -> fused [B, d_model]
        |
        +---> PolicyHead -> logits [B, act_dim]
        +---> ValueHead -> value [B, 1]

Memory Updates (during training):
    - Miras.update() weighted by |advantage|
    - LTM.store() for high-importance states (RMD-gated or episode-end)
"""

from typing import Optional, Dict, Tuple, Any

import torch
import torch.nn as nn

from .config import AgentConfig
from .world_model import WorldModel
from .adapters import AdapterBank, AdapterBankConfig
from humanlike_core.config import HumanLikeConfig
from humanlike_core.decision_router import HumanLikeRouter
from .miras import DualTierMiras
from .ltm import build_ltm, LTMWrapper
from .ssm_backbone import (
    HybridSSMAttentionBackbone,
    ObservationEncoder,
    detect_ssm_backend,
)


class MemoryFusion(nn.Module):
    """
    Fuses backbone output with Miras and LTM retrievals.

    Concatenates [encoded, miras_v, ltm_v] and projects to d_model.
    """

    def __init__(
        self,
        d_model: int,
        use_miras: bool = True,
        use_ltm: bool = True,
    ):
        super().__init__()
        self.use_miras = use_miras
        self.use_ltm = use_ltm

        # Compute input dimension
        num_sources = 1  # backbone
        if use_miras:
            num_sources += 1
        if use_ltm:
            num_sources += 1

        self.proj = nn.Linear(d_model * num_sources, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        encoded: torch.Tensor,
        miras_v: Optional[torch.Tensor] = None,
        ltm_v: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            encoded: [B, d_model] backbone output
            miras_v: [B, d_model] Miras retrieval (optional)
            ltm_v: [B, d_model] LTM retrieval (optional)
        Returns:
            [B, d_model] fused representation
        """
        parts = [encoded]
        if self.use_miras and miras_v is not None:
            parts.append(miras_v)
        if self.use_ltm and ltm_v is not None:
            parts.append(ltm_v)

        fused = torch.cat(parts, dim=-1)
        fused = self.proj(fused)
        fused = self.norm(fused)
        return fused


class RMDGate(nn.Module):
    """
    Recurrent Memory Distillation gate for selective LTM writes.

    Computes importance scores for states to determine which
    should be committed to long-term memory.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, d_model]
        Returns:
            [B, 1] importance scores (sigmoid applied)
        """
        return torch.sigmoid(self.gate(x))


class InfinityV3DualHybridAgent(nn.Module):
    """
    Unified Infinity V3 Dual Hybrid Agent.

    Combines:
    - Hybrid SSM/Attention backbone
    - Dual-Tier Miras parametric memory
    - FAISS-backed episodic LTM
    - Policy and value heads for PPO

    This is the canonical agent for the Infinity V3 system.
    """

    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim

        # Observation encoder
        self.obs_encoder = ObservationEncoder(
            obs_dim=cfg.obs_dim,
            d_model=d,
            is_sequential=False,
        )

        # Backbone
        # Mamba2-capable hybrid backbone. For the Mamba2-only build, set:
        #   cfg.backbone.use_mamba = True
        #   cfg.backbone.require_mamba2 = True
        #   cfg.backbone.use_attention = False
        self.backbone = HybridSSMAttentionBackbone(cfg.backbone)
        self.use_neocortex = False
        self._neo_state = None



        if cfg.log_mamba_backend:
            backend = detect_ssm_backend(cfg.backbone)
            backend_str = "Mamba2" if backend == "mamba2" else "fallback"
            print(f"SSM backend: {backend_str}")

        # Dual-Tier Miras
        if cfg.use_miras_in_forward:
            self.miras = DualTierMiras.from_config(cfg.miras)
            self.miras_key_proj = nn.Linear(d, d)
            self.miras_val_proj = nn.Linear(d, d)
        else:
            self.miras = None
            self.miras_key_proj = None
            self.miras_val_proj = None

        # LTM
        if cfg.use_ltm_in_forward:
            ltm = build_ltm(cfg.ltm)
            # Avoid registering LTM as an nn.Module submodule (keeps checkpoints lean).
            self.__dict__['ltm'] = ltm
            self.ltm_key_proj = nn.Linear(d, d)
            self.ltm_val_proj = nn.Linear(d, d)
            self.rmd_gate = RMDGate(d)
        else:
            self.ltm = None
            self.ltm_key_proj = None
            self.ltm_val_proj = None
            self.rmd_gate = None

        # Memory fusion
        self.fusion = MemoryFusion(
            d_model=d,
            use_miras=cfg.use_miras_in_forward,
            use_ltm=cfg.use_ltm_in_forward,
        )

        # Policy head (discrete actions)
        self.policy_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, cfg.act_dim),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

        
        # Latent world model (imagination)
        self.world_model: Optional[WorldModel] = None
        if getattr(cfg, "world_model", None) is not None and cfg.world_model.enabled:
            self.world_model = WorldModel(d_model=d, act_dim=cfg.act_dim, hidden=cfg.world_model.hidden)

        # Optional adapter bank (Phase-6 cognitive metric: adapter usage entropy)
        ad_cfg = getattr(self.cfg, 'adapters', None)
        try:
            ad_cfg_obj = AdapterBankConfig(**ad_cfg.__dict__) if ad_cfg is not None else AdapterBankConfig(enabled=False)
        except Exception:
            ad_cfg_obj = AdapterBankConfig(enabled=getattr(ad_cfg, 'enabled', False) if ad_cfg is not None else False)
        self.adapters = AdapterBank(dim=d, cfg=ad_cfg_obj)
        self.current_task_id = None
        self.last_adapter_info = {"adapter_probs": None, "adapter_entropy": None, "adapter_choice": None}
        self.current_task_id = None
        self.last_adapter_info = None

        self.memory_write_gate = nn.Sequential(
            nn.Linear(d, 1),
            nn.Sigmoid(),
        )

        # Buffer for episode tracking
        self._episode_states: list = []
        self._episode_states_by_env: dict[int, list[torch.Tensor]] = {}
        self._num_envs_runtime: int | None = None

        # v2.0: Runtime state
        self._mode = cfg.mode  # "train", "eval", "inference"
        self._temperature = cfg.temperature

        # Human-like routing (optional)
        self._humanlike_enabled = bool(getattr(cfg, "humanlike_enabled", False))
        if self._humanlike_enabled:
            hl_cfg = HumanLikeConfig(
                z_dim=int(getattr(cfg, "humanlike_z_dim", 128)),
                e_dim=int(getattr(cfg, "humanlike_e_dim", 6)),
                pause_uncertainty_threshold=float(getattr(cfg, "humanlike_pause_uncertainty_threshold", 0.75)),
                pause_confidence_threshold=float(getattr(cfg, "humanlike_pause_confidence_threshold", 0.35)),
                cf_k=int(getattr(cfg, "humanlike_cf_k", 6)),
                cf_horizon=int(getattr(cfg, "humanlike_cf_horizon", 2)),
                enabled=True,
            )
            # Use fused as workspace (w_dim=d) and encoded as recall proxy (r_dim=d)
            self.humanlike_router = HumanLikeRouter(hl_cfg, w_dim=d, r_dim=d, act_dim=cfg.act_dim)
            self._hl_states: Optional[Dict[str, Any]] = None
            self._hl_last_entropy_coef: Optional[torch.Tensor] = None
        else:
            self.humanlike_router = None
            self._hl_states = None
            self._hl_last_entropy_coef = None
    def set_mode(self, mode: str) -> None:
        """Set agent mode: 'train', 'eval', or 'inference'."""
        assert mode in ("train", "eval", "inference")
        self._mode = mode

    def set_temperature(self, t: float) -> None:
        """Set policy temperature for action sampling."""
        self._temperature = t

    def debug_state(self) -> Dict[str, Any]:
        """
        Return diagnostic state for debugging and logging.

        Returns:
            Dict with backbone, Miras, and LTM state info
        """
        state = {
            "mode": self._mode,
            "temperature": self._temperature,
            "episode_buffer_size": len(self._episode_states),
        }

        # Backbone info
        state["backbone"] = {
            "has_mamba": len(self.backbone.mamba_layers) > 0,
            "has_attention": len(self.backbone.attention_layers) > 0,
            "d_model": self.backbone.d_model,
        }

        # Miras info
        if self.miras is not None:
            miras_stats = self.miras.get_stats()
            state["miras"] = {
                "fast_B_norm": miras_stats.get("fast_B_norm", 0),
                "fast_C_norm": miras_stats.get("fast_C_norm", 0),
                "deep_B_norm": miras_stats.get("deep_B_norm", 0),
                "deep_C_norm": miras_stats.get("deep_C_norm", 0),
                "mix_ratio": miras_stats.get("mix_ratio", 0),
                "fast_err_l2": miras_stats.get("fast_err_l2"),
                "deep_err_l2": miras_stats.get("deep_err_l2"),
                "deep_retention": miras_stats.get("deep_retention"),
                "deep_gradB_norm": miras_stats.get("deep_gradB_norm"),
                "deep_gradC_norm": miras_stats.get("deep_gradC_norm"),
            }
        else:
            state["miras"] = None

        # LTM info
        if self.ltm is not None:
            state["ltm"] = {
                "size": self.ltm.size,
            }
        else:
            state["ltm"] = None

        return state

    def forward(
        self,
        obs: torch.Tensor,
        advantage: Optional[torch.Tensor] = None,
        store_for_ltm: bool = False,
        episode_env_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the agent.

        Args:
            obs: [B, obs_dim] observations
            advantage: [B] advantages for Miras weighting (optional)
            store_for_ltm: If True, buffer states for LTM storage
        Returns:
            Dict with 'logits', 'value', 'encoded'
        """
        # Encode observations
        x = self.obs_encoder(obs)  # [B, d_model]

        # Pass through backbone
        encoded = self.backbone(x)  # [B, d_model]

        write_prob = self.memory_write_gate(encoded)
        gate = torch.clamp(
            write_prob,
            min=float(self.cfg.write_gate_floor),
            max=float(self.cfg.write_gate_ceiling),
        )

        effective_write = None
        if advantage is not None:
            mode = self.cfg.miras_weight_mode
            if mode == "abs_adv":
                miras_w = advantage.abs()
            elif mode == "pos_adv":
                miras_w = torch.clamp(advantage, min=0.0)
            else:
                miras_w = torch.ones_like(advantage)

            effective_write = gate.squeeze(-1) * miras_w

        if store_for_ltm and self.ltm is not None and self._mode == "train":
            if episode_env_ids is not None:
                # Per-env episode buffering (for vectorized envs)
                env_ids = episode_env_ids.detach().to('cpu').tolist()
                for i, eid in enumerate(env_ids):
                    buf = self._episode_states_by_env.setdefault(int(eid), [])
                    buf.append(encoded[i:i+1].detach().cpu())
            else:
                self._episode_states.append(encoded.detach().cpu())

        # Miras read
        miras_v = None
        if self.miras is not None:
            miras_k = self.miras_key_proj(encoded)
            read_out = self.miras.read(miras_k, context=encoded)
            miras_v = read_out["v"]

        # LTM read
        ltm_v = None
        if self.ltm is not None:
            ltm_q = self.ltm_key_proj(encoded)
            ltm_v = self.ltm.retrieve(ltm_q, top_k=self.cfg.ltm.top_k)

        # Memory fusion
        fused = self.fusion(encoded, miras_v, ltm_v)

        # Adapter bank (task-conditioned) — safe no-op when disabled
        fused, ad_info = self.adapters(fused, task_id=self.current_task_id)
        self.last_adapter_info = ad_info

        # Policy and value heads
        logits = self.policy_head(fused)
        value = self.value_head(fused).squeeze(-1)

        # Memory updates only in train mode
        if self.training and self._mode == "train":
            self._update_memories(
                encoded=encoded,
                advantage=advantage,
                write_strength=effective_write,
                store_for_ltm=store_for_ltm,
            )

        return {
            "logits": logits,
            "value": value,
            "encoded": encoded,
            "fused": fused,
            "write_prob": gate,
            "effective_write_mean": (
                effective_write.mean()
                if effective_write is not None
                else torch.tensor(0.0, device=encoded.device)
            ),
        }

    @torch.no_grad()
    def _update_memories(
        self,
        encoded: torch.Tensor,
        advantage: Optional[torch.Tensor],
        write_strength: Optional[torch.Tensor],
        store_for_ltm: bool,
    ) -> None:
        """Update Miras and optionally buffer states for LTM."""
        # Miras update with advantage weighting
        if self.miras is not None:
            miras_k = self.miras_key_proj(encoded)
            miras_target = self.miras_val_proj(encoded).detach()

            weight = None
            if write_strength is not None:
                weight = write_strength.detach()

            self.miras.update(
                miras_k,
                miras_target,
                weight=weight,
                context=encoded,
            )

    def commit_to_ltm(
        self,
        rewards: Optional[torch.Tensor] = None,
        force: bool = False,
        env_id: Optional[int] = None,
    ) -> None:
        """
        Commit buffered states to LTM.

        Called at episode boundaries or when force=True.
        Uses RMD gate to select top-k% important states.
        """
        if not force and self._mode != "train":
            return

        # If per-env buffering is active, allow committing a single env or all envs.
        if env_id is not None:
            buf = self._episode_states_by_env.get(int(env_id), [])
            if self.ltm is None or not buf:
                return
            states = torch.cat(buf, dim=0)  # [T, d_model]
            self._episode_states_by_env[int(env_id)] = []
        else:
            # Fall back to legacy single-buffer if present
            if self.ltm is None:
                return
            if self._episode_states_by_env:
                # Commit all env buffers
                all_states = []
                for k in sorted(self._episode_states_by_env.keys()):
                    if self._episode_states_by_env[k]:
                        all_states.append(torch.cat(self._episode_states_by_env[k], dim=0))
                        self._episode_states_by_env[k] = []
                if not all_states:
                    return
                states = torch.cat(all_states, dim=0)
            else:
                if not self._episode_states:
                    return
                states = torch.cat(self._episode_states, dim=0)  # [T, d_model]
                self._episode_states = []

        # Compute importance scores with RMD gate
        if self.rmd_gate is not None and not force:
            with torch.no_grad():
                scores = self.rmd_gate(states).squeeze(-1)  # [T]
                k = max(1, int(states.shape[0] * self.cfg.rmd_commit_ratio))
                _, topk_idx = torch.topk(scores, k)
                states = states[topk_idx]

        # Commit to LTM
        if states.shape[0] > 0:
            keys = self.ltm_key_proj(states)
            values = self.ltm_val_proj(states)
            self.ltm.store(keys, values)

        # Clear buffer
        self._episode_states = []
        self._episode_states_by_env = {}
        if getattr(self, 'use_neocortex', False) and self._neo_state is not None:
            self.backbone.reset_state(self._neo_state)


    def reset_episode(self) -> None:
        """Reset episode-level state."""
        self._episode_states = []
        self._episode_states_by_env = {}
        if getattr(self, 'use_neocortex', False) and self._neo_state is not None:
            self.backbone.reset_state(self._neo_state)

        self.backbone.reset_state()

        # Human-like state reset (episode boundary)
        if getattr(self, "humanlike_router", None) is not None:
            self._hl_states = None

        if self.cfg.reset_miras_on_episode and self.miras is not None:
            self.miras.reset_state()


    def reset_episode_env(self, env_ids: list[int]) -> None:
        """Reset per-env episode state for vectorized environments.

        This is used with vectorized envs where only a subset of env instances
        terminate on a given step. We must clear any per-episode buffers and
        any per-env recurrent state for those indices only.
        """
        if not env_ids:
            return

        # Per-env episode buffers (for optional LTM commit on episode end)
        if hasattr(self, "_episode_states_by_env") and self._episode_states_by_env is not None:
            for eid in env_ids:
                if eid in self._episode_states_by_env:
                    self._episode_states_by_env[eid] = []

        # Human-like router state: re-init and copy just those env indices
        if getattr(self, "humanlike_router", None) is not None and getattr(self, "_hl_states", None) is not None:
            batch = int(self._hl_states["self"].z.shape[0])
            device = self._hl_states["self"].z.device
            fresh = self.humanlike_router.init_states(batch, device)
            for eid in env_ids:
                if 0 <= eid < batch:
                    for k in self._hl_states.keys():
                        src = fresh[k]
                        dst = self._hl_states[k]
                        for field, t in src.__dict__.items():
                            if isinstance(t, torch.Tensor):
                                getattr(dst, field)[eid].copy_(t[eid])

        # Neocortex / backbone recurrent state: zero-out per env (if present)
        if getattr(self, "use_neocortex", False) and getattr(self, "_neo_state", None) is not None:
            for eid in env_ids:
                for k in self._neo_state.keys():
                    if self._neo_state[k].shape[0] > eid:
                        self._neo_state[k][eid].zero_()

        # If the backbone has any per-env cached inference state, clear those indices too.
        if hasattr(self.backbone, "reset_state_env"):
            try:
                self.backbone.reset_state_env(env_ids)
            except Exception:
                # Fall back to a full reset if per-env reset isn't supported by the backend.
                self.backbone.reset_state()

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        store_for_ltm: bool = False,
        return_fused: bool = False,
        episode_env_ids: Optional[torch.Tensor] = None,
    ):
        """
        Get action from policy.

        Args:
            obs: [B, obs_dim] observations
            deterministic: If True, return argmax action
            return_fused: If True, also return fused workspace w
        Returns:
            (action, log_prob, value) or (action, log_prob, value, w)
        """
        out = self.forward(
            obs,
            store_for_ltm=store_for_ltm,
            episode_env_ids=episode_env_ids,
        )
        logits = out["logits"]
        value = out["value"]

        if deterministic or not getattr(self, "humanlike_router", None):
            if deterministic:
                action = torch.argmax(logits, dim=-1)
                log_prob = torch.zeros_like(value)
            else:
                scaled_logits = logits / max(self._temperature, 1e-8)
                dist = torch.distributions.Categorical(logits=scaled_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            if return_fused:
                w = out.get("fused", out["encoded"])
                return action, log_prob, value, w
            return action, log_prob, value

        # ---- Human-like routing path ----
        # Confidence/uncertainty proxies from logits
        scaled_logits = logits / max(self._temperature, 1e-8)
        dist0 = torch.distributions.Categorical(logits=scaled_logits)
        entropy = dist0.entropy().unsqueeze(-1)  # [B,1]
        probs = torch.softmax(scaled_logits, dim=-1)
        confidence = probs.max(dim=-1).values.unsqueeze(-1)  # [B,1]
        # Normalize uncertainty by log(|A|)
        uncertainty = entropy / max(float(torch.log(torch.tensor(float(self.cfg.act_dim)))), 1e-8)

        signals = {
            "uncertainty": uncertainty.detach(),
            "pred_err": torch.zeros_like(uncertainty),
            "td_err": torch.zeros_like(uncertainty),
            "reward": torch.zeros_like(uncertainty),
            "novelty": torch.zeros_like(uncertainty),
            "conflict": torch.zeros_like(uncertainty),
        }

        # Initialize router states if needed
        if self._hl_states is None or (self._hl_states["self"].z.shape[0] != obs.shape[0]):
            self._hl_states = self.humanlike_router.init_states(obs.shape[0], obs.device)

        w = out.get("fused", out["encoded"])
        r = out["encoded"]

        # Proposal: K samples from current policy distribution
        K = int(getattr(self.humanlike_router.cfg, "cf_k", 6))

        def proposal_fn(w_in, z_in):
            a = dist0.sample((K,))  # [K,B]
            return a.unsqueeze(-1)

        # Model step: prefer learned world model if present; otherwise fall back to proxy.
        def model_step_fn(latent, a):
            if self.world_model is not None:
                return self.world_model.model_step(latent, a)
            a_idx = a.long().view(-1)
            r_hat = scaled_logits.gather(-1, a_idx.view(-1, 1)).detach()
            u_hat = uncertainty.detach()
            return latent, r_hat, u_hat

        def value_fn(latent):
            return self.value_head(latent).view(-1,1).detach()

        def policy_action_fn():
            return dist0.sample()

        routed = self.humanlike_router.step(
            states=self._hl_states,
            w=w.detach(),
            r=r.detach(),
            act_dim=self.cfg.act_dim,
            logits=logits.detach(),
            uncertainty=uncertainty.detach(),
            confidence=confidence.detach(),
            signals=signals,
            proposal_fn=proposal_fn,
            model_step_fn=model_step_fn,
            value_fn=value_fn,
            policy_action_fn=policy_action_fn,
        )

        self._hl_states = routed["states"]
        self._hl_last_entropy_coef = routed["entropy_coef"]
        self._hl_last_do_cf = routed.get("do_cf", None)

        action = routed["action"].view(-1)
        log_prob = dist0.log_prob(action)

        if return_fused:
            return action, log_prob, value, w
        return action, log_prob, value
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        advantage: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Evaluate actions for PPO update.

        Args:
            obs: [B, obs_dim]
            actions: [B]
            advantage: [B] for Miras weighting
        Returns:
            (log_prob, value, entropy)
        """
        out = self.forward(obs, advantage=advantage)
        logits = out["logits"]
        value = out["value"]
        write_prob = out["write_prob"].squeeze(-1)
        effective_write_mean = out["effective_write_mean"]

        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_prob, value, entropy, write_prob, effective_write_mean

    @classmethod
    def from_config(cls, cfg: AgentConfig) -> "InfinityV3DualHybridAgent":
        """Create agent from config."""
        return cls(cfg)

    def save(self, path: str) -> None:
        """Save agent state."""
        state = {
            "model_state_dict": self.state_dict(),
            "config": self.cfg,
        }
        if self.ltm is not None:
            state["ltm_state"] = self.ltm.state_dict_ltm()
        torch.save(state, path)

    @classmethod
    def load(
        cls,
        path: str,
        device: str = "cpu",
    ) -> "InfinityV3DualHybridAgent":
        """Load agent from checkpoint."""
        state = torch.load(path, map_location=device)
        agent = cls(state["config"])
        agent.load_state_dict(state["model_state_dict"])
        if "ltm_state" in state and agent.ltm is not None:
            agent.ltm.load_state_dict_ltm(state["ltm_state"])
        return agent.to(device)

    def shutdown(self) -> None:
        """Clean shutdown (stop async LTM writer if active)."""
        if self.ltm is not None:
            self.ltm.shutdown()


def build_agent(cfg: AgentConfig) -> InfinityV3DualHybridAgent:
    """Build agent from config."""
    return InfinityV3DualHybridAgent(cfg)

