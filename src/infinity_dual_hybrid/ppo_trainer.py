"""
ppo_trainer.py

PPO (Proximal Policy Optimization) Trainer for Infinity V3.

Features:
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective
- Value function loss with optional clipping
- Entropy bonus for exploration
- Optional KL penalty against old policy
- Multi-environment rollout collection
- Gradient clipping
- Learning rate scheduling

Usage:
    trainer = PPOTrainer(agent, cfg)
    for iteration in range(cfg.max_iterations):
        rollouts = trainer.collect_rollouts(envs)
        stats = trainer.train_step(rollouts)
"""

from typing import Any, Dict, List, Tuple, Optional, NamedTuple
import numpy as np

import torch
from torch.amp import autocast, GradScaler
import torch.nn as nn

from .config import PPOConfig
from .utils import obs_to_tensor, PinnedTransfer
from .agent import InfinityV3DualHybridAgent
from .trace import Tracer


class RolloutBatch(NamedTuple):
    """Container for rollout data."""
    observations: torch.Tensor  # [N, obs_dim]
    actions: torch.Tensor       # [N]
    log_probs: torch.Tensor     # [N]
    values: torch.Tensor        # [N]
    rewards: torch.Tensor       # [N]
    dones: torch.Tensor         # [N]
    advantages: torch.Tensor    # [N]
    returns: torch.Tensor       # [N]
    w: torch.Tensor             # [N, hidden]
    w_next: torch.Tensor        # [N, hidden]
    router_do_cf: torch.Tensor  # [N]
    adapter_choice: torch.Tensor  # [N] (int64, -1 if disabled)
    adapter_entropy: torch.Tensor  # [N] (float, 0 if disabled)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation.

    Args:
        rewards: [T] rewards
        values: [T+1] value estimates (includes bootstrap)
        dones: [T] episode termination flags
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
    Returns:
        (advantages, returns) both [T]
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)

    gae = 0.0
    for t in reversed(range(T)):
        next_non_terminal = 1.0 - dones[t]
        delta = (
            rewards[t]
            + gamma * values[t + 1] * next_non_terminal
            - values[t]
        )
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[t] = gae
        returns[t] = gae + values[t]

    return advantages, returns


def compute_gae_torch(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Torch vectorized GAE across environments.

    Shapes:
        rewards: [T, E]
        values:  [T+1, E] (bootstrap value at last step)
        dones:   [T, E] (1.0 if done else 0.0)

    Returns:
        advantages: [T, E]
        returns:    [T, E]
    """
    assert rewards.dim() == 2 and dones.dim() == 2, "rewards/dones must be [T,E]"
    assert values.dim() == 2 and values.shape[0] == rewards.shape[0] + 1, "values must be [T+1,E]"
    T, E = rewards.shape
    advantages = torch.zeros((T, E), device=rewards.device, dtype=rewards.dtype)
    returns = torch.zeros((T, E), device=rewards.device, dtype=rewards.dtype)

    gae = torch.zeros((E,), device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[t] = gae
        returns[t] = gae + values[t]
    return advantages, returns


class PPOTrainer:
    """
    PPO Trainer for InfinityV3DualHybridAgent.

    Handles:
    - Rollout collection from environments
    - Advantage estimation with GAE
    - Policy and value optimization with clipping
    - Entropy bonus and optional KL penalty
    """

    def __init__(
        self,
        agent: InfinityV3DualHybridAgent,
        cfg: PPOConfig,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        self.agent = agent
        self.cfg = cfg
        self.device = device
        self.seed = seed
        self._seeded_env_ids = set()

        # Reusable pinned-CPU staging buffers (CUDA only)
        self._pinned_obs: Optional[PinnedTransfer] = None
        self._pinned_rew: Optional[PinnedTransfer] = None
        self._pinned_done: Optional[PinnedTransfer] = None

        # Optimizer
        # Optimizer (exclude world model params so we can train it separately)
        if getattr(self.agent, "world_model", None) is not None:
            base_params = [p for n, p in self.agent.named_parameters() if not n.startswith("world_model.")]
        else:
            base_params = list(self.agent.parameters())

        self.optimizer = torch.optim.AdamW(
            base_params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        
        # Mixed precision (AMP) support (optional)
        amp_cfg = None
        if isinstance(self.cfg, dict):
            amp_cfg = (self.cfg.get("amp", {}) or {})
            self.amp_enabled = bool(amp_cfg.get("enabled", False))
            dtype_name = str(amp_cfg.get("dtype", "fp16")).lower()
        else:
            amp_cfg = getattr(self.cfg, "amp", None)
            self.amp_enabled = bool(getattr(amp_cfg, "enabled", False)) if amp_cfg is not None else False
            dtype_name = str(getattr(amp_cfg, "dtype", "fp16")).lower() if amp_cfg is not None else "fp16"

        if dtype_name in ("bf16", "bfloat16"):
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16

        self.scaler = GradScaler(enabled=(self.amp_enabled and str(self.device).startswith("cuda")))

# World model optimizer (if enabled)
        self.wm_opt = None
        if getattr(self.agent, "world_model", None) is not None and self.agent.world_model is not None:
            wm_cfg = getattr(self.agent.cfg, "world_model", None)
            lr = getattr(wm_cfg, "lr", cfg.learning_rate)
            wd = getattr(wm_cfg, "weight_decay", cfg.weight_decay)
            self.wm_opt = torch.optim.AdamW(self.agent.world_model.parameters(), lr=lr, weight_decay=wd)

        self.current_lr = cfg.learning_rate

        # Optional: old policy for KL penalty
        self._old_policy_params: Optional[Dict[str, torch.Tensor]] = None
        if cfg.use_kl_penalty:
            self._store_old_policy()

        # v2.0: Adaptive KL target
        self.kl_coef = cfg.kl_coef

        # Stats tracking
        self.iteration = 0
        self.grad_explosion_count = 0

        # Debug trace (opt-in via env vars)
        self.tracer = Tracer.from_env()

    def _store_old_policy(self) -> None:
        """Store copy of policy parameters for KL penalty."""
        self._old_policy_params = {
            name: param.clone().detach()
            for name, param in self.agent.named_parameters()
        }

def collect_rollouts(
    self,
    envs: Any,
    steps: Optional[int] = None,
) -> RolloutBatch:
    """
    Collect rollouts from environments.

    Stores rollouts as [T, E, ...] tensors, computes GAE in torch, then flattens
    to [N=T*E] for PPO minibatching.

    Args:
        envs: List of Gym-like environments
        steps: Number of steps per env (default: cfg.steps_per_rollout)

    Returns:
        RolloutBatch with collected data
    """
    steps = int(steps or self.cfg.steps_per_rollout)

    # Trace header (best-effort env id)
    env_id = "unknown"
    try:
        if hasattr(envs, "single_env") and hasattr(envs.single_env, "spec") and envs.single_env.spec is not None:
            env_id = str(envs.single_env.spec.id)
        elif hasattr(envs, "envs") and envs.envs and hasattr(envs.envs[0], "spec") and envs.envs[0].spec is not None:
            env_id = str(envs.envs[0].spec.id)
        elif isinstance(envs, list) and envs and hasattr(envs[0], "spec") and envs[0].spec is not None:
            env_id = str(envs[0].spec.id)
    except Exception:
        env_id = "unknown"
    num_envs = getattr(envs, "num_envs", None)
    vector_env = num_envs is not None and not isinstance(envs, list)
    if not vector_env:
        num_envs = len(envs)

    # Trace: rollout begin
    try:
        self.tracer.rollout_begin(env_id=str(env_id), num_envs=int(num_envs), rollout_steps=int(steps))
    except Exception:
        pass

    # Reset envs and agent state (HARD_RESET_CONTRACT)
    if vector_env:
        try:
            reset_out = envs.reset(seed=int(self.seed) if self.seed is not None else None)
        except TypeError:
            reset_out = envs.reset()
        if isinstance(reset_out, tuple):
            current_obs = reset_out[0]
        else:
            current_obs = reset_out
    else:
        current_obs: List = []
        for i, env in enumerate(envs):
            needs_seed = (
                self.seed is not None
                and id(env) not in self._seeded_env_ids
            )
            if needs_seed:
                try:
                    reset_out = env.reset(seed=int(self.seed) + int(i))
                except TypeError:
                    reset_out = env.reset()
                self._seeded_env_ids.add(id(env))
            else:
                reset_out = env.reset()
            if isinstance(reset_out, tuple):
                obs = reset_out[0]
            else:
                obs = reset_out
            current_obs.append(obs)
    self.agent.reset_episode()
    self.agent.eval()

    store_for_ltm = False
    if self.agent.ltm is not None:
        store_for_ltm = bool(self.agent.cfg.ltm.store_on_episode_end)

    # Infer obs_dim from first observation
    if vector_env:
        obs0 = np.asarray(current_obs[0], dtype=np.float32)
    else:
        obs0 = np.asarray(current_obs[0], dtype=np.float32)
    obs_dim = int(obs0.shape[-1]) if obs0.ndim > 0 else 1

    # Pinned host staging buffers make non_blocking H2D copies truly async.
    use_pinned = bool(getattr(self.cfg, 'pin_memory', False)) and str(self.device) != 'cpu' and torch.cuda.is_available()
    if use_pinned:
        self._pinned_obs = PinnedStager(shape=(num_envs, obs_dim), dtype=torch.float32)
        self._pinned_rew = PinnedStager(shape=(num_envs,), dtype=torch.float32)
        self._pinned_done = PinnedStager(shape=(num_envs,), dtype=torch.float32)
    else:
        self._pinned_obs = None
        self._pinned_rew = None
        self._pinned_done = None

    obs_t = torch.zeros((steps, num_envs, obs_dim), dtype=torch.float32, device=self.device)
    act_t = torch.zeros((steps, num_envs), dtype=torch.long, device=self.device)
    logp_t = torch.zeros((steps, num_envs), dtype=torch.float32, device=self.device)
    rew_t = torch.zeros((steps, num_envs), dtype=torch.float32, device=self.device)
    val_t = torch.zeros((steps, num_envs), dtype=torch.float32, device=self.device)
    done_t = torch.zeros((steps, num_envs), dtype=torch.float32, device=self.device)
    do_cf_t = torch.zeros((steps, num_envs), dtype=torch.float32, device=self.device)  # router intervention mask
    w_t = torch.zeros((steps, num_envs, int(self.agent.cfg.hidden_dim)), dtype=torch.float32, device=self.device)
    adapter_choice_t = torch.full((steps, num_envs), -1, dtype=torch.long, device=self.device)
    adapter_entropy_t = torch.zeros((steps, num_envs), dtype=torch.float32, device=self.device)
    # Per-env episode tracking (vector-safe)
    ep_returns = np.zeros((num_envs,), dtype=np.float32)
    ep_lengths = np.zeros((num_envs,), dtype=np.int32)
    completed_ep_returns: list[float] = []
    completed_ep_lengths: list[int] = []

    for step in range(steps):
        # Encode obs batch
        obs_batch = obs_to_tensor(current_obs, device=self.device, pinned=self._pinned_obs)
        with torch.no_grad():
            actions, log_probs, values, fused = self.agent.get_action(
                obs_batch,
                store_for_ltm=store_for_ltm,
                return_fused=True,
                episode_env_ids=(torch.arange(int(num_envs), device=obs_batch.device) if store_for_ltm and (vector_env or int(num_envs) > 1) else None),
            )

        # Trace (after agent forward, before env step)
        agent_out: Dict[str, Any] = {
            "fused": fused,
            "value": values,
        }

        # Store rollout tensors
        obs_t[step] = obs_batch.to(obs_t.dtype)
        act_t[step] = actions.view(-1).to(act_t.dtype)
        logp_t[step] = log_probs.view(-1).to(logp_t.dtype)
        val_t[step] = values.view(-1).to(val_t.dtype)
        w_t[step] = fused.view(num_envs, -1).to(w_t.dtype)

        # Adapter diagnostics (if enabled)
        ad = getattr(self.agent, 'last_adapter_info', None)
        if isinstance(ad, dict) and ad.get('adapter_choice') is not None:
            try:
                adapter_choice_t[step] = ad.get('adapter_choice').view(-1).to(adapter_choice_t.dtype).to(self.device)
                adapter_entropy_t[step] = ad.get('adapter_entropy').view(-1).to(adapter_entropy_t.dtype).to(self.device)
            except Exception:
                pass
        # Router intervention mask (if humanlike router enabled)
        do_cf = getattr(self.agent, '_hl_last_do_cf', None)
        if do_cf is None:
            do_cf_t[step] = 0.0
        else:
            do_cf_t[step] = do_cf.view(-1).float().to(self.device)


        # Step envs
        if vector_env:
            act_np = actions.detach().to('cpu').numpy()
            step_out = envs.step(act_np)
            if len(step_out) == 5:
                next_obs, reward, terminated, truncated, info = step_out
                done_flags = np.asarray(terminated, dtype=np.bool_) | np.asarray(truncated, dtype=np.bool_)
            else:
                next_obs, reward, done, info = step_out
                done_flags = np.asarray(done, dtype=np.bool_)

            # Write rewards/dones
            rew_t[step] = (self._pinned_rew.to_device(reward, device=self.device) if self._pinned_rew is not None else torch.as_tensor(reward, dtype=torch.float32, device=self.device))
            done_t[step] = (self._pinned_done.to_device(done_flags.astype(np.float32), device=self.device) if self._pinned_done is not None else torch.as_tensor(done_flags.astype(np.float32), dtype=torch.float32, device=self.device))

            # Trace: rollout step (vector)
            try:
                agent_out = {
                    "fused": fused,
                    "value": values,
                }
                self.tracer.rollout_step(
                    obs=obs_batch,
                    agent_out=agent_out,
                    action=actions,
                    reward=rew_t[step],
                    done=done_t[step],
                )
            except Exception:
                pass

            # Per-env episode stats
            ep_returns += np.asarray(reward, dtype=np.float32)
            ep_lengths += 1

            # Episode boundaries: reset only done envs if supported
            if bool(done_flags.any()):
                done_ids = np.nonzero(done_flags)[0].tolist()

                # Record per-env episodes
                for eid in done_ids:
                    completed_ep_returns.append(float(ep_returns[int(eid)]))
                    completed_ep_lengths.append(int(ep_lengths[int(eid)]))
                    ep_returns[int(eid)] = 0.0
                    ep_lengths[int(eid)] = 0

                try:
                    reset_out = envs.reset(options={"reset_mask": done_flags})
                    reset_obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                    next_obs = np.asarray(next_obs)
                    next_obs[done_flags] = np.asarray(reset_obs)[done_flags]
                except Exception:
                    # Fallback: leave terminal obs in-place if partial reset not supported
                    pass

                # Commit LTM per env + reset agent per-env episode state
                for eid in done_ids:
                    self.agent.commit_to_ltm(env_id=int(eid))
                self.agent.reset_episode_env([int(eid) for eid in done_ids])

            current_obs = next_obs
        else:
            next_obs_list: List = []
            for i, env in enumerate(envs):
                action = int(actions[i].item())
                step_out = env.step(action)

                if len(step_out) == 5:
                    next_obs, reward, done, truncated, _info = step_out
                    done_flag = bool(done or truncated)
                else:
                    next_obs, reward, done, _info = step_out
                    done_flag = bool(done)

                rew_t[step, i] = float(reward)
                done_t[step, i] = 1.0 if done_flag else 0.0

                ep_returns[i] += float(reward)
                ep_lengths[i] += 1

                if done_flag:
                    completed_ep_returns.append(float(ep_returns[i]))
                    completed_ep_lengths.append(int(ep_lengths[i]))
                    ep_returns[i] = 0.0
                    ep_lengths[i] = 0
                    reset_out = env.reset()
                    if isinstance(reset_out, tuple):
                        next_obs = reset_out[0]
                    self.agent.commit_to_ltm(env_id=int(i))
                    self.agent.reset_episode_env([int(i)])

                next_obs_list.append(next_obs)

            # Trace: rollout step (non-vector)
            try:
                self.tracer.rollout_step(
                    obs=obs_batch,
                    agent_out=agent_out,
                    action=actions,
                    reward=rew_t[step],
                    done=done_t[step],
                )
            except Exception:
                pass

            current_obs = next_obs_list

    # Bootstrap values for the last observations
    with torch.no_grad():
        last_obs_batch = obs_to_tensor(current_obs, device=self.device, pinned=self._pinned_obs)
        _, _, last_values, last_fused = self.agent.get_action(last_obs_batch, return_fused=True)

    # values for GAE need [T+1, E]
    w_next_t = torch.cat([w_t[1:], last_fused.view(1, num_envs, -1)], dim=0)

    values_gae = torch.cat([val_t, last_values.view(1, num_envs).to(val_t.dtype)], dim=0)

    advantages, returns = compute_gae_torch(
        rewards=rew_t,
        values=values_gae,
        dones=done_t,
        gamma=self.cfg.gamma,
        gae_lambda=self.cfg.gae_lambda,
    )

    # Flatten to [N=T*E] (views, stay on device)
    N = steps * num_envs
    rollouts_on_cpu = bool(getattr(self.cfg, 'rollouts_on_cpu', False))
    if rollouts_on_cpu:
        obs_flat = obs_t.reshape(N, obs_dim).detach().cpu()
        act_flat = act_t.reshape(N).detach().cpu()
        logp_flat = logp_t.reshape(N).detach().cpu()
        val_flat = val_t.reshape(N).detach().cpu()
        rew_flat = rew_t.reshape(N).detach().cpu()
        done_flat = done_t.reshape(N).detach().cpu()
        do_cf_flat = do_cf_t.reshape(N).detach().cpu()
        adv_flat = advantages.reshape(N).detach().cpu()
        ret_flat = returns.reshape(N).detach().cpu()
    else:
        obs_flat = obs_t.reshape(N, obs_dim).detach()
        act_flat = act_t.reshape(N).detach()
        logp_flat = logp_t.reshape(N).detach()
        val_flat = val_t.reshape(N).detach()
        rew_flat = rew_t.reshape(N).detach()
        done_flat = done_t.reshape(N).detach()
        do_cf_flat = do_cf_t.reshape(N).detach()
        adv_flat = advantages.reshape(N).detach()
        ret_flat = returns.reshape(N).detach()


    # Latent rollouts + adapter diagnostics
    w_flat = (w_t.reshape(N, int(self.agent.cfg.hidden_dim)).detach().cpu() if rollouts_on_cpu else w_t.reshape(N, int(self.agent.cfg.hidden_dim)).detach())
    w_next_flat = (w_next_t.reshape(N, int(self.agent.cfg.hidden_dim)).detach().cpu() if rollouts_on_cpu else w_next_t.reshape(N, int(self.agent.cfg.hidden_dim)).detach())
    adapter_choice_flat = (adapter_choice_t.reshape(N).detach().cpu() if rollouts_on_cpu else adapter_choice_t.reshape(N).detach())
    adapter_entropy_flat = (adapter_entropy_t.reshape(N).detach().cpu() if rollouts_on_cpu else adapter_entropy_t.reshape(N).detach())

    # Stash episode stats for logging
    if completed_ep_returns:
        self.last_episode_stats = {
            'episode_return_mean': float(np.mean(completed_ep_returns)),
            'episode_return_std': float(np.std(completed_ep_returns)),
            'episode_length_mean': float(np.mean(completed_ep_lengths)),
            'episode_length_std': float(np.std(completed_ep_lengths)),
            'episodes_finished': float(len(completed_ep_returns)),
        }
    else:
        self.last_episode_stats = {'episodes_finished': 0.0}
    # Normalize advantages (on-device)
    adv_mean = adv_flat.mean()
    adv_std = adv_flat.std().clamp_min(1e-8)
    adv_flat = (adv_flat - adv_mean) / adv_std

    return RolloutBatch(

        observations=obs_flat,
        actions=act_flat,
        log_probs=logp_flat,
        values=val_flat,
        rewards=rew_flat,
        dones=done_flat,
        advantages=adv_flat,
        returns=ret_flat,
        w=w_flat,
        w_next=w_next_flat,
        router_do_cf=do_cf_flat,
        adapter_choice=adapter_choice_flat,
        adapter_entropy=adapter_entropy_flat,
    )


def train_step(self, rollouts: RolloutBatch) -> Dict[str, float]:
    """
    Perform PPO training step on collected rollouts.

    Args:
        rollouts: RolloutBatch from collect_rollouts
    Returns:
        Dict of training statistics
    """
    self.agent.train()
    cfg = self.cfg

    # Move to device
    obs = rollouts.observations.to(self.device)
    actions = rollouts.actions.to(self.device)
    old_log_probs = rollouts.log_probs.to(self.device)
    advantages = rollouts.advantages.to(self.device)
    returns = rollouts.returns.to(self.device)
    old_values = rollouts.values.to(self.device)
    router_do_cf = getattr(rollouts, 'router_do_cf', None)
    if router_do_cf is None:
        router_do_cf = torch.zeros_like(rollouts.actions, dtype=torch.float32).to(self.device)
    else:
        router_do_cf = router_do_cf.to(self.device).float()

    num_samples = obs.shape[0]
    num_minibatches = int((num_samples + cfg.batch_size - 1) // cfg.batch_size)
    self.tracer.update_begin(ppo_epoch=int(cfg.train_epochs), num_minibatches=num_minibatches)
    perm = None  # torch permutation indices (device)

    # Stats accumulators
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    total_grad_norm = 0.0
    total_memory_gate_loss = 0.0
    total_write_prob = 0.0
    write_prob_min = float("inf")
    write_prob_max = float("-inf")
    total_adv_used = 0.0
    adv_used_min = float("inf")
    adv_used_max = float("-inf")
    total_write_prob_adv_corr = 0.0
    total_effective_write = 0.0
    total_router_do_cf = 0.0
    num_updates = 0

    for epoch in range(cfg.train_epochs):
        perm = torch.randperm(num_samples, device=self.device)

        for start in range(0, num_samples, cfg.batch_size):
            end = start + cfg.batch_size
            mb_idx = perm[start:end]

            mb_obs = obs[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_logp = old_log_probs[mb_idx]
            mb_adv = advantages[mb_idx]
            mb_returns = returns[mb_idx]
            mb_router_do_cf = router_do_cf[mb_idx]
            _ = old_values[mb_idx]  # Reserved for value clipping

            adv = mb_adv
            if cfg.adv_norm:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            if cfg.adv_clip is not None and cfg.adv_clip > 0:
                adv = torch.clamp(adv, -cfg.adv_clip, cfg.adv_clip)
            adv_used = (
                torch.clamp(adv, min=0.0) if cfg.adv_positive_only else adv
            )

            with autocast(device_type=self.amp_device_type, enabled=self.scaler.is_enabled(), dtype=self.amp_dtype):
                (
                    new_logp,
                    new_values,
                    entropy,
                    write_prob,
                    effective_write_mean,
                ) = self.agent.evaluate_actions(
                mb_obs,
                mb_actions,
                advantage=adv,
            )

            if not bool(torch.isfinite(new_logp).all().item()):
                raise FloatingPointError("Non-finite PPO new_logp")
            if not bool(torch.isfinite(new_values).all().item()):
                raise FloatingPointError("Non-finite PPO new_values")
            if not bool(torch.isfinite(entropy).all().item()):
                raise FloatingPointError("Non-finite PPO entropy")
            if not bool(torch.isfinite(write_prob).all().item()):
                raise FloatingPointError("Non-finite PPO write_prob")
            if not bool(torch.isfinite(adv).all().item()):
                raise FloatingPointError("Non-finite PPO advantage")

            # Policy loss (clipped surrogate)
            ratio = (new_logp - mb_old_logp).exp()
            surr1 = ratio * adv
            surr2 = torch.clamp(
                ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps
            ) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (optionally clipped)
            value_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()

            # Entropy bonus
            entropy_loss = entropy.mean()

            eps = 1e-8
            gate = torch.clamp(
                write_prob,
                min=float(self.agent.cfg.write_gate_floor),
                max=float(self.agent.cfg.write_gate_ceiling),
            )
            memory_gate_loss = -(
                adv_used.detach() * torch.log(gate + eps)
            ).mean()

            x = adv_used.detach()
            y = write_prob.detach()
            x0 = x - x.mean()
            y0 = y - y.mean()
            denom = (x0.std() * y0.std()) + 1e-8
            corr = float((x0 * y0).mean().item() / denom.item())

            # Total loss
            # Dynamic entropy (human-like emotion controller can override)
            dynamic_entropy_coef = cfg.entropy_coef
            hl_ent = getattr(self.agent, "_hl_last_entropy_coef", None)
            if hl_ent is not None:
                try:
                    dynamic_entropy_coef = float(hl_ent.mean().item())
                    # sanity clamp
                    dynamic_entropy_coef = max(0.0, min(dynamic_entropy_coef, 1.0))
                except Exception:
                    dynamic_entropy_coef = cfg.entropy_coef

            loss = (
                policy_loss
                + cfg.value_loss_coef * value_loss
                - dynamic_entropy_coef * entropy_loss
                + cfg.mem_gate_coef * memory_gate_loss
            )

            # Trace: update minibatch (best-effort, opt-in)
            try:
                eval_out = {
                    "value": new_values,
                    "entropy": entropy,
                    "logp": new_logp,
                }
                losses = {
                    "total": float(loss.item()),
                    "policy": float(policy_loss.item()),
                    "value": float(value_loss.item()),
                    "entropy": float(entropy_loss.item()),
                    "mem_gate": float(memory_gate_loss.item()),
                }
                self.tracer.update_minibatch(
                    mb_idx=int(start // max(int(cfg.batch_size), 1)),
                    obs=mb_obs,
                    actions=mb_actions,
                    adv=adv,
                    returns=mb_returns,
                    eval_out=eval_out,
                    losses=losses,
                    wm_losses=None,
                )
            except Exception:
                pass

            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("Non-finite PPO loss")

            # Optional KL penalty with adaptive coefficient
            if cfg.use_kl_penalty:
                kl = (mb_old_logp - new_logp).mean()
                loss = loss + self.kl_coef * kl
                total_kl += kl.item()

            # Optimize
            self.optimizer.zero_grad()
            if self.scaler.is_enabled():
                # AMP backward + step
                self.scaler.scale(loss).backward()
                # unscale for clipping / grad norm
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()
            
            # Track gradient norm before clipping
            grad_norm = 0.0
            if cfg.track_grad_norm:
                for p in self.agent.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                total_grad_norm += grad_norm

            # Gradient explosion detection and LR reduction
            if grad_norm > cfg.grad_explosion_threshold:
                self.grad_explosion_count += 1
                if self.grad_explosion_count >= 3:
                    self.current_lr *= cfg.lr_reduce_factor
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = self.current_lr
                    self.grad_explosion_count = 0

            nn.utils.clip_grad_norm_(
                self.agent.parameters(), cfg.max_grad_norm
            )
            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # --- World model supervised update (optional) ---
            if self.wm_opt is not None and getattr(self.agent, "world_model", None) is not None:
                wm = self.agent.world_model
                wm_cfg = getattr(self.agent.cfg, "world_model", None)

                # World model trains on real transitions (no synthetic pseudo rollouts).
                mb_w = rollouts.w.to(self.device)[mb_idx]
                mb_w_next = rollouts.w_next.to(self.device)[mb_idx]
                mb_a = mb_actions.to(self.device)
                mb_r = rollouts.rewards.to(self.device)[mb_idx].view(-1, 1)
                mb_d = rollouts.dones.to(self.device)[mb_idx].view(-1, 1)

                w = mb_w
                w_next = mb_w_next
                a = mb_a
                r = mb_r
                d = mb_d

                mask = (1.0 - d)

                # Heteroscedastic world model loss (NLL) + done BCE
                wm_mask = (1.0 - d)  # [B,1] mask transitions that cross episode boundaries
                wm_loss, wm_metrics = wm.loss(
                    w=w,
                    a=a,
                    w_next_target=w_next,
                    r_target=r,
                    done_target=d,
                    mask=wm_mask,
                )

                w_next_coef = getattr(wm_cfg, "w_next_coef", 1.0) if wm_cfg is not None else 1.0
                r_coef = getattr(wm_cfg, "r_coef", 1.0) if wm_cfg is not None else 1.0
                done_coef = getattr(wm_cfg, "done_coef", 1.0) if wm_cfg is not None else 1.0

                # Scale components if desired (defaults = 1)
                wm_loss = w_next_coef * wm_loss  # keep for back-compat scaling knobs

                self.wm_opt.zero_grad(set_to_none=True)
                wm_loss.backward()
                torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
                self.wm_opt.step()

                # Calibration (ECE) for reward and latent transitions
                try:
                    with torch.no_grad():
                        out = wm.forward(w, a)
                        # reward calibration
                        eps_r = float(getattr(wm_cfg, "calib_eps_r", 0.5) if wm_cfg is not None else 0.5)
                        sigma_r = torch.exp(0.5 * out.r_logvar).detach().cpu().numpy().reshape(-1)
                        r_err = (out.r_hat - r).detach().cpu().numpy().reshape(-1)
                        r_acc = (np.abs(r_err) <= eps_r).astype(np.float64)
                        r_conf = prob_within_eps_gaussian(eps_r, sigma_r)
                        ece_r, bins_r = expected_calibration_error(r_conf, r_acc, bins=int(getattr(wm_cfg, "calib_bins", 10) if wm_cfg is not None else 10))
                        wm_metrics["wm/calib_ece_r"] = float(ece_r)

                        # latent calibration
                        eps_w = float(getattr(wm_cfg, "calib_eps_w", 0.25) if wm_cfg is not None else 0.25)
                        frac_w = float(getattr(wm_cfg, "calib_frac_w", 0.8) if wm_cfg is not None else 0.8)
                        w_sigma = torch.exp(0.5 * out.w_next_logvar).detach().cpu().numpy()  # [B,D]
                        w_err = (out.w_next_mu - w_next).detach().cpu().numpy()              # [B,D]
                        per_dim_acc = (np.abs(w_err) <= eps_w).astype(np.float64)
                        acc_w = (per_dim_acc.mean(axis=1) >= frac_w).astype(np.float64)
                        per_dim_conf = prob_within_eps_gaussian(eps_w, w_sigma)
                        conf_w = per_dim_conf.mean(axis=1)
                        ece_w, bins_w = expected_calibration_error(conf_w, acc_w, bins=int(getattr(wm_cfg, "calib_bins", 10) if wm_cfg is not None else 10))
                        wm_metrics["wm/calib_ece_w"] = float(ece_w)

                        # stash bins for CLI artifact writing
                        self.last_wm_calibration = {
                            "reward": {
                                "ece": float(ece_r),
                                "edges": bins_r.bin_edges.tolist(),
                                "bin_conf": bins_r.bin_conf.tolist(),
                                "bin_acc": bins_r.bin_acc.tolist(),
                                "bin_count": bins_r.bin_count.tolist(),
                            },
                            "latent": {
                                "ece": float(ece_w),
                                "edges": bins_w.bin_edges.tolist(),
                                "bin_conf": bins_w.bin_conf.tolist(),
                                "bin_acc": bins_w.bin_acc.tolist(),
                                "bin_count": bins_w.bin_count.tolist(),
                            },
                        }
                except Exception:
                    pass

            # Accumulate stats
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_loss.item()
            total_memory_gate_loss += memory_gate_loss.item()
            total_write_prob += float(write_prob.mean().item())
            write_prob_min = min(
                write_prob_min,
                float(write_prob.min().item()),
            )
            write_prob_max = max(
                write_prob_max,
                float(write_prob.max().item()),
            )
            total_adv_used += float(adv_used.mean().item())
            adv_used_min = min(adv_used_min, float(adv_used.min().item()))
            adv_used_max = max(adv_used_max, float(adv_used.max().item()))
            total_write_prob_adv_corr += corr
            total_effective_write += float(effective_write_mean.item())
            total_router_do_cf += float(mb_router_do_cf.mean().item())
            num_updates += 1

    # Adaptive KL coefficient adjustment
    if cfg.use_kl_penalty and cfg.adaptive_kl:
        avg_kl = total_kl / max(num_updates, 1)
        if avg_kl > cfg.kl_target * cfg.kl_adapt_coef:
            self.kl_coef *= 1.5
        elif avg_kl < cfg.kl_target / cfg.kl_adapt_coef:
            self.kl_coef *= 0.5
        self.kl_coef = max(0.0001, min(self.kl_coef, 10.0))

    # Update old policy if using KL
    if cfg.use_kl_penalty:
        self._store_old_policy()

    self.iteration += 1

    metrics = {

        "policy_loss": total_policy_loss / num_updates,
        "value_loss": total_value_loss / num_updates,
        "entropy": total_entropy / num_updates,
        "kl": total_kl / max(num_updates, 1),
        "grad_norm": total_grad_norm / max(num_updates, 1),
        "memory_gate_loss": total_memory_gate_loss / num_updates,
        "mean_write_prob": total_write_prob / num_updates,
        "write_prob_min": write_prob_min,
        "write_prob_max": write_prob_max,
        "mean_adv_used": total_adv_used / num_updates,
        "adv_used_min": adv_used_min,
        "adv_used_max": adv_used_max,
        "router_intervention_rate": total_router_do_cf / max(num_updates, 1),
        "adapter_usage_entropy": adapter_usage_entropy,
        "adapter_selection_entropy_mean": adapter_selection_entropy_mean,
        "write_prob_adv_corr": total_write_prob_adv_corr / num_updates,
        "effective_write_mean": total_effective_write / num_updates,
        "learning_rate": self.current_lr,
        "kl_coef": self.kl_coef,
        "mean_reward": rollouts.rewards.mean().item(),
        "mean_return": rollouts.returns.mean().item(),
        "mean_advantage": rollouts.advantages.mean().item(),
    }

    # Attach per-rollout episode stats (vector-safe)
    if hasattr(self, 'last_episode_stats') and isinstance(self.last_episode_stats, dict):
        for k, v in self.last_episode_stats.items():
            metrics[f'episodes/{k}'] = float(v)

    return metrics

def evaluate(
    self,
    envs: Any,
    num_episodes: int = 5,
    deterministic: bool = True,
) -> Dict[str, float]:
    """
    Evaluate agent on environments.

    Args:
        envs: List of environments
        num_episodes: Episodes to run
        deterministic: Use deterministic policy
    Returns:
        Dict with evaluation stats
    """
    self.agent.eval()

    episode_rewards = []
    episode_lengths = []

    for ep in range(num_episodes):
        env = envs[0]  # Use first env
        if self.seed is not None:
            try:
                reset_out = env.reset(seed=int(self.seed) + int(ep))
            except TypeError:
                reset_out = env.reset()
        else:
            reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        done = False
        ep_reward = 0.0
        ep_length = 0

        while not done:
            obs_t = torch.tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            with torch.no_grad():
                action, _, _ = self.agent.get_action(
                    obs_t,
                    deterministic=deterministic,
                )

            step_out = env.step(action[0].item())
            if len(step_out) == 5:
                obs, reward, done, truncated, _ = step_out
                done = done or truncated
            else:
                obs, reward, done, _ = step_out

            ep_reward += reward
            ep_length += 1

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        self.agent.reset_episode()

    return {
        "eval_mean_reward": float(np.mean(episode_rewards)),
        "eval_std_reward": float(np.std(episode_rewards)),
        "eval_mean_length": float(np.mean(episode_lengths)),
    }

def save(self, path: str) -> None:
    """Save trainer state."""
    state = {
        "agent": self.agent.state_dict(),
        "optimizer": self.optimizer.state_dict(),
        "wm_optimizer": (self.wm_opt.state_dict() if self.wm_opt is not None else None),
        "iteration": self.iteration,
        "cfg_dict": (self.cfg.__dict__ if hasattr(self.cfg, "__dict__") else self.cfg),
    }
    if self.agent.ltm is not None:
        state["ltm"] = self.agent.ltm.state_dict_ltm()
    torch.save(state, path)

def load(self, path: str) -> None:
    """Load trainer state."""
    state = torch.load(path, map_location=self.device, weights_only=True)
    self.agent.load_state_dict(state["agent"])
    self.optimizer.load_state_dict(state["optimizer"])
    if getattr(self, "wm_opt", None) is not None and state.get("wm_optimizer") is not None:
        self.wm_opt.load_state_dict(state["wm_optimizer"])
    self.iteration = state["iteration"]
    if "ltm" in state and self.agent.ltm is not None:
        self.agent.ltm.load_state_dict_ltm(state["ltm"])
# --- Bind functional implementations onto PPOTrainer (fix accidental nesting) ---
PPOTrainer.collect_rollouts = collect_rollouts
PPOTrainer.train_step = train_step
PPOTrainer.evaluate = evaluate
PPOTrainer.save = save
PPOTrainer.load = load
