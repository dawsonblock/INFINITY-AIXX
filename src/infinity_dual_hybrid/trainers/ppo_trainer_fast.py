"""
Fast PPO trainer (experimental).

This is an optional, simplified trainer path primarily intended for quick iteration
on small Gymnasium tasks (e.g., CartPole) when you don't need the full multitask,
memory logging, or advanced validation stack.

It uses the same Agent interface as the main PPOTrainer:
    action, log_prob, value = agent.get_action(obs_tensor)

Enable via config:
    trainer:
      kind: fast

If you need the full feature set, use the default trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from infinity_dual_hybrid.logger import UnifiedLogger, LoggerConfig

from infinity_dual_hybrid.utils import obs_to_tensor, set_global_seed



@dataclass
class RolloutBatchFast:
    observations: torch.Tensor   # [T, obs_dim] or [T, n_env, obs_dim]
    actions: torch.Tensor        # [T] or [T, n_env]
    log_probs: torch.Tensor      # [T] or [T, n_env]
    values: torch.Tensor         # [T] or [T, n_env]
    rewards: torch.Tensor        # [T] or [T, n_env]
    dones: torch.Tensor          # [T] or [T, n_env]
    advantages: torch.Tensor     # [T] or [T, n_env]
    returns: torch.Tensor        # [T] or [T, n_env]


class PPOTrainerFast:
    """
    A lean PPO trainer. Designed to be API-compatible enough for train_cli usage.
    """
    def __init__(self, agent, config: Dict[str, Any], device=None, seed: int = 0, logger: Optional[Logger] = None):
        self.agent = agent
        self.config = config
        self.device = device or getattr(agent, "device", torch.device("cpu"))
        self.logger = logger or UnifiedLogger(LoggerConfig(log_dir=str(config.get("log_dir", "runs")), experiment_name=str(config.get("run_name", "fast"))))
        self.seed = int(config.get("seed", seed))
        _set_seed(self.seed)

        lr = float(config.get("learning_rate", 3e-4))
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr)

        self.gamma = float(config.get("gamma", 0.99))
        self.gae_lambda = float(config.get("gae_lambda", 0.95))
        self.clip_coef = float(config.get("clip_coef", 0.2))
        self.entropy_coef = float(config.get("entropy_coef", 0.01))
        self.value_coef = float(config.get("value_coef", 0.5))
        self.max_grad_norm = float(config.get("max_grad_norm", 0.5))

        self.batch_size = int(config.get("batch_size", 256))
        self.update_epochs = int(config.get("update_epochs", 4))

    @torch.no_grad()
    def collect_rollouts(self, envs, num_steps: int) -> RolloutBatchFast:
        obs = envs.reset()
        if isinstance(obs, tuple):  # gymnasium reset returns (obs, info)
            obs = obs[0]

        observations = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []

        for _ in range(num_steps):
            obs_tensor = torch.from_numpy(np.asarray(obs)).float().to(self.device)
            action, log_prob, value = self.agent.get_action(obs_tensor)

            # Ensure numpy action
            a_np = action.detach().cpu().numpy()
            next_obs = envs.step(a_np)
            if len(next_obs) == 5:
                obs2, reward, terminated, truncated, _info = next_obs
                done = np.logical_or(terminated, truncated)
            else:
                obs2, reward, done, _info = next_obs

            observations.append(np.asarray(obs))
            actions.append(a_np)
            log_probs.append(log_prob.detach().cpu().numpy())
            values.append(value.detach().cpu().numpy())
            rewards.append(np.asarray(reward))
            dones.append(np.asarray(done, dtype=np.float32))

            obs = obs2

        # Bootstrap
        obs_tensor = torch.from_numpy(np.asarray(obs)).float().to(self.device)
        _a, _lp, bootstrap_value = self.agent.get_action(obs_tensor)
        bootstrap_value = bootstrap_value.detach().cpu().numpy()

        obs_t = torch.from_numpy(np.asarray(observations)).float()
        act_t = torch.from_numpy(np.asarray(actions)).long()
        lp_t = torch.from_numpy(np.asarray(log_probs)).float()
        val_t = torch.from_numpy(np.asarray(values)).float().squeeze(-1)
        rew_t = torch.from_numpy(np.asarray(rewards)).float()
        done_t = torch.from_numpy(np.asarray(dones)).float()

        adv, ret = self.compute_gae(rew_t, val_t, done_t, bootstrap_value)

        return RolloutBatchFast(
            observations=obs_t,
            actions=act_t,
            log_probs=lp_t,
            values=val_t,
            rewards=rew_t,
            dones=done_t,
            advantages=adv,
            returns=ret,
        )

    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, bootstrap_value_np) -> Tuple[torch.Tensor, torch.Tensor]:
        # supports vectorized envs
        T = rewards.shape[0]
        if rewards.ndim == 1:
            rewards = rewards.view(T, 1)
            values = values.view(T, 1)
            dones = dones.view(T, 1)
        n_env = rewards.shape[1]

        adv = torch.zeros((T, n_env), dtype=torch.float32)
        lastgaelam = torch.zeros((n_env,), dtype=torch.float32)

        # bootstrap_value_np may be scalar or vector
        if np.isscalar(bootstrap_value_np):
            next_values = torch.tensor([bootstrap_value_np] * n_env, dtype=torch.float32)
        else:
            next_values = torch.from_numpy(np.asarray(bootstrap_value_np)).float().view(-1)

        for t in reversed(range(T)):
            nextnonterminal = 1.0 - dones[t]
            nextvalue = next_values if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * nextvalue * nextnonterminal - values[t]
            lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            adv[t] = lastgaelam

        returns = adv + values
        if returns.shape[1] == 1:
            return adv.view(T), returns.view(T)
        return adv, returns

    def update(self, batch: RolloutBatchFast) -> Dict[str, float]:
        obs = batch.observations
        actions = batch.actions
        old_log_probs = batch.log_probs
        advantages = batch.advantages
        returns = batch.returns

        # flatten
        if obs.ndim == 3:
            T, N, D = obs.shape
            obs = obs.reshape(T * N, D)
            actions = actions.reshape(T * N)
            old_log_probs = old_log_probs.reshape(T * N)
            advantages = advantages.reshape(T * N)
            returns = returns.reshape(T * N)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        idxs = np.arange(obs.shape[0])
        approx_kl = 0.0
        clipfrac = 0.0

        for _ in range(self.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, len(idxs), self.batch_size):
                mb = idxs[start:start + self.batch_size]
                obs_mb = obs[mb].to(self.device)
                act_mb = actions[mb].to(self.device)
                old_lp_mb = old_log_probs[mb].to(self.device)
                adv_mb = advantages[mb].to(self.device)
                ret_mb = returns[mb].to(self.device)

                # re-evaluate
                new_action, new_log_prob, value = self.agent.get_action(obs_mb)
                # force logprob for selected action
                # (agent.get_action returns log_prob of chosen action already; ensure aligned)
                new_lp = new_log_prob

                ratio = torch.exp(new_lp - old_lp_mb)
                pg_loss1 = -adv_mb * ratio
                pg_loss2 = -adv_mb * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                value = value.view(-1)
                v_loss = 0.5 * (ret_mb - value).pow(2).mean()

                entropy = -new_lp.mean()  # proxy; for discrete policies this is not exact

                loss = pg_loss + self.value_coef * v_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl += (old_lp_mb - new_lp).mean().item()
                    clipfrac += (torch.abs(ratio - 1.0) > self.clip_coef).float().mean().item()

        approx_kl /= max(1, self.update_epochs)
        clipfrac /= max(1, self.update_epochs)
        return {
            "loss/policy": float(pg_loss.detach().cpu().item()),
            "loss/value": float(v_loss.detach().cpu().item()),
            "stats/approx_kl": float(approx_kl),
            "stats/clipfrac": float(clipfrac),
        }

    def train(self, envs, total_timesteps: int, rollout_steps: int = 2048) -> None:
        steps = 0
        while steps < total_timesteps:
            batch = self.collect_rollouts(envs, rollout_steps)
            metrics = self.update(batch)
            steps += rollout_steps
            for k, v in metrics.items():
                self.logger.log({k: v}, step=steps)
            self.logger.flush()
