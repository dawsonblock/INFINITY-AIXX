# src/infinity_dual_hybrid/trace.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


def _sh(x: Any) -> str:
    if x is None:
        return "None"
    if torch.is_tensor(x):
        s = "x".join(map(str, x.shape))
        return f"{s} {str(x.dtype).replace('torch.', '')} {x.device.type}"
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__}[{len(x)}]"
    if isinstance(x, dict):
        return f"dict[{len(x)}]"
    return type(x).__name__


def _stat(x: Any) -> str:
    if not torch.is_tensor(x) or x.numel() == 0:
        return ""
    with torch.no_grad():
        xf = x.float()
        m = torch.nanmean(xf).item()
        s = torch.nanstd(xf).item()
        mn = torch.nanmin(xf).item()
        mx = torch.nanmax(xf).item()
    return f" μ={m:.4g} σ={s:.4g} min={mn:.4g} max={mx:.4g}"


@dataclass
class TraceCfg:
    enabled: bool = False
    every_n_steps: int = 25
    every_n_updates: int = 1
    max_lines_per_print: int = 60
    print_tensors_stats: bool = True
    show_cuda_mem: bool = False


class Tracer:
    def __init__(self, cfg: TraceCfg):
        self.cfg = cfg
        self._roll_step = 0
        self._update_step = 0
        self._lines: list[str] = []

    @staticmethod
    def from_env() -> "Tracer":
        enabled = os.environ.get("INFINITY_TRACE", "0") == "1"
        every_n_steps = int(os.environ.get("INFINITY_TRACE_EVERY", "25"))
        show_cuda_mem = os.environ.get("INFINITY_TRACE_CUDA", "0") == "1"
        cfg = TraceCfg(enabled=enabled, every_n_steps=every_n_steps, show_cuda_mem=show_cuda_mem)
        return Tracer(cfg)

    def _cuda_mem(self) -> str:
        if not self.cfg.show_cuda_mem or not torch.cuda.is_available():
            return ""
        a = torch.cuda.memory_allocated() / (1024 ** 2)
        r = torch.cuda.memory_reserved() / (1024 ** 2)
        return f" | cuda MB alloc={a:.0f} reserv={r:.0f}"

    def _maybe_flush(self, force: bool = False) -> None:
        if not self.cfg.enabled:
            self._lines.clear()
            return
        if not force and len(self._lines) < self.cfg.max_lines_per_print:
            return
        print("\n".join(self._lines))
        self._lines.clear()

    def _p(self, msg: str) -> None:
        if not self.cfg.enabled:
            return
        self._lines.append(msg)
        self._maybe_flush(False)

    # --- rollout tracing ---
    def rollout_begin(self, env_id: str, num_envs: int, rollout_steps: int) -> None:
        self._p(f"[rollout] begin env={env_id} num_envs={num_envs} steps={rollout_steps}{self._cuda_mem()}")

    def rollout_step(self, obs: Any, agent_out: Dict[str, Any], action: Any, reward: Any, done: Any) -> None:
        if not self.cfg.enabled:
            return
        self._roll_step += 1
        if (self._roll_step % self.cfg.every_n_steps) != 0:
            return

        self._p(f"[rollout] t={self._roll_step}{self._cuda_mem()}")
        self._p(f"  obs: {_sh(obs)}")
        for k in ("encoded", "fused", "logits", "value", "write_prob", "effective_write_mean"):
            if k in agent_out:
                x = agent_out[k]
                line = f"  {k}: {_sh(x)}"
                if self.cfg.print_tensors_stats:
                    line += _stat(x)
                self._p(line)

        self._p(f"  action: {_sh(action)}{_stat(action) if self.cfg.print_tensors_stats else ''}")
        self._p(f"  reward: {_sh(reward)}{_stat(reward) if self.cfg.print_tensors_stats else ''}")
        self._p(f"  done: {_sh(done)}{_stat(done) if self.cfg.print_tensors_stats else ''}")
        self._maybe_flush(force=True)

    # --- update tracing ---
    def update_begin(self, ppo_epoch: int, num_minibatches: int) -> None:
        if not self.cfg.enabled:
            return
        self._update_step += 1
        if (self._update_step % self.cfg.every_n_updates) != 0:
            return
        self._p(f"[update] begin update={self._update_step} ppo_epoch={ppo_epoch} minibatches={num_minibatches}{self._cuda_mem()}")

    def update_minibatch(
        self,
        mb_idx: int,
        obs: Any,
        actions: Any,
        adv: Any,
        returns: Any,
        eval_out: Dict[str, Any],
        losses: Dict[str, Any],
        wm_losses: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.cfg.enabled:
            return
        if (self._update_step % self.cfg.every_n_updates) != 0:
            return

        self._p(f"[update] mb={mb_idx}{self._cuda_mem()}")
        self._p(f"  obs: {_sh(obs)}")
        self._p(f"  actions: {_sh(actions)}")
        self._p(f"  adv: {_sh(adv)}{_stat(adv) if self.cfg.print_tensors_stats else ''}")
        self._p(f"  returns: {_sh(returns)}{_stat(returns) if self.cfg.print_tensors_stats else ''}")

        for k in ("logits", "value", "entropy", "logp"):
            if k in eval_out:
                x = eval_out[k]
                line = f"  eval.{k}: {_sh(x)}"
                if self.cfg.print_tensors_stats:
                    line += _stat(x)
                self._p(line)

        for k, v in losses.items():
            self._p(f"  loss.{k}: {v:.6g}" if isinstance(v, (float, int)) else f"  loss.{k}: {v}")

        if wm_losses:
            for k, v in wm_losses.items():
                self._p(f"  wm.{k}: {v:.6g}" if isinstance(v, (float, int)) else f"  wm.{k}: {v}")

        self._maybe_flush(force=True)
