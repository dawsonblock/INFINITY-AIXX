from __future__ import annotations


from contextlib import contextmanager
from typing import Any, Optional

import numpy as np
import torch

class PinnedTransfer:
    """Reusable pinned-CPU staging buffer for fast H2D copies.

    Why: torch.Tensor.to(device, non_blocking=True) only becomes truly async when the
    source tensor lives in pinned (page-locked) host memory. This class keeps one
    pinned buffer around and reuses it every step to avoid alloc churn.
    """

    def __init__(self, *, shape: tuple[int, ...], dtype: torch.dtype = torch.float32):
        self.shape = tuple(int(x) for x in shape)
        self.dtype = dtype
        self.buf: Optional[torch.Tensor] = None

    def _ensure(self, shape: tuple[int, ...]) -> torch.Tensor:
        shape = tuple(int(x) for x in shape)
        if self.buf is None or tuple(self.buf.shape) != shape or self.buf.dtype != self.dtype:
            # pinned CPU tensor
            self.buf = torch.empty(shape, dtype=self.dtype, device="cpu", pin_memory=True)
        return self.buf

    def to_device(self, arr: Any, *, device: torch.device | str) -> torch.Tensor:
        np_arr = np.asarray(arr)
        buf = self._ensure(tuple(np_arr.shape))
        src = torch.as_tensor(np_arr, dtype=self.dtype)
        buf.copy_(src, non_blocking=False)
        return buf.to(device=device, non_blocking=True)




class PinnedStager:
    """Pinned host + reusable device buffer for fixed-shape batches.

    This eliminates per-step CUDA tensor allocations:
      - copy numpy -> pinned host (via numpy view)
      - async copy pinned host -> device buffer (non_blocking=True)
      - return the device buffer (same object every step)
    """

    def __init__(self, *, shape: tuple[int, ...], dtype: torch.dtype = torch.float32):
        self.shape = tuple(int(x) for x in shape)
        self.dtype = dtype
        self.host: torch.Tensor = torch.empty(self.shape, dtype=self.dtype, device="cpu", pin_memory=True)
        self._host_np = self.host.numpy()
        self.dev: Optional[torch.Tensor] = None

    def stage(self, arr: Any, *, device: torch.device | str) -> torch.Tensor:
        np_arr = np.asarray(arr, dtype=np.float32)
        if tuple(np_arr.shape) != self.shape:
            raise ValueError(f"PinnedStager shape mismatch: expected {self.shape}, got {tuple(np_arr.shape)}")
        # numpy-to-numpy copy avoids creating a temporary torch tensor.
        np.copyto(self._host_np, np_arr, casting="unsafe")
        if str(device) == "cpu":
            return self.host
        if self.dev is None or self.dev.device != torch.device(device) or tuple(self.dev.shape) != self.shape or self.dev.dtype != self.dtype:
            self.dev = torch.empty(self.shape, dtype=self.dtype, device=device)
        self.dev.copy_(self.host, non_blocking=True)
        return self.dev


def obs_to_tensor(
    obs: Any,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    pinned: Optional[Any] = None,
) -> torch.Tensor:
    """Convert an environment observation batch to a torch Tensor on `device`.

    If `pinned` is provided and the device is CUDA, uses a pinned-host + reusable device
    buffer path (no per-step CUDA allocations).
    """
    arr = np.asarray(obs, dtype=np.float32)
    if pinned is not None and str(device) != "cpu" and torch.cuda.is_available():
        # Support both legacy PinnedTransfer and the newer PinnedStager.
        if hasattr(pinned, "stage"):
            return pinned.stage(arr, device=device)
        if hasattr(pinned, "to_device"):
            return pinned.to_device(arr, device=device)
    t = torch.as_tensor(arr, dtype=dtype)
    if str(device) != "cpu":
        return t.to(device=device, non_blocking=True)
    return t


def set_global_seed(seed: int, *, deterministic: bool = False) -> None:
    """Set numpy/torch RNG seeds.

    If deterministic=True, toggles PyTorch deterministic algorithms where possible.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older torch or unsupported ops — best effort.
            pass


@contextmanager
def maybe_autocast(enabled: bool, device: str):
    """Context manager for autocast, safe on CPU."""
    if enabled and device.startswith("cuda"):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        yield
