from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import math

import numpy as np


@dataclass
class CalibrationBins:
    bin_edges: np.ndarray          # [B+1]
    bin_conf: np.ndarray           # [B]
    bin_acc: np.ndarray            # [B]
    bin_count: np.ndarray          # [B]


def expected_calibration_error(conf: np.ndarray, acc: np.ndarray, bins: int = 10) -> Tuple[float, CalibrationBins]:
    """Compute Expected Calibration Error (ECE).

    Args:
        conf: predicted probabilities in [0,1], shape [N]
        acc: binary outcomes {0,1}, shape [N]
        bins: number of bins

    Returns:
        (ece, CalibrationBins)
    """
    conf = np.asarray(conf, dtype=np.float64)
    acc = np.asarray(acc, dtype=np.float64)
    conf = np.clip(conf, 0.0, 1.0)

    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_conf = np.zeros((bins,), dtype=np.float64)
    bin_acc = np.zeros((bins,), dtype=np.float64)
    bin_count = np.zeros((bins,), dtype=np.int64)

    # assign
    ids = np.minimum(np.digitize(conf, edges, right=False) - 1, bins - 1)
    ids = np.maximum(ids, 0)

    for b in range(bins):
        m = ids == b
        if not np.any(m):
            continue
        bin_count[b] = int(m.sum())
        bin_conf[b] = float(conf[m].mean())
        bin_acc[b] = float(acc[m].mean())

    n = max(int(len(conf)), 1)
    ece = 0.0
    for b in range(bins):
        if bin_count[b] == 0:
            continue
        w = bin_count[b] / n
        ece += w * abs(bin_acc[b] - bin_conf[b])

    return float(ece), CalibrationBins(edges, bin_conf, bin_acc, bin_count)


def prob_within_eps_gaussian(eps: float, sigma: np.ndarray) -> np.ndarray:
    """P(|X| <= eps) for X ~ N(0, sigma^2)."""
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = np.maximum(sigma, 1e-8)
    z = eps / (sigma * math.sqrt(2.0))
    # erf approximation via numpy
    return np.clip(np.erf(z), 0.0, 1.0)
