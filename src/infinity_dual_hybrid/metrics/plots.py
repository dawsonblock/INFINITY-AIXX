from __future__ import annotations

from typing import Optional, Dict, Any, Sequence
import json
import os

import numpy as np

from .calibration import CalibrationBins


def _maybe_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def save_reliability_diagram(out_png: str, bins: CalibrationBins, title: str = "Reliability Diagram") -> bool:
    """Save a reliability diagram PNG. Returns False if matplotlib is unavailable."""
    plt = _maybe_import_matplotlib()
    if plt is None:
        return False

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    x = bins.bin_conf
    y = bins.bin_acc
    # only plot non-empty bins
    m = bins.bin_count > 0
    x = x[m]
    y = y[m]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot([0,1],[0,1], linestyle="--")
    ax.plot(x, y, marker="o")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title(title)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return True


def save_metric_timeseries(out_png: str, steps: Sequence[float], series: Dict[str, Sequence[float]], title: str) -> bool:
    """Save a simple timeseries plot. Returns False if matplotlib is unavailable."""
    plt = _maybe_import_matplotlib()
    if plt is None:
        return False
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for k, ys in series.items():
        ax.plot(steps, ys, label=k)
    ax.set_xlabel("global_step")
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return True
