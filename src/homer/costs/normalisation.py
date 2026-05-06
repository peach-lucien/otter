"""Cost-matrix normalisation for stability across modalities + species."""
from __future__ import annotations

import numpy as np


def normalise_cost(d: np.ndarray, *, scheme: str = "max") -> np.ndarray:
    """Normalise a cost matrix to [0, 1] for stability across species.

    Schemes:
        "max"    — divide by max off-diagonal value
        "mean"   — divide by mean off-diagonal value (so mean→1)
        "median" — divide by median off-diagonal value
        "none"   — no-op
    """
    if scheme == "none":
        return d
    off = d[~np.eye(d.shape[0], dtype=bool)]
    if scheme == "max":
        s = float(off.max())
    elif scheme == "mean":
        s = float(off.mean())
    elif scheme == "median":
        s = float(np.median(off))
    else:
        raise ValueError(f"unknown scheme: {scheme}")
    return d / max(s, 1e-12)
