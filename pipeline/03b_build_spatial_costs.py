"""Build & cache the within-species *spatial* cost matrices.

Adds to outputs/anndata/full_costs.npz two new arrays:
    Cm_xyz : (1864, 1864) pairwise xyz distance, mouse, normalised to max=1
    Ch_xyz : (2094, 2094) same for human

These will be combined with the FC costs as:
    C_m = β · Cm_FC + (1-β) · Cm_xyz
to encode the spatial-smoothness prior in the GW relational term.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from homer.data import load_cached  # noqa: E402

ANN = ROOT / "outputs" / "anndata"


def pairwise_xyz_normalised(var) -> np.ndarray:
    c = var[["x", "y", "z"]].values.astype(np.float64)
    lo = c.min(0, keepdims=True); hi = c.max(0, keepdims=True)
    cn = (c - lo) / np.maximum(hi - lo, 1e-9)              # per-species [0,1]^3 cube
    sq = (cn * cn).sum(1, keepdims=True)
    d2 = sq + sq.T - 2.0 * cn @ cn.T
    d = np.sqrt(np.clip(d2, 0, None))
    np.fill_diagonal(d, 0.0)
    off = d[~np.eye(d.shape[0], dtype=bool)]
    return (d / max(float(off.max()), 1e-9)).astype(np.float32)


def main():
    H, _ = load_cached("human", cache_dir=ANN)
    M_, _ = load_cached("mouse", cache_dir=ANN)
    Cm_xyz = pairwise_xyz_normalised(M_.var)
    Ch_xyz = pairwise_xyz_normalised(H.var)
    print(f"Cm_xyz {Cm_xyz.shape} max={Cm_xyz.max()} off-diag mean={Cm_xyz[~np.eye(Cm_xyz.shape[0], dtype=bool)].mean():.3f}")
    print(f"Ch_xyz {Ch_xyz.shape} max={Ch_xyz.max()} off-diag mean={Ch_xyz[~np.eye(Ch_xyz.shape[0], dtype=bool)].mean():.3f}")

    # Augment the existing npz, preserving Cm/Ch/M_xyz
    existing = np.load(ANN / "full_costs.npz")
    np.savez_compressed(
        ANN / "full_costs.npz",
        Cm=existing["Cm"], Ch=existing["Ch"], M_xyz=existing["M_xyz"],
        Cm_xyz=Cm_xyz, Ch_xyz=Ch_xyz,
        n_m=existing["n_m"], n_h=existing["n_h"],
    )
    print(f"updated {ANN / 'full_costs.npz'}")


if __name__ == "__main__":
    main()
