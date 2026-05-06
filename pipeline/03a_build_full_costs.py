"""Build & cache the full cost matrices for the 1864 × 2094 FGW.

Outputs (single .npz to keep them coherent):
    outputs/anndata/full_costs.npz
        Cm      : (1864, 1864) float32   normalised 1-r between mouse nodes
        Ch      : (2094, 2094) float32   normalised 1-r between human nodes
        M_xyz   : (1864, 2094) float32   per-species-normalised xyz euclidean
        Cm_max, Ch_max, M_xyz_max : the original normalisation scalars
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from homer.data import load_cached                            # noqa: E402
from homer.costs import correlation_distance, normalise_cost  # noqa: E402

ANN = ROOT / "outputs" / "anndata"
OUT = ANN / "full_costs.npz"


def main():
    t = time.time()
    H, _ = load_cached("human", cache_dir=ANN)
    M_, _ = load_cached("mouse", cache_dir=ANN)
    print(f"loaded AnnDatas in {time.time()-t:.1f}s", flush=True)

    t = time.time()
    Cm = normalise_cost(correlation_distance(M_.uns["fc_mean"]), scheme="max").astype(np.float32)
    Ch = normalise_cost(correlation_distance(H.uns["fc_mean"]), scheme="max").astype(np.float32)
    print(f"Cm{Cm.shape} Ch{Ch.shape} in {time.time()-t:.1f}s "
          f"(off-diag mean: m={Cm[~np.eye(Cm.shape[0],dtype=bool)].mean():.3f} "
          f"h={Ch[~np.eye(Ch.shape[0],dtype=bool)].mean():.3f})", flush=True)

    t = time.time()
    # xyz cross-species cost — per-species normalised to [0,1]^3
    cm = M_.var[["x","y","z"]].values
    ch = H.var[["x","y","z"]].values
    cm = (cm - cm.min(0, keepdims=True)) / np.maximum(cm.max(0, keepdims=True) - cm.min(0, keepdims=True), 1e-9)
    ch = (ch - ch.min(0, keepdims=True)) / np.maximum(ch.max(0, keepdims=True) - ch.min(0, keepdims=True), 1e-9)
    sq_m = (cm * cm).sum(1, keepdims=True)
    sq_h = (ch * ch).sum(1, keepdims=True)
    M_xyz = np.sqrt(np.clip(sq_m + sq_h.T - 2.0 * cm @ ch.T, 0, None)).astype(np.float32)
    M_xyz_max = float(M_xyz.max())
    M_xyz = M_xyz / max(M_xyz_max, 1e-9)
    print(f"M_xyz{M_xyz.shape} in {time.time()-t:.1f}s (max={M_xyz_max:.3f})", flush=True)

    np.savez_compressed(
        OUT, Cm=Cm, Ch=Ch, M_xyz=M_xyz,
        n_m=Cm.shape[0], n_h=Ch.shape[0],
    )
    print(f"saved → {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
