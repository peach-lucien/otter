"""Subject-bootstrap utility for the point-anchor configurations defined below.

These aggregates do not include the canonical regional correspondence entries and must not be used as uncertainty estimates for pi_canonical.npy."""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import ot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from otter.data import _MAT_TOPKEY, _mat_path, load_cached                  # noqa: E402
from otter.data.anchors import get_anchor_index                             # noqa: E402

ANN  = ROOT / "outputs" / "anndata"
PI   = ROOT / "outputs" / "coupling"; PI.mkdir(parents=True, exist_ok=True)
LOG  = ROOT / "outputs" / "logs"

warnings.filterwarnings("ignore")


CONFIGS = {
    "fc_plus_SC": dict(use_sc=True,  fc_weight=0.7, sc_weight=0.3,
                        xyz_weight=0.5, alpha=0.5, epsilon=5e-3,
                        max_iter=25, tol=1e-5),
    "fc_only":    dict(use_sc=False, fc_weight=1.0, sc_weight=0.0,
                        xyz_weight=0.5, alpha=0.5, epsilon=5e-3,
                        max_iter=25, tol=1e-5),
}


def stream_mean_fc_weighted(species: str, subj_idx: np.ndarray) -> np.ndarray:
    """Mean FC under bootstrap weights = bincount(subj_idx)."""
    p = _mat_path(species, None)
    top = _MAT_TOPKEY[species]
    with h5py.File(str(p), "r") as f:
        rr = f[f"{top}/rr"]
        n_subj_total, n_nodes, _ = rr.shape
        block = rr.chunks[1] if rr.chunks else 256
        weights = np.bincount(subj_idx.astype(np.int64),
                              minlength=n_subj_total).astype(np.float32)
        sum_fc = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        cnt    = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        for b in range((n_nodes + block - 1) // block):
            j0, j1 = b * block, min((b + 1) * block, n_nodes)
            cd = rr[:, :, j0:j1].astype(np.float32, copy=False)
            valid = ~np.isnan(cd)
            cd_z  = np.where(valid, cd, 0.0)
            X = cd_z.reshape(n_subj_total, -1)
            sum_fc[:, j0:j1] += (weights @ X).reshape(n_nodes, j1 - j0).astype(np.float64)
            V = valid.astype(np.float32).reshape(n_subj_total, -1)
            cnt[:, j0:j1]    += (weights @ V).reshape(n_nodes, j1 - j0).astype(np.float64)
    mu = (sum_fc / np.maximum(cnt, 1)).astype(np.float32)
    mu[cnt == 0] = np.nan
    return mu


def correlation_distance_norm(fc: np.ndarray) -> np.ndarray:
    d = 1.0 - fc
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    if not np.isfinite(d).all():
        m = np.nanmean(d[~np.eye(d.shape[0], dtype=bool)])
        d = np.where(np.isfinite(d), d, m)
    off = d[~np.eye(d.shape[0], dtype=bool)]
    return (d / max(float(off.max()), 1e-12)).astype(np.float64)


def build_M_full(M_xyz, idx_m, idx_h, *, lam=1.0, xyz_w=0.5):
    M = (xyz_w * M_xyz).astype(np.float64)
    for k, mp in enumerate(idx_m.pos):
        M[mp, :] = lam; M[mp, idx_h.pos[k]] = 0.0
    for k, hp in enumerate(idx_h.pos):
        M[M[:, hp] < lam, hp] = lam; M[idx_m.pos[k], hp] = 0.0
    return M


def solve_one(seed: int, idx_m, idx_h, M_xyz_norm, Cm_SC, Ch_SC,
              cfg: dict) -> np.ndarray:
    """Fit one point-anchor bootstrap sample for a named configuration."""
    rng = np.random.default_rng(seed)
    n_h_subj = 113; n_m_subj = 105
    h_idx = rng.choice(n_h_subj, size=n_h_subj, replace=True)
    m_idx = rng.choice(n_m_subj, size=n_m_subj, replace=True)
    fc_h = stream_mean_fc_weighted("human", h_idx)
    fc_m = stream_mean_fc_weighted("mouse", m_idx)
    Cm_FC = correlation_distance_norm(fc_m)
    Ch_FC = correlation_distance_norm(fc_h)
    if cfg["use_sc"]:
        Cm = cfg["fc_weight"] * Cm_FC + cfg["sc_weight"] * Cm_SC.astype(np.float64)
        Ch = cfg["fc_weight"] * Ch_FC + cfg["sc_weight"] * Ch_SC.astype(np.float64)
    else:
        Cm, Ch = Cm_FC, Ch_FC
    M = build_M_full(M_xyz_norm, idx_m, idx_h, xyz_w=cfg["xyz_weight"])
    p = np.full(Cm.shape[0], 1.0 / Cm.shape[0])
    pi, _ = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein(
        M=M, C1=Cm, C2=Ch, p=p,
        alpha=cfg["alpha"], epsilon=cfg["epsilon"],
        max_iter=cfg["max_iter"], tol=cfg["tol"], log=True,
    )
    return pi.astype(np.float32)


def load_state(state_path: Path):
    if state_path.exists():
        d = np.load(state_path)
        return {"sum_pi": d["sum_pi"], "sum_pi2": d["sum_pi2"],
                "sum_argmax_match": d["sum_argmax_match"],
                "n_done": int(d["n_done"]), "seeds_done": d["seeds_done"].tolist()}
    return None


def save_state(state, state_path: Path):
    np.savez_compressed(state_path,
        sum_pi=state["sum_pi"].astype(np.float32),
        sum_pi2=state["sum_pi2"].astype(np.float32),
        sum_argmax_match=state["sum_argmax_match"].astype(np.int32),
        n_done=state["n_done"],
        seeds_done=np.asarray(state["seeds_done"], dtype=np.int64),
    )


def main(args):
    cfg = CONFIGS[args.config]
    print(f"Bootstrap config: {args.config}  {cfg}")

    H, _ = load_cached("human", cache_dir=ANN)
    M_, _ = load_cached("mouse", cache_dir=ANN)
    idx_h = get_anchor_index(H.var); idx_m = get_anchor_index(M_.var)
    n_m, n_h = M_.uns["n_nodes"], H.uns["n_nodes"]

    # Cache paths are per-config to avoid mixing FC-only and FC+SC state
    state_path = PI  / f"bootstrap_state_{args.config}.npz"
    agg_path   = PI  / f"bootstrap_aggregate_{args.config}.npz"
    summ_path  = LOG / f"bootstrap_summary_{args.config}.json"

    if args.report:
        st = load_state(state_path)
        if st is None or st["n_done"] == 0:
            print(f"no bootstrap state at {state_path}"); return
        n = st["n_done"]
        mean_pi = st["sum_pi"] / n
        std_pi  = np.sqrt(np.maximum(st["sum_pi2"] / n - mean_pi ** 2, 0))
        argmax_freq = st["sum_argmax_match"] / n
        # Compare each bootstrap argmax with the matching saved configuration.
        ref_path = PI / f"pi_{args.config}.npy"
        if not ref_path.exists():
            print(f"reference π not found: {ref_path}; using mean_pi argmax instead.")
            ref_argmax = mean_pi.argmax(axis=1)
        else:
            pi_ref = np.load(ref_path).astype(np.float64)
            ref_argmax = pi_ref.argmax(axis=1)
        per_row_stability = argmax_freq[np.arange(n_m), ref_argmax]
        # Cell-level stability  (stability of every (i, j) cell; coupling is mostly hard so
        # most cells are 0 most of the time, which is "perfectly stable at 0")
        s_max = max(std_pi.max(), 1e-9)
        cell_stability = (1.0 - std_pi / s_max).astype(np.float32)
        summary = {
            "config":                  args.config,
            "n_iterations":            int(n),
            # Fraction of bootstrap argmaxes matching the saved configuration.
            "argmax_row_stability_mean":   float(per_row_stability.mean()),
            "argmax_row_stability_median": float(np.median(per_row_stability)),
            "argmax_row_frac_above_0.8":   float((per_row_stability > 0.8).mean()),
            "argmax_row_frac_above_0.5":   float((per_row_stability > 0.5).mean()),
            "argmax_row_frac_perfect":     float((per_row_stability == 1.0).mean()),
            # Per-cell stability
            "cell_stability_mean":         float(cell_stability.mean()),
            "cell_stability_median":       float(np.median(cell_stability)),
            "cell_frac_stable_above_0.8":  float((cell_stability > 0.8).mean()),
        }
        np.savez_compressed(agg_path,
            mean_pi=mean_pi.astype(np.float32),
            std_pi=std_pi.astype(np.float32),
            argmax_freq=argmax_freq.astype(np.float32),
            per_row_stability=per_row_stability.astype(np.float32),
            n_iter=n,
        )
        summ_path.write_text(json.dumps(summary, indent=2))
        print(f"saved aggregate → {agg_path}")
        print(f"saved summary   → {summ_path}")
        for k, v in summary.items():
            print(f"  {k:30s}: {v}")
        return

    # Build M_xyz_norm + SC costs once
    d = np.load(ANN / "full_costs.npz")
    M_xyz_norm = d["M_xyz"].astype(np.float64)
    Cm_SC = d["Cm_SC"] if cfg["use_sc"] else None
    Ch_SC = d["Ch_SC"] if cfg["use_sc"] else None

    st = load_state(state_path)
    if st is None:
        st = {"sum_pi":  np.zeros((n_m, n_h), dtype=np.float64),
              "sum_pi2": np.zeros((n_m, n_h), dtype=np.float64),
              "sum_argmax_match": np.zeros((n_m, n_h), dtype=np.int32),
              "n_done": 0, "seeds_done": []}
    next_seed = max(st["seeds_done"]) + 1 if st["seeds_done"] else 1000

    for k in range(args.iters):
        seed = next_seed + k
        t = time.time()
        pi = solve_one(seed, idx_m, idx_h, M_xyz_norm, Cm_SC, Ch_SC, cfg).astype(np.float64)
        st["sum_pi"]  += pi
        st["sum_pi2"] += pi * pi
        argmax_idx = pi.argmax(axis=1)
        st["sum_argmax_match"][np.arange(n_m), argmax_idx] += 1
        st["seeds_done"].append(seed)
        st["n_done"]  += 1
        elapsed = time.time() - t
        print(f"  seed={seed} done in {elapsed:.1f}s; n_done={st['n_done']}", flush=True)
        save_state(st, state_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="fc_plus_SC", choices=list(CONFIGS.keys()))
    ap.add_argument("--iters",  type=int, default=2,
                    help="iterations to add this run (each ~14-18s)")
    ap.add_argument("--report", action="store_true",
                    help="dump aggregate stats; don't run new iterations")
    main(ap.parse_args())
