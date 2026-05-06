"""POT-backed wrappers for entropic (Fused) Gromov–Wasserstein.

Why a wrapper instead of calling POT directly:
  - Multi-restart with diverse G0 init to escape GW's local minima.
  - Consistent return type (FGWResult) across solvers.
  - Insulation point: lets us swap in OTT-JAX or moscot later without touching
    the model classes.

This module is internal — model classes in homer.models import from it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import ot


@dataclass
class FGWResult:
    pi: np.ndarray                       # (n1, n2) coupling
    loss: float                          # GW (or FGW) objective at convergence
    log: dict
    n_iter: int
    init_seed: int                       # which restart produced this result
    converged: bool


def _uniform(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n, dtype=np.float64)


def _random_g0(n1: int, n2: int, *, p: np.ndarray, q: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
    """Random valid initial coupling: rejection-free Sinkhorn-ish projection
    of a random non-negative matrix onto the marginal constraints.
    """
    A = rng.uniform(0.5, 1.5, size=(n1, n2))
    for _ in range(50):
        A = A * (p / A.sum(axis=1).clip(min=1e-12))[:, None]
        A = A * (q / A.sum(axis=0).clip(min=1e-12))[None, :]
    return A


def gw_loss(C1: np.ndarray, C2: np.ndarray, pi: np.ndarray,
            loss_fun: str = "square_loss") -> float:
    """GW objective Σ_{i,j,k,l} L(C1[i,k], C2[j,l]) π[i,j] π[k,l].

    Uses the Peyré 2016 closed-form for square_loss to avoid materialising the
    (n1, n2, n1, n2) outer-product tensor (which would be 111 TiB for our
    1864×2094 production size). The formula:

        L = Σ C1²[i,k]·p[i]·p[k] + Σ C2²[j,l]·q[j]·q[l]
            − 2·trace(C1 @ π @ C2 @ π.T)

    where p, q are the row/column marginals of π. O(n1²·n2 + n1·n2²) memory and
    compute, vs O(n1²·n2²) for the naive 4D version.
    """
    if loss_fun != "square_loss":
        raise NotImplementedError(loss_fun)
    p = pi.sum(axis=1)                           # (n1,)
    q = pi.sum(axis=0)                           # (n2,)
    term1 = float((C1 * C1 * np.outer(p, p)).sum())
    term2 = float((C2 * C2 * np.outer(q, q)).sum())
    # trace(C1 @ π @ C2 @ π.T) — cyclic: trace((π.T @ C1 @ π) @ C2)
    cross = float((C1 @ pi @ C2 * pi).sum())
    return term1 + term2 - 2.0 * cross


def entropic_gw(
    C1: np.ndarray,
    C2: np.ndarray,
    *,
    p: Optional[np.ndarray] = None,
    q: Optional[np.ndarray] = None,
    epsilon: float = 5e-3,
    loss_fun: str = "square_loss",
    max_iter: int = 1000,
    tol: float = 1e-9,
    G0: Optional[np.ndarray] = None,
    solver: str = "PGD",
) -> FGWResult:
    """Single-shot entropic GW via POT."""
    n1, n2 = C1.shape[0], C2.shape[0]
    p = _uniform(n1) if p is None else p
    q = _uniform(n2) if q is None else q
    pi, log = ot.gromov.entropic_gromov_wasserstein(
        C1.astype(np.float64), C2.astype(np.float64), p, q, loss_fun,
        epsilon=epsilon, G0=G0, max_iter=max_iter, tol=tol,
        solver=solver, log=True,
    )
    pi = np.asarray(pi, dtype=np.float64)
    # Use POT's reported loss when available (avoids re-computing). dict.get's
    # default arg is eagerly evaluated so we use an explicit if/else.
    if "gw_dist" in log:
        loss = float(log["gw_dist"])
    else:
        try:
            loss = gw_loss(C1, C2, pi, loss_fun)
        except Exception:
            loss = float("nan")
    return FGWResult(pi=pi, loss=loss, log=log, n_iter=int(log.get("n_iter", 0)),
                     init_seed=-1, converged=bool(log.get("converged", True)))


def entropic_gw_multistart(
    C1: np.ndarray,
    C2: np.ndarray,
    *,
    p: Optional[np.ndarray] = None,
    q: Optional[np.ndarray] = None,
    epsilon: float = 5e-3,
    loss_fun: str = "square_loss",
    n_restarts: int = 10,
    seeds: Optional[list[int]] = None,
    use_uniform_init: bool = True,
    **kwargs,
) -> tuple[FGWResult, list[FGWResult]]:
    """Run multiple restarts with diverse G0 init; return best (lowest loss)."""
    n1, n2 = C1.shape[0], C2.shape[0]
    p = _uniform(n1) if p is None else p
    q = _uniform(n2) if q is None else q
    seeds = seeds if seeds is not None else list(range(n_restarts))
    results: list[FGWResult] = []
    if use_uniform_init:
        r = entropic_gw(C1, C2, p=p, q=q, epsilon=epsilon, loss_fun=loss_fun, **kwargs)
        r.init_seed = -1
        results.append(r)
    for s in seeds:
        rng = np.random.default_rng(s)
        G0 = _random_g0(n1, n2, p=p, q=q, rng=rng)
        r = entropic_gw(C1, C2, p=p, q=q, epsilon=epsilon, loss_fun=loss_fun,
                        G0=G0, **kwargs)
        r.init_seed = s
        results.append(r)
    best = min(results, key=lambda r: r.loss)
    return best, results


def entropic_semirelaxed_fgw_multistart(
    M: np.ndarray,
    C1: np.ndarray,
    C2: np.ndarray,
    p: np.ndarray,
    *,
    alpha: float = 0.5,
    epsilon: float = 5e-3,
    max_iter: int = 25,
    tol: float = 1e-5,
    n_random_inits: int = 4,
    seeds: Optional[list[int]] = None,
    anchor_warm: Optional[tuple] = None,
) -> tuple[np.ndarray, dict]:
    """Multistart semirelaxed entropic FGW.

    Runs:
      1. one default uniform-init solve
      2. n_random_inits Sinkhorn-projected random G0 inits
      3. (optionally) one anchor-warmstart run

    Returns (best_pi, info_dict) with best_init / best_loss / all_losses /
    loss_range / loss_spread / n_restarts.
    """
    n_m, n_h = M.shape
    p = np.asarray(p, dtype=np.float64)
    q = np.full(n_h, 1.0 / n_h, dtype=np.float64)

    def _solve(G0_arr=None, init_name="uniform"):
        try:
            pi, log = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein(
                M=M, C1=C1, C2=C2, p=p,
                alpha=alpha, epsilon=epsilon,
                max_iter=max_iter, tol=tol, log=True,
                G0=G0_arr,
            )
            loss = float(log.get("srfgw_dist", log.get("fgw_dist", float("inf"))))
            return {"pi": pi, "loss": loss, "init": init_name}
        except Exception as e:
            return {"pi": None, "loss": float("inf"), "init": init_name, "err": str(e)}

    results = []
    results.append(_solve(None, "uniform"))

    seeds = seeds if seeds is not None else list(range(n_random_inits))
    for s in seeds:
        rng = np.random.default_rng(s)
        G0 = _random_g0(n_m, n_h, p=p, q=q, rng=rng)
        results.append(_solve(G0, f"random_seed{s}"))

    if anchor_warm is not None:
        idx_m_pos, idx_h_pos = anchor_warm
        G0 = np.outer(p, q).astype(np.float64).copy()
        for k, mp in enumerate(idx_m_pos):
            hp = idx_h_pos[k]
            G0[mp, :] *= 0.05
            G0[mp, hp] = p[mp] * 0.95
        G0 = G0 * (p / G0.sum(axis=1).clip(min=1e-12))[:, None]
        results.append(_solve(G0, "anchor_warm"))

    valid = [r for r in results if r["pi"] is not None]
    best = min(valid, key=lambda r: r["loss"])
    losses = [r["loss"] for r in valid]
    info = {
        "best_init":   best["init"],
        "best_loss":   float(best["loss"]),
        "all_losses":  [(r["init"], float(r["loss"])) for r in results],
        "loss_range":  [float(min(losses)), float(max(losses))],
        "loss_spread": float(max(losses) - min(losses)),
        "n_restarts":  len(results),
    }
    return best["pi"], info


def entropic_fgw(
    C1: np.ndarray,
    C2: np.ndarray,
    M: np.ndarray,
    *,
    p: Optional[np.ndarray] = None,
    q: Optional[np.ndarray] = None,
    alpha: float = 0.5,
    epsilon: float = 5e-3,
    loss_fun: str = "square_loss",
    max_iter: int = 1000,
    tol: float = 1e-9,
    G0: Optional[np.ndarray] = None,
    solver: str = "PGD",
) -> FGWResult:
    """Single-shot entropic Fused GW via POT.

    alpha ∈ [0, 1]: 1 → pure GW (relational only), 0 → pure W on features (M).
    """
    n1, n2 = C1.shape[0], C2.shape[0]
    p = _uniform(n1) if p is None else p
    q = _uniform(n2) if q is None else q
    pi, log = ot.gromov.entropic_fused_gromov_wasserstein(
        M=M.astype(np.float64), C1=C1.astype(np.float64), C2=C2.astype(np.float64),
        p=p, q=q, loss_fun=loss_fun, alpha=alpha, epsilon=epsilon,
        G0=G0, max_iter=max_iter, tol=tol, solver=solver, log=True,
    )
    pi = np.asarray(pi, dtype=np.float64)
    loss = float(log.get("fgw_dist", log.get("gw_dist", np.nan)))
    return FGWResult(pi=pi, loss=loss, log=log, n_iter=int(log.get("n_iter", 0)),
                     init_seed=-1, converged=bool(log.get("converged", True)))
