"""Anchor (Garin atlas) helpers and recovery metrics.

The 42 Garin anchors per species form 21 pair_ids × 2 hemispheres. Anchors with
the same (pair_id, hemisphere) are putative cross-species homologues.

Splitting note: the *network* assignment (PAIRID_TO_NETWORK, assign_networks) lives
in `homer.data.networks`. This module contains only the anchor-index machinery
and held-out recovery metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class AnchorIndex:
    pos: np.ndarray            # positional index into adata.var / fc_mean (0-based)
    pair_ids: np.ndarray       # length-N anchor pair ids (1..21)
    hemispheres: np.ndarray    # length-N strings 'L'/'R'
    keys: list[tuple[int, str]]  # length-N (pair_id, hemi) — sorted

    def __len__(self) -> int:
        return len(self.pos)


def get_anchor_index(var: pd.DataFrame) -> AnchorIndex:
    """Extract the 42 Garin anchors from an adata.var, sorted by (pair_id, hemi)
    so the same ordering is produced for both species.
    """
    a = (
        var.loc[var["garin_anchor"]]
        .reset_index()
        .sort_values(["anchor_pair_id", "hemisphere"], kind="stable")
    )
    pos = a["numid"].astype(int).values - 1               # 0-based positional index
    pair_ids = a["anchor_pair_id"].astype(int).values
    hemis = a["hemisphere"].values
    keys = list(zip(pair_ids.tolist(), hemis.tolist()))
    return AnchorIndex(pos=pos, pair_ids=pair_ids, hemispheres=hemis, keys=keys)


def true_assignment(idx_m: AnchorIndex, idx_h: AnchorIndex) -> np.ndarray:
    """Return an (n_anchors_m,) array mapping each mouse-anchor row to its
    correct human-anchor column index."""
    if idx_m.keys != idx_h.keys:
        raise ValueError("anchor key orderings differ between species")
    return np.arange(len(idx_m), dtype=np.int64)


def top_k_accuracy(pi: np.ndarray, true_idx: np.ndarray, k: int = 1) -> float:
    """Fraction of source rows whose top-k argsort along axis=1 contains the true target."""
    n = pi.shape[0]
    top = np.argsort(-pi, axis=1)[:, :k]
    correct = sum(true_idx[i] in top[i] for i in range(n))
    return correct / n


def pair_id_accuracy(pi: np.ndarray, idx_m: AnchorIndex, idx_h: AnchorIndex) -> float:
    """Fraction of source rows whose argmax has the correct anchor_pair_id."""
    pred = pi.argmax(axis=1)
    correct = idx_m.pair_ids == idx_h.pair_ids[pred]
    return float(correct.mean())


def hemisphere_accuracy(pi: np.ndarray, idx_m: AnchorIndex, idx_h: AnchorIndex) -> float:
    """Fraction of source rows whose argmax has the correct hemisphere."""
    pred = pi.argmax(axis=1)
    correct = idx_m.hemispheres == idx_h.hemispheres[pred]
    return float(correct.mean())


def metrics_summary(pi: np.ndarray, idx_m: AnchorIndex, idx_h: AnchorIndex) -> dict[str, float]:
    """Compact one-shot metrics dict for a coupling pi between mouse and human anchors."""
    true_idx = true_assignment(idx_m, idx_h)
    return {
        "top1":       top_k_accuracy(pi, true_idx, k=1),
        "top5":       top_k_accuracy(pi, true_idx, k=5),
        "pair_id":    pair_id_accuracy(pi, idx_m, idx_h),
        "hemisphere": hemisphere_accuracy(pi, idx_m, idx_h),
        "n":          int(pi.shape[0]),
    }


# ---------------------------------------------------------------------------
# Anchor supervision via cross-species cost matrix
# ---------------------------------------------------------------------------
def anchor_supervision_cost(
    idx_m: AnchorIndex,
    idx_h: AnchorIndex,
    visible_pair_ids: Sequence[int],
    *,
    lam: float = 1.0,
) -> np.ndarray:
    """Build an (n_anchors_m, n_anchors_h) cross-species feature cost matrix M
    that encodes supervision on the visible anchor pairs.
    """
    n_m, n_h = len(idx_m), len(idx_h)
    visible = set(int(p) for p in visible_pair_ids)
    M = np.zeros((n_m, n_h), dtype=np.float64)
    for i in range(n_m):
        m_pair = int(idx_m.pair_ids[i]); m_hemi = idx_m.hemispheres[i]
        m_vis = m_pair in visible
        for j in range(n_h):
            h_pair = int(idx_h.pair_ids[j]); h_hemi = idx_h.hemispheres[j]
            h_vis = h_pair in visible
            if m_vis and h_vis:
                if (m_pair == h_pair) and (m_hemi == h_hemi):
                    M[i, j] = 0.0
                else:
                    M[i, j] = lam
            elif m_vis != h_vis:
                M[i, j] = lam
            else:
                M[i, j] = 0.0
    return M


def held_out_indices(
    idx_m: AnchorIndex,
    idx_h: AnchorIndex,
    visible_pair_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mouse_idx, human_idx) of held-out anchors (positional, into the
    42-anchor array — NOT into the full adata.var)."""
    visible = set(int(p) for p in visible_pair_ids)
    m_held = np.where(np.array([int(p) not in visible for p in idx_m.pair_ids]))[0]
    h_held = np.where(np.array([int(p) not in visible for p in idx_h.pair_ids]))[0]
    return m_held, h_held


def held_out_metrics_graded(
    pi: np.ndarray,
    idx_m: AnchorIndex,
    idx_h: AnchorIndex,
    held_out_pair_ids: Sequence[int],
    var_h=None,
) -> dict[str, float]:
    """Like held_out_metrics but returns rank- and distance-graded metrics
    alongside binary top-1.

    Adds:
        top5            — fraction of held-out anchors whose correct partner is
                          in the top-5 of argsort(π[i, :]) restricted to held-out.
        mean_rank       — mean rank of correct partner (1 = perfect; max = #held).
        median_rank     — median of the same.
        mean_xyz_dist   — mean per-species-normalised xyz distance between the
                          predicted argmax and the correct held-out anchor.
    """
    visible_pair_ids = [p for p in idx_m.pair_ids
                        if int(p) not in set(int(x) for x in held_out_pair_ids)]
    m_held, h_held = held_out_indices(idx_m, idx_h, visible_pair_ids)
    if len(m_held) == 0:
        return {"top1": float("nan"), "top5": float("nan"),
                "pair_id": float("nan"), "hemisphere": float("nan"),
                "mean_rank": float("nan"), "median_rank": float("nan"),
                "n": 0}
    sub = pi[np.ix_(m_held, h_held)]
    pred_h_local = sub.argmax(axis=1)
    pred_h_global = h_held[pred_h_local]
    correct_top1 = pred_h_global == m_held
    correct_pair = idx_m.pair_ids[m_held] == idx_h.pair_ids[pred_h_global]
    correct_hemi = idx_m.hemispheres[m_held] == idx_h.hemispheres[pred_h_global]

    order = np.argsort(-sub, axis=1)
    true_local = np.arange(len(m_held))
    ranks = np.zeros(len(m_held), dtype=np.int64)
    for i in range(len(m_held)):
        ranks[i] = int(np.where(order[i] == true_local[i])[0][0]) + 1
    top5 = float((ranks <= 5).mean())

    out = {
        "top1":        float(correct_top1.mean()),
        "top5":        top5,
        "pair_id":     float(correct_pair.mean()),
        "hemisphere":  float(correct_hemi.mean()),
        "mean_rank":   float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
        "max_rank_possible": int(len(m_held)),
        "n":           int(len(m_held)),
    }

    if var_h is not None:
        xyz = var_h[["x", "y", "z"]].values.astype(np.float64)
        lo = xyz.min(0, keepdims=True); hi = xyz.max(0, keepdims=True)
        xyz_n = (xyz - lo) / np.maximum(hi - lo, 1e-9)
        ah = xyz_n[idx_h.pos]
        true_xyz = ah[h_held[true_local]]
        pred_xyz = ah[h_held[pred_h_local]]
        d = np.linalg.norm(true_xyz - pred_xyz, axis=1)
        out["mean_xyz_dist"]   = float(d.mean())
        out["median_xyz_dist"] = float(np.median(d))
    return out


def held_out_metrics(
    pi: np.ndarray,
    idx_m: AnchorIndex,
    idx_h: AnchorIndex,
    held_out_pair_ids: Sequence[int],
) -> dict[str, float]:
    """Recovery metrics computed only on held-out anchors. argmax is restricted
    to the held-out human columns."""
    m_held, h_held = held_out_indices(
        idx_m, idx_h,
        [p for p in idx_m.pair_ids
         if int(p) not in set(int(x) for x in held_out_pair_ids)],
    )
    if len(m_held) == 0:
        return {"top1": float("nan"), "pair_id": float("nan"),
                "hemisphere": float("nan"), "n": 0}
    sub = pi[np.ix_(m_held, h_held)]
    pred_h_local = sub.argmax(axis=1)
    pred_h_global = h_held[pred_h_local]
    correct_top1 = (pred_h_global == m_held)
    correct_pair = idx_m.pair_ids[m_held] == idx_h.pair_ids[pred_h_global]
    correct_hemi = idx_m.hemispheres[m_held] == idx_h.hemispheres[pred_h_global]
    return {
        "top1":       float(correct_top1.mean()),
        "pair_id":    float(correct_pair.mean()),
        "hemisphere": float(correct_hemi.mean()),
        "n":          int(len(m_held)),
    }


def kfold_pair_ids(pair_ids: Sequence[int], n_splits: int = 5,
                   seed: int = 42) -> list[tuple[list[int], list[int]]]:
    """Yield (visible, held_out) pair_id splits for k-fold CV over pair_ids."""
    rng = np.random.default_rng(seed)
    pids = np.array(sorted(set(int(p) for p in pair_ids)))
    rng.shuffle(pids)
    folds = np.array_split(pids, n_splits)
    out = []
    for k in range(n_splits):
        held = sorted(folds[k].tolist())
        visible = sorted([int(p) for p in pids if int(p) not in set(held)])
        out.append((visible, held))
    return out
