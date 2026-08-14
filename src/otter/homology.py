"""Aggregate the parcel-level coupling to the 21 Garin homology classes.

The coupling pi is defined over 1,864 mouse and 2,094 human parcels. Most summaries of it
are read at the level of the 21 cross-species homology classes that the Garin anchors define,
so the collapse from parcels to classes has to be defined once and used everywhere.

Public:
    GARIN_NAMES                        class id 1..21 to name
    coarse_region(var)                 assign each parcel to a class
    row_normalise(pi)                  each mouse row sums to 1
    region_aggregate(pi, M, H)         the 21 x 21 class matrix and its labels

The class assignment is nearest same-hemisphere anchor centroid. It is not interchangeable
with any other parcel-to-region mapping in this repository. Two implementations of the
collapse once gave 0.262 and 0.275 for the same diagonal mean, which is close enough to look
right and far enough apart to matter, so there is one definition here and every consumer
imports it.
"""
from __future__ import annotations

import numpy as np

# The 21 Garin homology classes (anchor_pair_id 1..21), hemisphere stripped.
GARIN_NAMES: dict[int, str] = {
    1: "mPFC", 2: "Motor/premotor", 3: "Somatosensory", 4: "Post. parietal",
    5: "Visual striate", 6: "Visual extrastriate", 7: "Auditory", 8: "Temporal (MIPT)",
    9: "Insula", 10: "Septum", 11: "Olfactory", 12: "Periarchicortex", 13: "Striatum",
    14: "Basal forebrain", 15: "Pallidum", 16: "Claustrum", 17: "Amygdala",
    18: "Hypothalamus", 19: "Thalamus", 20: "Pons", 21: "Tectum",
}


def coarse_region(var) -> np.ndarray:
    """Assign each parcel to one of the 21 classes by nearest same-hemisphere anchor centroid.

    ``var`` is the ``.var`` frame of a loaded AnnData, carrying x, y, z, hemisphere,
    garin_anchor and anchor_pair_id. Returns an integer array of class ids in 1..21.
    """
    xyz = var[["x", "y", "z"]].to_numpy()
    hemi = var["hemisphere"].astype(str).to_numpy()
    is_anchor = var["garin_anchor"].fillna(False).to_numpy().astype(bool)
    apid = var["anchor_pair_id"].to_numpy()
    out = np.zeros(len(var), dtype=int)
    for h in ("L", "R"):
        amask = is_anchor & (hemi == h) & np.isfinite(apid)
        a_xyz = xyz[amask]
        a_pid = apid[amask].astype(int)
        tmask = hemi == h
        d = ((xyz[tmask][:, None, :] - a_xyz[None, :, :]) ** 2).sum(-1)
        out[tmask] = a_pid[d.argmin(1)]
    return out


def row_normalise(pi: np.ndarray) -> np.ndarray:
    """Scale the coupling so each mouse row sums to 1."""
    return pi / pi.sum(1, keepdims=True).clip(1e-12)


def region_aggregate(pi: np.ndarray, M, H):
    """Collapse the parcel-level coupling to the 21 x 21 homology classes.

    Returns (Arow, labels). Arow[i, j] is the fraction of mouse class i's routed mass that
    lands on human class j, so each row sums to 1 and the diagonal is the self-mass.
    """
    P = row_normalise(pi)
    mc, hc = coarse_region(M.var), coarse_region(H.var)
    K = len(GARIN_NAMES)
    colbin = np.zeros((pi.shape[0], K))
    for k in range(1, K + 1):
        cols = np.where(hc == k)[0]
        if len(cols):
            colbin[:, k - 1] = P[:, cols].sum(1)
    A = np.zeros((K, K))
    for k in range(1, K + 1):
        rows = np.where(mc == k)[0]
        if len(rows):
            A[k - 1] = colbin[rows].sum(0)
    Arow = A / A.sum(1, keepdims=True).clip(1e-12)
    return Arow, [GARIN_NAMES[k] for k in range(1, K + 1)]
