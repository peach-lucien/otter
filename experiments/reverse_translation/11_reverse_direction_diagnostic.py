#!/usr/bin/env python3
"""Mirror the canonical semirelaxed solve and log direction diagnostics.

The production coupling fixes the mouse marginal and leaves the human marginal free.  This
diagnostic captures the exact cross-species and within-species costs passed to the canonical
solver, transposes the problem, and fixes the human marginal instead.  It writes every plotted
aggregate diagnostic quantity to
``outputs/logs/reverse_translation_direction_diagnostic.json``.

Run from ``otter/``::

    PYTHONPATH=src python experiments/reverse_translation/11_reverse_direction_diagnostic.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import ot

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from otter.data import load_pi, pi_provenance  # noqa: E402
from otter.models import MultimodalFGW  # noqa: E402
from otter.repro import (  # noqa: E402
    ALPHA,
    EPSILON,
    FC_WEIGHT,
    SC_WEIGHT,
    XYZ_WEIGHT,
    anchor_warped_xyz,
    load_inputs,
)

OUT = ROOT / "outputs/logs/reverse_translation_direction_diagnostic.json"
MIN_PARCELS_FOR_LOWEST = 4

GARIN_NAMES = {
    1: "mPFC", 2: "Motor/premotor", 3: "Somatosensory", 4: "Post. parietal",
    5: "Visual striate", 6: "Visual extrastriate", 7: "Auditory",
    8: "Temporal (MIPT)", 9: "Insula", 10: "Septum", 11: "Olfactory",
    12: "Periarchicortex", 13: "Striatum", 14: "Basal forebrain", 15: "Pallidum",
    16: "Claustrum", 17: "Amygdala", 18: "Hypothalamus", 19: "Thalamus",
    20: "Pons", 21: "Tectum",
}


def array_sha256(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
    h.update(a.tobytes())
    return h.hexdigest()


def row_normalise(a: np.ndarray) -> np.ndarray:
    return a / a.sum(1, keepdims=True).clip(1e-300)


def coarse_region(var) -> np.ndarray:
    """Nearest same-hemisphere Garin anchor class."""
    xyz = var[["x", "y", "z"]].to_numpy(float)
    hemi = var["hemisphere"].astype(str).to_numpy()
    is_anchor = var["garin_anchor"].fillna(False).to_numpy().astype(bool)
    pair_id = var["anchor_pair_id"].to_numpy()
    out = np.zeros(len(var), dtype=int)
    for side in ("L", "R"):
        anchors = is_anchor & (hemi == side) & np.isfinite(pair_id)
        targets = hemi == side
        d2 = ((xyz[targets, None, :] - xyz[anchors][None, :, :]) ** 2).sum(-1)
        out[targets] = pair_id[anchors].astype(int)[d2.argmin(1)]
    return out


def capture_canonical_geometry(M, H, costs, packs, xyz):
    """Refit canonically while intercepting the exact solver arrays."""
    real_solver = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein
    captured = {}

    def recorder(*args, **kwargs):
        for key in ("M", "C1", "C2", "p"):
            captured[key] = np.array(kwargs[key], dtype=np.float64, copy=True)
        return real_solver(*args, **kwargs)

    ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein = recorder
    try:
        model = MultimodalFGW(
            use_sc=True, sc_weight=SC_WEIGHT, fc_weight=FC_WEIGHT,
            epsilon=EPSILON, xyz_weight=XYZ_WEIGHT, lam_anchor=1.0, alpha=ALPHA,
        )
        model.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"],
                  region_anchors=packs, M_xyz=xyz)
    finally:
        ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein = real_solver
    return model.pi, captured, model.fit_info_, real_solver


def main() -> int:
    M, H, costs, packs = load_inputs()
    released = load_pi()
    fitted, geom, forward_info, solver = capture_canonical_geometry(
        M, H, costs, packs, anchor_warped_xyz(M, H)
    )
    entrywise_r = float(np.corrcoef(released.ravel(), fitted.ravel())[0, 1])
    argmax_match = float(np.mean(released.argmax(1) == fitted.argmax(1)))
    if entrywise_r < 0.999999 or argmax_match < 0.999999:
        raise RuntimeError("Captured forward solve does not reproduce the released coupling")

    n_human, n_mouse = len(H.var), len(M.var)
    reverse, reverse_log = solver(
        M=geom["M"].T, C1=geom["C2"], C2=geom["C1"],
        p=np.full(n_human, 1.0 / n_human, dtype=np.float64),
        alpha=ALPHA, epsilon=EPSILON, max_iter=25, tol=1e-5, log=True,
    )

    forward_incoming = released.sum(0) / (1.0 / n_human)
    reverse_incoming = reverse.sum(0) / (1.0 / n_mouse)

    labels = M.var["region_vote_ns_aba"].astype(object).to_numpy()
    structures = sorted(x for x in set(labels) if isinstance(x, str))
    forward_rows = row_normalise(released)
    reverse_rows = row_normalise(reverse.T)
    structure_rows = []
    for name in structures:
        mask = labels == name
        # Summing the unnormalised coupling within each structure preserves parcel mass before
        # each direction is normalised to a human distribution; this is the ED9 definition.
        f = released[mask].sum(0)
        r = reverse.T[mask].sum(0)
        f /= f.sum()
        r /= r.sum()
        structure_rows.append({
            "structure": name,
            "n_parcels": int(mask.sum()),
            "correlation": float(np.corrcoef(f, r)[0, 1]),
            "same_top_human_parcel": bool(f.argmax() == r.argmax()),
            "median_reverse_incoming_relative_to_uniform": float(np.median(reverse_incoming[mask])),
        })
    correlations = np.asarray([row["correlation"] for row in structure_rows])
    eligible = [row for row in structure_rows if row["n_parcels"] >= MIN_PARCELS_FOR_LOWEST]
    lowest = sorted(eligible, key=lambda row: row["median_reverse_incoming_relative_to_uniform"])[:12]

    mouse_class = coarse_region(M.var)
    human_class = coarse_region(H.var)
    reverse_probability = row_normalise(reverse)
    class_matrix = np.zeros((21, 21), dtype=float)
    for human_id in range(1, 22):
        hmask = human_class == human_id
        for mouse_id in range(1, 22):
            class_matrix[human_id - 1, mouse_id - 1] = reverse_probability[np.ix_(hmask, mouse_class == mouse_id)].sum()
    class_matrix = row_normalise(class_matrix)

    summary = {
        **pi_provenance(),
        "analysis": "canonical cost geometry with source/target direction transposed",
        "canonical_recipe": {
            "alpha": ALPHA, "epsilon": EPSILON, "xyz_weight": XYZ_WEIGHT,
            "fc_weight": FC_WEIGHT, "sc_weight": SC_WEIGHT, "max_iter": 25, "tol": 1e-5,
        },
        "geometry_sha256": {key: array_sha256(value) for key, value in geom.items()},
        "forward_refit": {
            "entrywise_r_with_release": entrywise_r,
            "argmax_match_with_release": argmax_match,
            "loss": float(forward_info.loss),
        },
        "reverse_fit": {
            "shape": list(reverse.shape),
            "loss": float(reverse_log.get("srfgw_dist", reverse_log.get("fgw_dist"))),
            "final_error": float(reverse_log["err"][-1]),
        },
        "incoming_mass": {
            "forward_human_relative_to_uniform": forward_incoming.tolist(),
            "reverse_mouse_relative_to_uniform": reverse_incoming.tolist(),
            "fraction_below_0.1_uniform": {
                "forward_human": float(np.mean(forward_incoming < 0.1)),
                "reverse_mouse": float(np.mean(reverse_incoming < 0.1)),
            },
        },
        "structure_agreement": {
            "label_column": "region_vote_ns_aba",
            "n_structures": len(structure_rows),
            "median_r": float(np.median(correlations)),
            "iqr_r": [float(x) for x in np.quantile(correlations, [0.25, 0.75])],
            "same_top_fraction": float(np.mean([row["same_top_human_parcel"] for row in structure_rows])),
            "rows": structure_rows,
        },
        "lowest_reverse_incoming": {
            "minimum_parcels": MIN_PARCELS_FOR_LOWEST,
            "n_eligible_structures": len(eligible),
            "rows": lowest,
        },
        "parcel_agreement": {
            "same_top_fraction": float(np.mean(forward_rows.argmax(1) == reverse_rows.argmax(1))),
            "forward_median_top_mass": float(np.median(forward_rows.max(1))),
        },
        "reverse_garin_classes": {
            "labels": [GARIN_NAMES[i] for i in range(1, 22)],
            "matrix_human_rows_mouse_columns": class_matrix.tolist(),
            "mean_self_mass": float(np.diag(class_matrix).mean()),
            "uniform_expectation": 1.0 / 21.0,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"forward <0.1 uniform: {summary['incoming_mass']['fraction_below_0.1_uniform']['forward_human']:.3f}")
    print(f"reverse <0.1 uniform: {summary['incoming_mass']['fraction_below_0.1_uniform']['reverse_mouse']:.3f}")
    print(f"structure agreement: median r={np.median(correlations):.3f}, n={len(correlations)}")
    print(f"reverse class self-mass: {np.diag(class_matrix).mean():.3f}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
