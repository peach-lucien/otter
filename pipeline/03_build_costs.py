"""Pipeline step 03, build all cost matrices for the FGW solver.

Produces a single output that the production solve and every CV / validation
script reads:

    outputs/anndata/full_costs.npz

Contains:
    Within-species relational (used as C in the FGW objective):
      Cm, Ch          (n×n). FC distance, log1p + 1−r, max-normalised
      Cm_xyz, Ch_xyz  (n×n), pairwise xyz distance in [0,1] cube
      Cm_SC, Ch_SC    (n×n). SC log1p + 1−r distance (Allen summary-structure)
      Cm_gene, Ch_gene(n×n), gene-coexpression distance (full per-species)
      Cm_SC_knox      (1864,1864). Knox-augmented SC (comparative; populated
                                    by `pipeline/00_external/06_knox_sc.py`)

    Cross-species (used as M in the FGW objective):
      M_xyz   (n_m, n_h), per-species-normalised xyz distance
      M_gene  (n_m, n_h), ortholog-cosine distance
      M_gene_valid (n_m, n_h) bool, both species have ortholog data
      M_anchor (n_m, n_h), distance in anchor-relationship FC vector

    Scalars:
      n_m, n_h, node counts

Usage:
    PYTHONPATH=src python pipeline/03_build_costs.py

This script is idempotent. Re-running it overwrites `full_costs.npz` from
scratch using the upstream AnnDatas + data_external/ files.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from homer.data import load_cached                                       # noqa: E402
from homer.data.anchors import get_anchor_index                          # noqa: E402
from homer.costs import (                                                # noqa: E402
    correlation_distance, normalise_cost,
    sc_correlation_distance, gene_correlation_distance,
    cross_species_gene_cost, cross_species_anchor_M,
)

ANN = ROOT / "outputs" / "anndata"
EXT = ROOT / "data_external"
OUT = ANN / "full_costs.npz"


def _per_species_xyz_normalised(xyz: np.ndarray) -> np.ndarray:
    """Return xyz scaled into the per-species [0,1]^3 unit cube."""
    lo = xyz.min(0, keepdims=True); hi = xyz.max(0, keepdims=True)
    return (xyz - lo) / np.maximum(hi - lo, 1e-9)


def _pairwise_xyz_normalised(var) -> np.ndarray:
    """Within-species pairwise xyz distance in the per-species cube,
    max-normalised to [0, 1]."""
    cn = _per_species_xyz_normalised(var[["x", "y", "z"]].values.astype(np.float64))
    sq = (cn * cn).sum(1, keepdims=True)
    d = np.sqrt(np.clip(sq + sq.T - 2.0 * cn @ cn.T, 0, None))
    np.fill_diagonal(d, 0.0)
    off = d[~np.eye(d.shape[0], dtype=bool)]
    return (d / max(float(off.max()), 1e-9)).astype(np.float32)


def _od_mean(c: np.ndarray) -> float:
    return float(c[~np.eye(c.shape[0], dtype=bool)].mean())


def main():
    print("Loading AnnDatas…", flush=True)
    t0 = time.time()
    M, _ = load_cached("mouse", cache_dir=ANN)
    H, _ = load_cached("human", cache_dir=ANN)
    print(f"  loaded in {time.time()-t0:.1f}s "
          f"(mouse: {len(M.var)} nodes, human: {len(H.var)} nodes)")

    # 1. Within-species FC relational cost (1 − r, max-normalised)
    print("\n[1/5] FC distance (Cm, Ch)…", flush=True)
    t = time.time()
    Cm = normalise_cost(correlation_distance(M.uns["fc_mean"]), scheme="max").astype(np.float32)
    Ch = normalise_cost(correlation_distance(H.uns["fc_mean"]), scheme="max").astype(np.float32)
    print(f"  Cm{Cm.shape} off-diag mean={_od_mean(Cm):.3f}  "
          f"Ch{Ch.shape} off-diag mean={_od_mean(Ch):.3f}  "
          f"({time.time()-t:.1f}s)")

    # 2. Within-species xyz pairwise distance (Cm_xyz, Ch_xyz)
    print("\n[2/5] within-species xyz distance (Cm_xyz, Ch_xyz)…", flush=True)
    t = time.time()
    Cm_xyz = _pairwise_xyz_normalised(M.var)
    Ch_xyz = _pairwise_xyz_normalised(H.var)
    print(f"  Cm_xyz off-diag mean={_od_mean(Cm_xyz):.3f}  "
          f"Ch_xyz off-diag mean={_od_mean(Ch_xyz):.3f}  ({time.time()-t:.1f}s)")

    # 3. Cross-species xyz cost (M_xyz)
    print("\n[3/5] cross-species xyz cost (M_xyz)…", flush=True)
    t = time.time()
    cm = _per_species_xyz_normalised(M.var[["x", "y", "z"]].values.astype(np.float64))
    ch = _per_species_xyz_normalised(H.var[["x", "y", "z"]].values.astype(np.float64))
    sq_m = (cm * cm).sum(1, keepdims=True)
    sq_h = (ch * ch).sum(1, keepdims=True)
    M_xyz = np.sqrt(np.clip(sq_m + sq_h.T - 2.0 * cm @ ch.T, 0, None)).astype(np.float32)
    M_xyz = M_xyz / max(float(M_xyz.max()), 1e-9)
    print(f"  M_xyz{M_xyz.shape} mean={M_xyz.mean():.3f}  ({time.time()-t:.1f}s)")

    # 4. Within-species SC + gene relational costs
    sc_m  = np.load(EXT / "mouse_sc.npy")
    sc_h  = np.load(EXT / "human_sc.npy")
    ge_m  = np.load(EXT / "mouse_genes.npy")
    ge_h  = np.load(EXT / "human_genes.npy")
    ge_m_o = np.load(EXT / "mouse_genes_aligned.npy")
    ge_h_o = np.load(EXT / "human_genes_aligned.npy")

    print("\n[4/5] within-species SC + gene distances…", flush=True)
    t = time.time()
    Cm_SC = normalise_cost(sc_correlation_distance(sc_m), scheme="max").astype(np.float32)
    Ch_SC = normalise_cost(sc_correlation_distance(sc_h), scheme="max").astype(np.float32)
    Cm_gene = normalise_cost(gene_correlation_distance(ge_m), scheme="max").astype(np.float32)
    Ch_gene = normalise_cost(gene_correlation_distance(ge_h), scheme="max").astype(np.float32)
    print(f"  Cm_SC mean={_od_mean(Cm_SC):.3f}  Ch_SC mean={_od_mean(Ch_SC):.3f}")
    print(f"  Cm_gene mean={_od_mean(Cm_gene):.3f}  Ch_gene mean={_od_mean(Ch_gene):.3f}  "
          f"({time.time()-t:.1f}s)")

    # 5. Cross-species: gene + anchor-relationship M
    print("\n[5/5] cross-species gene + anchor-relationship M…", flush=True)
    t = time.time()
    M_gene = cross_species_gene_cost(ge_m_o, ge_h_o).astype(np.float32)
    M_gene = M_gene / max(float(M_gene.max()), 1e-9)
    valid_m = np.isfinite(ge_m_o).any(axis=1)
    valid_h = np.isfinite(ge_h_o).any(axis=1)
    M_gene_valid = (valid_m[:, None] & valid_h[None, :]).astype(np.uint8)
    print(f"  M_gene{M_gene.shape} bilateral coverage = "
          f"{M_gene_valid.sum()} cells ({M_gene_valid.mean():.1%})")

    idx_m = get_anchor_index(M.var); idx_h = get_anchor_index(H.var)
    M_anchor = cross_species_anchor_M(M.uns["fc_mean"], H.uns["fc_mean"],
                                       idx_m.pos, idx_h.pos).astype(np.float32)
    M_anchor = M_anchor / max(float(M_anchor.max()), 1e-9)
    print(f"  M_anchor{M_anchor.shape} ({time.time()-t:.1f}s)")

    # ---- Write everything to a single npz ----
    np.savez_compressed(
        OUT,
        Cm=Cm, Ch=Ch,
        Cm_xyz=Cm_xyz, Ch_xyz=Ch_xyz,
        M_xyz=M_xyz,
        Cm_SC=Cm_SC, Ch_SC=Ch_SC,
        Cm_gene=Cm_gene, Ch_gene=Ch_gene,
        M_gene=M_gene, M_gene_valid=M_gene_valid,
        M_anchor=M_anchor,
        n_m=Cm.shape[0], n_h=Ch.shape[0],
    )
    print(f"\n✓ saved → {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")
    print(f"  Note: Cm_SC_knox is added separately by "
          f"pipeline/00_external/06_knox_sc.py")


if __name__ == "__main__":
    main()
