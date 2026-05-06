"""Extend outputs/anndata/full_costs.npz with SC + gene-expression costs.

After this runs, full_costs.npz contains (in addition to FC + xyz):
    Cm_SC,   Ch_SC      — within-species relational distance from SC
    Cm_gene, Ch_gene    — within-species relational distance from gene fingerprint
    M_gene              — cross-species cosine distance from ortholog vectors

Inputs (must exist in data_external/):
    mouse_sc.npy, human_sc.npy
    mouse_genes_aligned.npy, human_genes_aligned.npy
    mouse_genes.npy        (for the within-mouse gene-coexpression cost)
    human_genes.npy        (for the within-human gene-coexpression cost)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from homer.costs import (                                  # noqa: E402
    sc_correlation_distance, gene_correlation_distance, cross_species_gene_cost,
    cross_species_anchor_M, normalise_cost,
)

ANN  = ROOT / "outputs" / "anndata"
EXT  = ROOT / "data_external"


def main():
    print("loading inputs...")
    sc_m  = np.load(EXT / "mouse_sc.npy")
    sc_h  = np.load(EXT / "human_sc.npy")
    ge_m  = np.load(EXT / "mouse_genes.npy")              # full mouse gene matrix
    ge_h  = np.load(EXT / "human_genes.npy")              # full human gene matrix
    ge_m_o = np.load(EXT / "mouse_genes_aligned.npy")     # ortholog-aligned
    ge_h_o = np.load(EXT / "human_genes_aligned.npy")
    print(f"  sc_m {sc_m.shape}, sc_h {sc_h.shape}")
    print(f"  ge_m {ge_m.shape}, ge_h {ge_h.shape}, orthologs {ge_m_o.shape[1]}")

    # 1. SC relational costs (log-transformed first, since SC is heavy-tailed)
    print("computing SC correlation distances...")
    Cm_SC = normalise_cost(sc_correlation_distance(sc_m), scheme="max").astype(np.float32)
    Ch_SC = normalise_cost(sc_correlation_distance(sc_h), scheme="max").astype(np.float32)
    od = lambda c: c[~np.eye(c.shape[0], dtype=bool)]
    print(f"  Cm_SC: mean={od(Cm_SC).mean():.3f}  Ch_SC: mean={od(Ch_SC).mean():.3f}")

    # 2. Gene-coexpression relational costs (use FULL per-species matrices —
    #    we want ALL within-species genes, not just orthologs)
    print("computing gene-coexpression distances...")
    Cm_gene = normalise_cost(gene_correlation_distance(ge_m), scheme="max").astype(np.float32)
    Ch_gene = normalise_cost(gene_correlation_distance(ge_h), scheme="max").astype(np.float32)
    print(f"  Cm_gene: mean={od(Cm_gene).mean():.3f}  Ch_gene: mean={od(Ch_gene).mean():.3f}")

    # 3. Cross-species cost from ortholog vectors
    print("computing cross-species gene cost (orthologs)...")
    M_gene = cross_species_gene_cost(ge_m_o, ge_h_o).astype(np.float32)
    M_gene_norm = M_gene / max(M_gene.max(), 1e-9)
    print(f"  M_gene: shape={M_gene.shape}  range [{M_gene.min():.3f}, {M_gene.max():.3f}]")

    # 3b. Coverage mask: True where BOTH species have at least some non-NaN
    # ortholog data. Used downstream to apply M_gene only where it's meaningful;
    # outside this mask, M_gene contributes nothing (vs. our previous behaviour
    # of adding a max-cost penalty that scrambled subcortical assignments).
    valid_m = np.isfinite(ge_m_o).any(axis=1)              # (1864,) bool
    valid_h = np.isfinite(ge_h_o).any(axis=1)              # (2094,) bool
    M_gene_valid = (valid_m[:, None] & valid_h[None, :]).astype(np.uint8)
    print(f"  M_gene_valid: {M_gene_valid.sum()} cells with bilateral coverage "
          f"({M_gene_valid.mean():.1%} of {M_gene.size})")

    # 4. Save (preserving existing FC, xyz, M_xyz)
    print("merging into full_costs.npz...")
    existing = dict(np.load(ANN / "full_costs.npz"))
    # 4. Anchor-relationship cross-species cost ----------------------------
    # Each node gets a vector of FC values to each anchor; cross-species
    # distance between these vectors is meaningful because the 42 anchors are
    # in known 1-to-1 correspondence between species.
    print("computing anchor-relationship cross-species cost...")
    from homer.data import load_cached
    from homer.data.anchors import get_anchor_index
    H_ad, _ = load_cached("human", cache_dir=ANN)
    M_ad, _ = load_cached("mouse", cache_dir=ANN)
    idx_h = get_anchor_index(H_ad.var); idx_m = get_anchor_index(M_ad.var)
    fc_m = M_ad.uns["fc_mean"]; fc_h = H_ad.uns["fc_mean"]
    M_anchor = cross_species_anchor_M(fc_m, fc_h, idx_m.pos, idx_h.pos).astype(np.float32)
    M_anchor_norm = M_anchor / max(M_anchor.max(), 1e-9)
    print(f"  M_anchor: shape={M_anchor.shape} range [{M_anchor.min():.3f}, {M_anchor.max():.3f}]")

    existing.update({
        "Cm_SC":        Cm_SC,
        "Ch_SC":        Ch_SC,
        "Cm_gene":      Cm_gene,
        "Ch_gene":      Ch_gene,
        "M_gene":       M_gene_norm,
        "M_gene_valid": M_gene_valid,
        "M_anchor":     M_anchor_norm,
    })
    np.savez_compressed(ANN / "full_costs.npz", **existing)
    print(f"  saved → {ANN / 'full_costs.npz'}")
    print(f"  size: {(ANN / 'full_costs.npz').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
