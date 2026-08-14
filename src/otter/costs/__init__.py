"""Cost-matrix builders for FGW: relational + cross-species + normalisation.

Re-exports the public API at the package root:
    from otter.costs import correlation_distance, cross_species_anchor_M, normalise_cost
"""
from otter.costs.relational import (
    anchor_relationship_features,
    correlation_distance,
    fisher_z_distance,
    gene_correlation_distance,
    geodesic_fc_distance,
    sc_correlation_distance,
)
from otter.costs.crossspecies import (
    cross_species_anchor_M,
    cross_species_gene_cost,
)
from otter.costs.normalisation import normalise_cost

__all__ = [
    "correlation_distance",
    "fisher_z_distance",
    "geodesic_fc_distance",
    "sc_correlation_distance",
    "gene_correlation_distance",
    "anchor_relationship_features",
    "cross_species_anchor_M",
    "cross_species_gene_cost",
    "normalise_cost",
]
