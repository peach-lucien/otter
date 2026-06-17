"""Data layer: I/O, anchors, network labels, and exploratory metrics.

Re-exports the public API at the package root for backward compatibility:
    from homer.data import load_cached, build_anndata, parse_t_table
    from homer.data import DATA_DIR, stream_mean_fc_subset, ...

The full sub-modules are also accessible:
    from homer.data import anchors, networks, eda, io
"""
from homer.data import anchors, eda, fetch, io, networks, region_anchors, supplementary_anchors
from homer.data.io import (
    DATA_DIR,
    build_anndata,
    load_cached,
    load_metadata,
    load_pi,
    load_struct,
    parse_t_table,
    stream_mean_fc,
    stream_mean_fc_subset,
    stream_subject_nan_stats,
    _MAT_TOPKEY,
    _mat_path,
)
from homer.data.fetch import DataNotFound, ensure_data, fetch_tier
from homer.data.anchors import (
    AnchorIndex,
    assign_parcels_to_nearest_anchor_region,
    build_xyz_weight_array,
    get_anchor_index,
    held_out_indices,
    held_out_metrics,
    held_out_metrics_graded,
    kfold_pair_ids,
    metrics_summary,
    true_assignment,
)
from homer.data.networks import (
    NETWORKS,
    PAIRID_TO_NETWORK,
    assign_networks,
    network_mismatch_mask,
)

__all__ = [
    # io
    "DATA_DIR", "build_anndata", "load_cached", "load_metadata", "load_pi",
    "load_struct", "parse_t_table", "stream_mean_fc", "stream_mean_fc_subset",
    "stream_subject_nan_stats",
    # fetch
    "DataNotFound", "ensure_data", "fetch_tier",
    # anchors
    "AnchorIndex", "get_anchor_index", "held_out_indices", "held_out_metrics",
    "held_out_metrics_graded", "kfold_pair_ids", "metrics_summary",
    "true_assignment", "assign_parcels_to_nearest_anchor_region",
    "build_xyz_weight_array",
    # networks
    "NETWORKS", "PAIRID_TO_NETWORK", "assign_networks", "network_mismatch_mask",
    # sub-modules
    "anchors", "eda", "fetch", "io", "networks",
]
