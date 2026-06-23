"""homer, cross-species brain region mapping via Fused Gromov–Wasserstein.

The high-level public API:

    # Models (sklearn-style fit/pi/predict/save/load):
    from homer.models import (
        UnsupervisedGW, SupervisedFGW, MultimodalFGW, HierarchicalFGW,
    )

    # Data layer:
    from homer.data import (
        load_cached, build_anndata, parse_t_table,
        get_anchor_index, NETWORKS, assign_networks,
    )

    # Evaluation:
    from homer.eval import (
        anchor_loo_cv, subject_kfold_cv, fc_translation_quality,
        random_pi_null, permuted_anchor_null, bootstrap_pi,
    )

    # Cost matrix builders (advanced):
    from homer.costs import (
        correlation_distance, sc_correlation_distance,
        cross_species_anchor_M, normalise_cost,
    )

The minimal end-to-end recipe:

    >>> from homer.data import load_cached
    >>> from homer.models import MultimodalFGW
    >>> M, _ = load_cached("mouse", cache_dir="outputs/anndata")
    >>> H, _ = load_cached("human", cache_dir="outputs/anndata")
    >>> import numpy as np
    >>> d = np.load("outputs/anndata/full_costs.npz")
    >>> model = MultimodalFGW(use_sc=True, sc_weight=0.3)
    >>> model.fit(M, H, Cm_SC=d["Cm_SC"], Ch_SC=d["Ch_SC"])
    >>> model.pi.shape
    (1864, 2094)
    >>> model.evaluate(eval_kind='translation')
"""
__version__ = "0.1.0"

from homer.models import (
    FGWModel,
    HierarchicalFGW,
    MultimodalFGW,
    SupervisedFGW,
    UnsupervisedGW,
)

__all__ = [
    "FGWModel",
    "UnsupervisedGW",
    "SupervisedFGW",
    "MultimodalFGW",
    "HierarchicalFGW",
]
