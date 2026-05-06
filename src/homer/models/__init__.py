"""FGW model classes — sklearn-style fit/pi/predict/save/load API.

Four production levels (in order of complexity):
    UnsupervisedGW   — plain entropic GW on FC, no anchors, no spatial
    SupervisedFGW    — anchor-supervised semirelaxed FGW + xyz spatial
    MultimodalFGW    — adds SC, gene, M_anchor terms (the production winner)
    HierarchicalFGW  — per-network sub-solves (cleaner WN, hurts global CV)

Plus the underlying solver helpers in `homer.models._solver`.
"""
from homer.models._solver import (
    FGWResult,
    entropic_fgw,
    entropic_gw,
    entropic_gw_multistart,
    entropic_semirelaxed_fgw_multistart,
    gw_loss,
)
from homer.models.base import FGWModel
from homer.models.unsupervised import UnsupervisedGW
from homer.models.supervised import SupervisedFGW
from homer.models.multimodal import MultimodalFGW
from homer.models.hierarchical import HierarchicalFGW, hierarchical_semirelaxed_fgw

# FUGW is a comparative addition — requires fugw + torch optional deps.
# Lazy-load so importing homer.models doesn't pull in the heavy chain when
# the user hasn't installed them.
try:
    from homer.models.fugw import FUGWModel
    _HAS_FUGW = True
except ImportError:
    FUGWModel = None
    _HAS_FUGW = False

__all__ = [
    # Model classes
    "FGWModel",
    "UnsupervisedGW",
    "SupervisedFGW",
    "MultimodalFGW",
    "HierarchicalFGW",
    "FUGWModel",   # only non-None if `pip install fugw torch` succeeded
    # Solver helpers
    "FGWResult",
    "entropic_gw",
    "entropic_gw_multistart",
    "entropic_fgw",
    "entropic_semirelaxed_fgw_multistart",
    "hierarchical_semirelaxed_fgw",
    "gw_loss",
]
