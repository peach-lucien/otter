"""Builders for the canonical regional correspondence entries.

Each module returns one or more :class:`RegionAnchorEntry` objects. Use
:func:`build_default_pack_entries` to obtain the 26 entries in canonical
fitting order; :mod:`otter.data.anchor_packs.registry` is the source of truth
for their membership.
"""
from otter.data.anchor_packs.biccn_motor import build_biccn_motor_region_anchors
from otter.data.anchor_packs.tectum import build_tectum_region_anchors
from otter.data.anchor_packs.olfactory import build_olfactory_region_anchors
from otter.data.anchor_packs.cingulate import build_cingulate_region_anchors
from otter.data.anchor_packs.amygdala import build_amygdala_region_anchors
from otter.data.anchor_packs.hippocampal import build_hippocampal_region_anchors
from otter.data.anchor_packs.lateral_pfc import build_lateral_pfc_region_anchors
from otter.data.anchor_packs.striatum import build_striatum_region_anchors
from otter.data.anchor_packs.entorhinal import build_entorhinal_region_anchors
from otter.data.anchor_packs.visual import build_visual_region_anchors
from otter.data.anchor_packs.pag import build_pag_region_anchors
from otter.data.anchor_packs.perirhinal import build_perirhinal_region_anchors
from otter.data.anchor_packs.auditory import build_auditory_region_anchors
from otter.data.anchor_packs.somatosensory import build_somatosensory_region_anchors
from otter.data.anchor_packs.ppc import build_ppc_region_anchors

from otter.data.anchor_packs.registry import (
    DEFAULT_PACK_NAMES,
    PACKS,
    PackSpec,
    build_default_pack_entries,
)

__all__ = [
    "build_biccn_motor_region_anchors",
    "build_tectum_region_anchors",
    "build_olfactory_region_anchors",
    "build_cingulate_region_anchors",
    "build_amygdala_region_anchors",
    "build_hippocampal_region_anchors",
    "build_lateral_pfc_region_anchors",
    "build_striatum_region_anchors",
    "build_entorhinal_region_anchors",
    "build_visual_region_anchors",
    "build_pag_region_anchors",
    "build_perirhinal_region_anchors",
    "build_auditory_region_anchors",
    "build_somatosensory_region_anchors",
    "build_ppc_region_anchors",
    # Canonical registry.
    "PACKS",
    "PackSpec",
    "DEFAULT_PACK_NAMES",
    "build_default_pack_entries",
]
