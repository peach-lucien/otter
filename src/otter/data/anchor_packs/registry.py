"""Single source of truth for the canonical regional-entry composition.

The canonical coupling uses every entry flagged ``default=True`` below:
15 modules and 26 regional correspondence entries. Consumers should call
:func:`build_default_pack_entries` rather than duplicating this list.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from otter.data.anchor_packs.amygdala import build_amygdala_region_anchors
from otter.data.anchor_packs.auditory import build_auditory_region_anchors
from otter.data.anchor_packs.biccn_motor import build_biccn_motor_region_anchors
from otter.data.anchor_packs.cingulate import build_cingulate_region_anchors
from otter.data.anchor_packs.entorhinal import build_entorhinal_region_anchors
from otter.data.anchor_packs.hippocampal import build_hippocampal_region_anchors
from otter.data.anchor_packs.lateral_pfc import build_lateral_pfc_region_anchors
from otter.data.anchor_packs.olfactory import build_olfactory_region_anchors
from otter.data.anchor_packs.pag import build_pag_region_anchors
from otter.data.anchor_packs.perirhinal import build_perirhinal_region_anchors
from otter.data.anchor_packs.ppc import build_ppc_region_anchors
from otter.data.anchor_packs.somatosensory import build_somatosensory_region_anchors
from otter.data.anchor_packs.striatum import build_striatum_region_anchors
from otter.data.anchor_packs.tectum import build_tectum_region_anchors
from otter.data.anchor_packs.visual import build_visual_region_anchors


@dataclass(frozen=True)
class PackSpec:
    """Registry entry for one anchor pack.

    Attributes
    ----------
    name : str
        Pack identifier (matches the module name in ``otter.data.anchor_packs``).
    builder : Callable
        ``build_<name>_region_anchors(M_var, H_var, *, atlas_root=...)``.
    default : bool
        Whether this pack is part of the canonical composition
        (``pi_canonical.npy``).
    note : str
        What the pack covers and any metric trade-off.
    """
    name: str
    builder: Callable
    default: bool
    note: str = ""


# ---------------------------------------------------------------------------
# Canonical entries in fitting order. Changing the order or membership requires
# a new coupling and regenerated dependent analyses.
# ---------------------------------------------------------------------------
PACKS: dict[str, PackSpec] = {
    "biccn_motor": PackSpec(
        "biccn_motor", build_biccn_motor_region_anchors, default=True,
        note="M1 + M2/PMd (Bakken 2021)."),
    "tectum": PackSpec(
        "tectum", build_tectum_region_anchors, default=True,
        note="Superior + Inferior Colliculus (Isa 2021)."),
    "olfactory": PackSpec(
        "olfactory", build_olfactory_region_anchors, default=True,
        note="Piriform + anterior olfactory nucleus (Mori 2014)."),
    "amygdala": PackSpec(
        "amygdala", build_amygdala_region_anchors, default=True,
        note="Cortical subplate / amygdala (Janak & Tye 2015)."),
    "hippocampal": PackSpec(
        "hippocampal", build_hippocampal_region_anchors, default=True,
        note="Subiculum + CA1 + CA3 + DG (Strange 2014)."),
    "cingulate": PackSpec(
        "cingulate", build_cingulate_region_anchors, default=True,
        note="Subgenual ACC + retrosplenial (Vogt 2012). The anchor target is "
             "subgenual; the anterior-cingulate validation target is pregenual."),
    "lateral_pfc": PackSpec(
        "lateral_pfc", build_lateral_pfc_region_anchors, default=True,
        note="OFC only (Wallis 2011). The Prelimbic->dlPFC entry is excluded by "
             "default, contested (Preuss 1995) and contradicted by the Schaeffer "
             "2020 falsification test; pass include_dlpfc=True for ablations."),
    "striatum": PackSpec(
        "striatum", build_striatum_region_anchors, default=True,
        note="Caudoputamen dorsolateral/ventromedial (Voorn 2004)."),
    "entorhinal": PackSpec(
        "entorhinal", build_entorhinal_region_anchors, default=True,
        note="Entorhinal cortex (Franjic 2021)."),
    "visual": PackSpec(
        "visual", build_visual_region_anchors, default=True,
        note="Mouse LM -> human V2 extrastriate (Wang & Burkhalter 2007)."),
    "pag": PackSpec(
        "pag", build_pag_region_anchors, default=True,
        note="Periaqueductal gray (Ezra 2015)."),
    "perirhinal": PackSpec(
        "perirhinal", build_perirhinal_region_anchors, default=True,
        note="Perirhinal cortex (Burwell 1995)."),
    "auditory": PackSpec(
        "auditory", build_auditory_region_anchors, default=True,
        note="A1 core + belt (Hackett 2001)."),
    "somatosensory": PackSpec(
        "somatosensory", build_somatosensory_region_anchors, default=True,
        note="Face/hand/leg S1 body-map (Penfield 1937; Seelke 2012). The face and leg "
             "anchors sit outside the S1 validation ball."),
    "ppc": PackSpec(
        "ppc", build_ppc_region_anchors, default=True,
        note="Posterior parietal cortex / BA7 (Whitlock 2017)."),
}


# Ordered list of modules in the canonical composition.
DEFAULT_PACK_NAMES: list[str] = [n for n, s in PACKS.items() if s.default]


def build_default_pack_entries(M_var, H_var, *, atlas_root="."):
    """Build the regional correspondence entries for the canonical coupling.

    Concatenates, in registry order, the entries from every pack with
    ``default=True``. This is the composition fitted into
    ``outputs/coupling/pi_canonical.npy``.

    Parameters
    ----------
    M_var, H_var : pandas.DataFrame
        ``.var`` tables of the mouse and human AnnData.
    atlas_root : path-like
        Repository root, forwarded to each pack builder so it can locate the
        Beauchamp 2022 DSURQE atlas under ``data_external/``.

    Returns
    -------
    list[RegionAnchorEntry]
        Concatenated region-anchor entries, ready to pass as
        ``region_anchors=`` to ``MultimodalFGW.fit``.
    """
    entries: list = []
    for name in DEFAULT_PACK_NAMES:
        entries += PACKS[name].builder(M_var, H_var, atlas_root=atlas_root)
    return entries
