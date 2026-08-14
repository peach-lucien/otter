"""Single source of truth for anchor-pack composition.

Every consumer imports :data:`DEFAULT_PACK_NAMES` /
:func:`build_default_pack_entries` from this module, so the recommended
composition has one definition.

The recommended model, ``outputs/coupling/pi_fc_plus_SC_with_all_packs.npy``,
is composed from every pack flagged ``default=True`` below: all 15 packs, 26
region-anchor entries. In a multi-benchmark comparison the full set leads on
the TransBrain literature-homology benchmark and ties for best on Beauchamp,
while a smaller set leads on Beauchamp alone and by a narrow margin,
reflecting that benchmark's coarse validation balls.

Adding or removing a pack from the recommended model is a one-line change here
(flip ``default``); the compose script, the GUI builder and the multi-source
trust step all read the result.

Changing :data:`DEFAULT_PACK_NAMES` changes which packs the recommended π is
fitted with. After any such change, re-run
``experiments/anchor_packs/compose_all.py`` (or
``pipeline/run_recommended_model.py``) so the saved π matches this registry.
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
        Whether this pack is part of the recommended all-packs composition
        (``pi_fc_plus_SC_with_all_packs.npy``).
    note : str
        What the pack covers and any metric trade-off.
    """
    name: str
    builder: Callable
    default: bool
    note: str = ""


# ---------------------------------------------------------------------------
# The registry. ``default=True`` packs, in this order, are composed into the
# recommended π. The order determines pair_id ordering in the fit and must
# stay stable unless the coupling is re-fitted.
#
# All 15 packs are in the recommended composition; see the module docstring
# for the multi-benchmark rationale. A few carry trade-offs, noted below.
# ---------------------------------------------------------------------------
PACKS: dict[str, PackSpec] = {
    "biccn_motor": PackSpec(
        "biccn_motor", build_biccn_motor_region_anchors, default=True,
        note="M1 + M2/PMd (Bakken 2021). Lifts Beauchamp Primary motor 0->100%."),
    "tectum": PackSpec(
        "tectum", build_tectum_region_anchors, default=True,
        note="Superior + Inferior Colliculus (Isa 2021). Lifts SC + IC 0->100%."),
    "olfactory": PackSpec(
        "olfactory", build_olfactory_region_anchors, default=True,
        note="Piriform + anterior olfactory nucleus (Mori 2014). Lifts Piriform 0->100%."),
    "amygdala": PackSpec(
        "amygdala", build_amygdala_region_anchors, default=True,
        note="Cortical subplate / amygdala (Janak & Tye 2015). Lifts amygdala 0->100%."),
    "hippocampal": PackSpec(
        "hippocampal", build_hippocampal_region_anchors, default=True,
        note="Subiculum + CA1 + CA3 + DG (Strange 2014). Lifts all 4 subfields 0->100%."),
    "cingulate": PackSpec(
        "cingulate", build_cingulate_region_anchors, default=True,
        note="Subgenual ACC + retrosplenial (Vogt 2012). Trade-off: shifts Beauchamp "
             "ACG 13%->9% (anchor target = subgenual, validation target = pregenual)."),
    "lateral_pfc": PackSpec(
        "lateral_pfc", build_lateral_pfc_region_anchors, default=True,
        note="OFC only (Wallis 2011). The Prelimbic->dlPFC entry is excluded by "
             "default, contested (Preuss 1995) and contradicted by the Schaeffer "
             "2020 falsification test; pass include_dlpfc=True for ablations."),
    "striatum": PackSpec(
        "striatum", build_striatum_region_anchors, default=True,
        note="Caudoputamen dorsolateral/ventromedial (Voorn 2004). Lifts Beauchamp "
             "Caudoputamen 12%->33%."),
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
        note="A1 core + belt (Hackett 2001). Lifts Beauchamp Primary auditory 22%->100%."),
    "somatosensory": PackSpec(
        "somatosensory", build_somatosensory_region_anchors, default=True,
        note="Face/hand/leg S1 body-map (Penfield 1937; Seelke 2012). Trade-off: shifts "
             "Beauchamp S1 20%->15% (face/leg anchors sit outside the validation ball)."),
    "ppc": PackSpec(
        "ppc", build_ppc_region_anchors, default=True,
        note="Posterior parietal cortex / BA7 (Whitlock 2017)."),
}


# Ordered list of the packs in the recommended all-packs composition.
DEFAULT_PACK_NAMES: list[str] = [n for n, s in PACKS.items() if s.default]


def build_default_pack_entries(M_var, H_var, *, atlas_root="."):
    """Build the region-anchor entries for the recommended all-packs π.

    Concatenates, in registry order, the entries from every pack with
    ``default=True``. This is the composition fitted into
    ``outputs/coupling/pi_fc_plus_SC_with_all_packs.npy``.

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
