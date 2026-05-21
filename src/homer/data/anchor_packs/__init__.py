"""Modular cross-species anchor *packs*.

Each pack is a small, self-contained module that exposes a single
``build_<name>_region_anchors(M_var, H_var, ...)`` function returning a
list of :class:`RegionAnchorEntry` objects. Pack modules:

  - ``biccn_motor`` — Bakken 2021 (BICCN) M1 / BA4 + M2 / PMd
  - ``tectum``      — Superior + Inferior Colliculus (Isa 2021; Winer & Schreiner 2005)
  - ``olfactory``   — Piriform cortex + Anterior olfactory nucleus (Mori 2014)
  - ``cingulate``   — Subgenual ACC + Retrosplenial (Vogt 2012)
  - ``amygdala``    — Cortical subplate / amygdala (Janak & Tye 2015)
  - ``hippocampal`` — Subiculum + CA1 + CA3 + Dentate gyrus (Strange 2014)
  - ``lateral_pfc`` — OFC + dlPFC (Wallis 2011; Carlén 2017; **dlPFC contested**)
  - ``striatum``    — Caudoputamen dorsolateral/ventromedial (Voorn 2004)
  - ``entorhinal``  — Entorhinal cortex (Franjic 2021)
  - ``visual``      — Mouse LM ↔ Human V2 (Wang & Burkhalter 2007)
  - ``pag``         — Periaqueductal gray (Ezra 2015; Kingsbury 2011)
  - ``perirhinal``  — Perirhinal cortex (Burwell 1995)
  - ``auditory``    — Mouse A1 + A2 ↔ human auditory core + belt (Hackett 2001)
  - ``somatosensory`` — S1 face/hand/leg body-map (Penfield 1937; Seelke 2012)
  - ``ppc``         — Posterior parietal cortex / BA7 (Whitlock 2017)

Packs are designed to compose: pass the concatenation of several packs as
``region_anchors=...`` to ``MultimodalFGW.fit`` to layer multiple sources
of supervision. Each pack reserves a non-overlapping ``pair_id`` range so
the entries stay distinguishable in logs and trust maps. Pid registry:

  ============= ============================================
  pid range     pack
  ============= ============================================
  1..21         Garin point anchors (original 21 pair_ids)
  30, 31        BICCN motor (M1, M2)
  32, 33        Tectum (Superior + Inferior Colliculus)
  34, 35        Olfactory (Piriform + Anterior olfactory nucleus)
  36, 37        Cingulate (Subgenual ACC + Retrosplenial) — opt-in
  38            Amygdala (Cortical subplate)
  39, 40, 41, 42  Hippocampal (Subiculum, CA1, CA3, Dentate gyrus)
  45, 46        Lateral PFC (OFC + dlPFC) — opt-in for dlPFC
  47, 48        Striatum (CP dorsolateral, CP ventromedial)
  49            Entorhinal cortex
  52            Visual extrastriate (LM ↔ V2)
  54            Periaqueductal gray (PAG)
  55            Perirhinal cortex
  56, 57        Auditory core + belt
  58, 59, 60    Somatosensory body-map (face, hand, leg) — opt-in (hurts Beauchamp S1 by 5 pp)
  61            Posterior parietal cortex
  ============= ============================================

The systematic atlas-derived pack
(:func:`homer.data.atlas_regions.build_garin_region_anchors_from_atlases`,
pid range 31..51 with ``pid_offset=30``) covers all 21 Garin pairs at once
and lives separately because it's a single object rather than per-region.
For per-region curation (literature-derived single homology pairs), use
the modules here.

Which packs make up the recommended model
-----------------------------------------
:mod:`homer.data.anchor_packs.registry` is the single source of truth for
which packs are composed into the recommended π
(``pi_fc_plus_SC_with_all_packs.npy``). Use :func:`build_default_pack_entries`
rather than re-listing the default pack builders by hand.

Designing a new pack
--------------------
Create a new file ``my_region.py`` in this directory exposing one
function ``build_my_region_anchors(M_var, H_var) -> list[RegionAnchorEntry]``.
Pick a pair_id range above 33 that doesn't clash with the existing packs.
Add the import below.

See :file:`biccn_motor.py` for the canonical pattern.
"""
from homer.data.anchor_packs.biccn_motor import build_biccn_motor_region_anchors
from homer.data.anchor_packs.tectum import build_tectum_region_anchors
from homer.data.anchor_packs.olfactory import build_olfactory_region_anchors
from homer.data.anchor_packs.cingulate import build_cingulate_region_anchors
from homer.data.anchor_packs.amygdala import build_amygdala_region_anchors
from homer.data.anchor_packs.hippocampal import build_hippocampal_region_anchors
from homer.data.anchor_packs.lateral_pfc import build_lateral_pfc_region_anchors
from homer.data.anchor_packs.striatum import build_striatum_region_anchors
from homer.data.anchor_packs.entorhinal import build_entorhinal_region_anchors
from homer.data.anchor_packs.visual import build_visual_region_anchors
from homer.data.anchor_packs.pag import build_pag_region_anchors
from homer.data.anchor_packs.perirhinal import build_perirhinal_region_anchors
from homer.data.anchor_packs.auditory import build_auditory_region_anchors
from homer.data.anchor_packs.somatosensory import build_somatosensory_region_anchors
from homer.data.anchor_packs.ppc import build_ppc_region_anchors

from homer.data.anchor_packs.registry import (
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
    # Pack registry — single source of truth for the recommended composition.
    "PACKS",
    "PackSpec",
    "DEFAULT_PACK_NAMES",
    "build_default_pack_entries",
]
