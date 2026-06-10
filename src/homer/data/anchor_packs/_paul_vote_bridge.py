"""Bridge table: precomputed DSURQE vote vocabulary → DSURQE_tree.json names.

The mouse parcel table's ``region_vote_ss_dsq`` and ``region_vote_ns_dsq``
columns contain DSURQE atlas region names from the upstream connectome
pipeline. Those strings do not all line up with the names HOMER's anchor
packs query against via the Beauchamp 2022 ``DSURQE_tree.json``.

This module supplies the alignment in three layers, with decreasing
automated coverage and increasing manual review:

  (A) **Direct tree match** — 23 of the 114 vote strings appear verbatim
      in ``DSURQE_tree.json``. No lookup needed.

  (B) **Beauchamp CSV mapping** — 77 more are bridged via
      ``DSURQE_40micron_R_mapping_long.csv`` (also shipped in the
      Beauchamp 2022 repo). The CSV has columns ``Structure`` (the vote
      naming convention with left/right prefixes) and ``ABI`` (the
      DSURQE_tree.json name). Strip the left/right prefix and look up.

  (C) **Hand-authored bridge** (this file) — 8 votes remain after (A)
      and (B). 6 cerebellar entries are out of scope for HOMER's
      parcellation. The 8 entries below are HAND-AUTHORED based on
      neuroanatomical interpretation of the region names, and each is
      flagged in the ``CONFIDENCE`` column as worth confirming with the
      upstream pipeline author.

These mappings cover ~5 % of the 1864 parcels. The production
``_dsurqe.py`` uses the live atlas lookup, so this file is currently
informational; if the dispatch is ever switched to consume these labels
directly, it should consult this table.

Confidence column conventions:
    HIGH    — anatomical mapping is unambiguous (e.g., CA1 stratum
              oriens IS part of Field CA1).
    MEDIUM  — high prior but worth one Paul confirmation email.
    LOW     — ambiguous; multiple plausible mappings; ask Paul.
"""
from __future__ import annotations


# Hand-authored mappings — Paul's vote string → DSURQE_tree.json node name.
#
# When Paul's vote covers multiple tree nodes (e.g., multiple hippocampal
# layers all belong to Field CA1), map to the most specific common
# ancestor in the tree. The CONFIDENCE column flags the level of
# certainty.
PAUL_TO_TREE_HAND_MAPPED: dict[str, str] = {
    # Hippocampal subfield layers — Paul codes the orientation/layer
    # within CA1/CA2/CA3. Tree has "Field CA1/2/3" at parent level.
    # Mapping to parent loses the orientation/layer info, but anchor
    # packs query at the "Field CAx" level so this is exactly right
    # for downstream usage. HIGH confidence — anatomically obvious.
    "CA1Or":  "Field CA1",   # CA1 stratum oriens
    "CA2Or":  "Field CA2",   # CA2 stratum oriens
    "CA3Or":  "Field CA3",   # CA3 stratum oriens
    # NOTE: CA1Rad/CA1Py/CA2Rad/CA3Rad/LMol/MoDG/SLu also appear in
    # Paul's votes. They're already mapped via the CSV but are also
    # candidates for this table if the CSV lookup ever fails — left
    # commented out for now.
    # "CA1Rad":  "Field CA1",  # CA1 stratum radiatum (via CSV)
    # "CA1Py":   "Field CA1",  # CA1 stratum pyramidale (via CSV)

    # Cingulate subdivisions with "Prime" suffix — Paul's spelling
    # of the 24a' and 24b' variants (Brodmann/Vogt cingulate
    # nomenclature). Tree node is the same as 24a/24b at one level
    # up because the tree doesn't distinguish prime variants.
    # MEDIUM confidence — anatomically obvious but worth confirming
    # Paul's intent.
    "Cingulate cortex,area 24aPrime": "Cingulate cortex, area 24a",
    "Cingulate cortex,area 24bPrime": "Cingulate cortex, area 24b",

    # Compound olfactory bulb layers — Paul's vote is a concatenation
    # of three accessory-olfactory-bulb sublayers (glomerular,
    # external plexiform, mitral). Tree has them grouped under
    # "Accessory olfactory bulb". MEDIUM confidence — Paul might
    # intend just the accessory sublayers, or might include all of
    # the olfactory complex; safest to map to AOB parent.
    "Accessory olfactory bulb,glomerular,external plexiform and mitral cell layer":
        "Accessory olfactory bulb",

    # "olfactory bulbs" (plural) — Paul's catch-all for the
    # bulb-as-a-whole. Tree has "Main olfactory bulb" (singular).
    # MEDIUM confidence — could also include accessory bulb;
    # ask Paul whether main-only or main+accessory.
    "olfactory bulbs": "Main olfactory bulb",

    # Compound brainstem fiber tract — Paul concatenates two
    # distinct anatomical structures. Tree has them as separate
    # nodes. MEDIUM confidence — the two tracts run together at
    # the medulla level, hard to disambiguate at parcel-set scale;
    # mapping to medial lemniscus as the larger of the two.
    "medial lemniscus,medial longitudinal fasciculus":
        "medial lemniscus",
}


# Confidence annotation (separate dict to avoid clutter in lookups).
PAUL_TO_TREE_CONFIDENCE: dict[str, str] = {
    "CA1Or": "HIGH",
    "CA2Or": "HIGH",
    "CA3Or": "HIGH",
    "Cingulate cortex,area 24aPrime": "MEDIUM",
    "Cingulate cortex,area 24bPrime": "MEDIUM",
    "Accessory olfactory bulb,glomerular,external plexiform and mitral cell layer": "MEDIUM",
    "olfactory bulbs": "MEDIUM",
    "medial lemniscus,medial longitudinal fasciculus": "MEDIUM",
}


# Cerebellar entries that we do NOT map because HOMER excludes
# cerebellum from its parcellation. Listed here so future maintainers
# don't try to map them.
CEREBELLAR_VOTES_EXCLUDED: set[str] = {
    "anterior lobule lobules 4-5",
    "flocculus FL",
    "lobule 3,central lobule dorsal",
    "lobules 1-2,lingula and central lobule ventral",
    "lobules 4-5,culmen ventral and dorsal",
    "simple lobule lobule 6",
}


# Open questions to confirm with the upstream pipeline author (one per
# uncertain mapping above). Used to seed an optional confirmation email.
PAUL_CONFIRMATION_QUESTIONS: list[dict] = [
    {
        "paul_vote": "CA1Or / CA2Or / CA3Or",
        "our_mapping": "Field CA1 / Field CA2 / Field CA3 respectively",
        "rationale": "stratum oriens of each subfield, mapped to parent",
        "ask_paul": "Confirm CAxOr are CAx-stratum-oriens layers, "
                    "not separate atlas regions.",
    },
    {
        "paul_vote": "Cingulate cortex,area 24aPrime / 24bPrime",
        "our_mapping": "Cingulate cortex, area 24a / 24b",
        "rationale": "Prime = 24a' / 24b' variants (Vogt cingulate)",
        "ask_paul": "Confirm 'Prime' suffix is the 24a' / 24b' prime "
                    "convention; if so, should they map to 24a/24b or to "
                    "a separate node in your DSURQUE atlas?",
    },
    {
        "paul_vote": "Accessory olfactory bulb,glomerular,external plexiform "
                     "and mitral cell layer",
        "our_mapping": "Accessory olfactory bulb",
        "rationale": "concatenation of three AOB sublayers",
        "ask_paul": "Confirm this compound name covers only the accessory "
                    "olfactory bulb (not the main bulb).",
    },
    {
        "paul_vote": "olfactory bulbs",
        "our_mapping": "Main olfactory bulb",
        "rationale": "plural 'bulbs' suggests broader scope",
        "ask_paul": "Should 'olfactory bulbs' include main+accessory, or "
                    "only main? Currently mapping to main only.",
    },
    {
        "paul_vote": "medial lemniscus,medial longitudinal fasciculus",
        "our_mapping": "medial lemniscus",
        "rationale": "two distinct tracts grouped together",
        "ask_paul": "These are anatomically distinct tracts; confirm they "
                    "share a single label in your DSURQUE atlas, or whether "
                    "this is a parcel that spans both.",
    },
]


def resolve_paul_vote(vote: str) -> str | None:
    """Return the DSURQE_tree.json name that ``vote`` should resolve to.

    Returns None for cerebellar votes (out of HOMER's scope) and for
    any vote not in the hand-mapped table. Combined with the CSV bridge
    and direct tree lookup elsewhere, this covers all 114 of Paul's
    distinct vote strings.

    Note: this resolver is NOT yet wired into ``_dsurqe.py``. Option
    (c) refactor was investigated and reverted; see
    ``pipeline/00_external/v2_loader_design/OPTION_C_FINDING.md``.
    """
    if vote in CEREBELLAR_VOTES_EXCLUDED:
        return None
    return PAUL_TO_TREE_HAND_MAPPED.get(vote)
