"""Bridge table: precomputed DSURQE vote vocabulary → DSURQE_tree.json names.

The mouse parcel table's ``region_vote_ss_dsq`` and ``region_vote_ns_dsq``
columns contain DSURQE atlas region names from the upstream connectome
pipeline. Those strings do not all line up with the names OTTER's anchor
packs query against via the Beauchamp 2022 ``DSURQE_tree.json``.

This module supplies the alignment in three layers, with decreasing
automated coverage and increasing manual review:

  (A) **Direct tree match**, 23 of the 114 vote strings appear verbatim
      in ``DSURQE_tree.json``. No lookup needed.

  (B) **Beauchamp CSV mapping**, 77 more are bridged via
      ``DSURQE_40micron_R_mapping_long.csv`` (also shipped in the
      Beauchamp 2022 repo). The CSV has columns ``Structure`` (the vote
      naming convention with left/right prefixes) and ``ABI`` (the
      DSURQE_tree.json name). Strip the left/right prefix and look up.

  (C) **Hand-authored bridge** (this file), 8 votes remain after (A)
      and (B). 6 cerebellar entries are out of scope for OTTER's
      parcellation. The 8 entries below are hand-authored from
      neuroanatomical interpretation and checked against the DSURQE atlas.
      Note the cingulate ``24aPrime`` / ``24bPrime`` votes are the distinct
      midcingulate (MCC) areas ``24a'`` / ``24b'``, separate regions from
      the anterior-cingulate ``24a`` / ``24b``, mapped to their own tree nodes.

These mappings cover ~5 % of the 1864 parcels. The production
``_dsurqe.py`` uses the live atlas lookup, so this file is currently
informational; if the dispatch is ever switched to consume these labels
directly, it should consult this table.

Confidence column conventions:
    CONFIRMED, checked against the DSURQE atlas.
    HIGH, anatomical mapping is unambiguous.
    MEDIUM, high prior, worth a second check.
    LOW, ambiguous; multiple plausible mappings.
"""
from __future__ import annotations


# Hand-authored mappings. Paul's vote string → DSURQE_tree.json node name.
#
# When Paul's vote covers multiple tree nodes (e.g., multiple hippocampal
# layers all belong to Field CA1), map to the most specific common
# ancestor in the tree. The CONFIDENCE column flags the level of
# certainty.
PAUL_TO_TREE_HAND_MAPPED: dict[str, str] = {
    # Hippocampal subfield layers, these code the stratum oriens layer
    # within CA1/CA2/CA3. The tree has "Field CAx" at parent level (with
    # a "Field CAx, stratum oriens" leaf beneath). Mapping to the parent
    # field is exactly right because anchor packs query at the "Field CAx"
    # level ("Or" is the stratum oriens layer of the same field, not a
    # separate region).
    "CA1Or":  "Field CA1",   # CA1 stratum oriens
    "CA2Or":  "Field CA2",   # CA2 stratum oriens
    "CA3Or":  "Field CA3",   # CA3 stratum oriens
    # NOTE: CA1Rad/CA1Py/CA2Rad/CA3Rad/LMol/MoDG/SLu also appear in
    # Paul's votes. They're already mapped via the CSV but are also
    # candidates for this table if the CSV lookup ever fails, left
    # commented out for now.
    # "CA1Rad":  "Field CA1",  # CA1 stratum radiatum (via CSV)
    # "CA1Py":   "Field CA1",  # CA1 stratum pyramidale (via CSV)

    # Cingulate "Prime" subdivisions, the 24a' / 24b' variants are the
    # midcingulate (MCC) areas, distinct regions from the anterior-cingulate
    # 24a/24b in both the DSURQE atlas and the tree; they must not be folded
    # into 24a/24b. The tree carries them as separate nodes
    # "Cingulate cortex: area 24a'" / "24b'", so the votes map there.
    "Cingulate cortex,area 24aPrime": "Cingulate cortex: area 24a'",
    "Cingulate cortex,area 24bPrime": "Cingulate cortex: area 24b'",

    # Compound accessory-olfactory-bulb layers (glomerular, external
    # plexiform, mitral). Tree groups them under "Accessory olfactory
    # bulb". This compound label is specific to the accessory olfactory
    # bulb only, it does not include the main bulb, so the AOB parent is
    # the correct target.
    "Accessory olfactory bulb,glomerular,external plexiform and mitral cell layer":
        "Accessory olfactory bulb",

    # "olfactory bulbs" (plural), the bulb as a whole. Maps to the main
    # olfactory bulb only (the atlas carries several separate OB regions;
    # the accessory bulb is not implied by this label).
    "olfactory bulbs": "Main olfactory bulb",

    # Compound brainstem fibre tract, a single region in the DSURQE atlas
    # (the two named tracts share one label), so the single tree node
    # "medial lemniscus" is the correct target.
    "medial lemniscus,medial longitudinal fasciculus":
        "medial lemniscus",
}


# Confidence annotation (separate dict to avoid clutter in lookups).
PAUL_TO_TREE_CONFIDENCE: dict[str, str] = {
    "CA1Or": "CONFIRMED",
    "CA2Or": "CONFIRMED",
    "CA3Or": "CONFIRMED",
    "Cingulate cortex,area 24aPrime": "CONFIRMED",
    "Cingulate cortex,area 24bPrime": "CONFIRMED",
    "Accessory olfactory bulb,glomerular,external plexiform and mitral cell layer": "CONFIRMED",
    "olfactory bulbs": "CONFIRMED",
    "medial lemniscus,medial longitudinal fasciculus": "CONFIRMED",
}


# Cerebellar entries that we do NOT map because OTTER excludes
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


def resolve_paul_vote(vote: str) -> str | None:
    """Return the DSURQE_tree.json name that ``vote`` should resolve to.

    Returns None for cerebellar votes (out of OTTER's scope) and for
    any vote not in the hand-mapped table. Combined with the CSV bridge
    and direct tree lookup elsewhere, this covers all 114 distinct vote
    strings.

    Note: this resolver is not wired into the production lookup, which
    resolves regions via the live DSURQE atlas volume (see ``_dsurqe.py``).
    """
    if vote in CEREBELLAR_VOTES_EXCLUDED:
        return None
    return PAUL_TO_TREE_HAND_MAPPED.get(vote)
