"""``_dsurqe.py`` uses the live atlas lookup, not the precomputed votes.

This test locks in the reason ``mouse_parcels_in_dsurqe_region`` resolves
regions via the live DSURQE atlas volume rather than the parcel table's
precomputed DSURQE vote labels:

  - The mouse parcel table ships ``region_vote_ss_dsq``, precomputed
    DSURQE vote labels per parcel.
  - OTTER's anchor packs query ``mouse_parcels_in_dsurqe_region(M, NAME)``
    with names like "Caudoputamen", "Periaqueductal gray", "Lateral
    visual area".
  - The vote vocabulary uses DIFFERENT NAMES, coarser-grained and with
    different conventions (e.g. "striatum" instead of "Caudoputamen";
    British "periaqueductal grey" instead of American "Periaqueductal
    gray"; "Secondary visual cortex,lateral area" instead of "Lateral
    visual area").

A naïve subtree-membership check against those votes would return EMPTY
sets for most pack queries because the names don't line up; consuming
them directly needs a name-mapping table first.

This test asserts the vocabulary mismatch so that the lookup is not
switched over without addressing it.
"""
from __future__ import annotations

import importlib.util
import importlib.machinery
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO.parent / "data_crossspecies"
V2_MAT   = DATA_DIR / "updated_connectom_0906_26" / "corrs_mouse_v2.mat"


needs_v2 = pytest.mark.skipif(
    not V2_MAT.exists(), reason=f"v2 mat file not present at {V2_MAT}"
)


def _load_io():
    pkg_otter = importlib.util.module_from_spec(
        importlib.machinery.ModuleSpec("otter", None)
    )
    pkg_data = importlib.util.module_from_spec(
        importlib.machinery.ModuleSpec("otter.data", None)
    )
    sys.modules.setdefault("otter", pkg_otter)
    sys.modules.setdefault("otter.data", pkg_data)
    spec = importlib.util.spec_from_file_location(
        "otter.data.io", REPO / "src/otter/data/io.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["otter.data.io"] = mod
    spec.loader.exec_module(mod)
    return mod


@needs_v2
def test_paul_vote_vocabulary_smaller_than_pack_query_vocabulary():
    """The vote vocabulary is coarser than the pack queries, and the two do not align.

    The parcel table ships ~114 unique vote strings, the DSURQE tree
    has ~590 named nodes, and the anchor-pack query names include
    region names that do not appear in the vote vocabulary at all.
    """
    IO = _load_io()
    meta = IO.load_metadata("mouse")
    df = IO.parse_t_table(meta["t"], meta["ht"])

    votes = set(df["region_vote_ss_dsq"].fillna("").to_list())
    votes.discard("")
    # ~114 distinct votes in the v2 file.
    assert 50 < len(votes) < 200, (
        f"unexpected vote-vocabulary size {len(votes)}; "
        f"v2 baseline was 114 distinct votes."
    )

    # Common anchor-pack query names, none should be in the vote
    # vocabulary, demonstrating the naming mismatch.
    pack_query_names_that_dont_appear = {
        "Caudoputamen",        # vote is 'striatum' (parent)
        "Periaqueductal gray", # vote is 'periaqueductal grey'
        "Lateral visual area", # vote is 'Secondary visual cortex,lateral area'
        "Primary motor area",  # vote is 'Primary motor cortex'
        "Field CA1",           # votes are 'CA1Or'/'CA1Rad'/'CA1Py'
        "Visual areas",        # votes are subdivisions
    }
    misses = pack_query_names_that_dont_appear & votes
    assert misses == set(), (
        f"naming convergence detected: {misses}. The vocabulary-mismatch "
        f"finding may no longer hold."
    )


@needs_v2
def test_dsurqe_offset_constant_still_present():
    """``DSURQE_OFFSET_MM`` is defined and used.

    The constant is required while the live atlas lookup is the production
    code path.
    """
    spec = importlib.util.spec_from_file_location(
        "otter.data.anchor_packs._dsurqe",
        REPO / "src/otter/data/anchor_packs/_dsurqe.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "DSURQE_OFFSET_MM")
    assert len(mod.DSURQE_OFFSET_MM) == 3
