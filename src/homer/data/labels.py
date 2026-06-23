"""Atlas label resolution for the mouse parcel table.

The mouse table ships labels in two distinct atlas vocabularies:

  - ``*_ABA`` columns contain Allen full NAME strings, e.g.
    "Anterior cingulate area ventral part layer 5". These resolve to
    Allen Mouse Brain Atlas (CCFv3) Structure IDs via the Allen API
    structure graph (graph_id=1).

  - ``*_DSURQUE`` columns contain DSURQE atlas-specific labels, e.g.
    "CA1Or", "CA1Py", "Accessory olfactory bulb,glomerular,external
    plexiform and mitral cell layer". These resolve via the Beauchamp 2022
    DSURQE_tree.json shipped at
    data_external/MouseHumanTranscriptomicSimilarity/AMBA/data/DSURQE_tree.json

The two atlases use disjoint vocabularies and disjoint ID spaces. We keep
the resolvers separate so a caller cannot accidentally look up a DSURQE
label in the Allen space (or vice-versa); each helper enumerates only its
own atlas.

Both helpers are tolerant of:
  - exact case-sensitive match (the canonical form)
  - all-commas-stripped match (Paul's MATLAB output drops commas inconsistently)
  - whitespace-trimmed match
  - case-insensitive match (last resort)

Returning ``None`` on unresolved labels, never raises. Empty-string and
None inputs return None.
"""
from __future__ import annotations

import json
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


_DATA_DIR = Path(__file__).resolve().parents[3] / "data_external"
_ALLEN_ONTOLOGY_URL = "http://api.brain-map.org/api/v2/structure_graph_download/1.json"
_ALLEN_CACHE_PATH = _DATA_DIR / "_allen_ontology_cache.json"
_DSURQE_TREE_PATH = (
    _DATA_DIR
    / "MouseHumanTranscriptomicSimilarity"
    / "AMBA"
    / "data"
    / "DSURQE_tree.json"
)


# ---------------------------------------------------------------------------
# Allen ontology
# ---------------------------------------------------------------------------

def _load_allen_ontology_raw() -> dict[str, Any]:
    """Load Allen structure graph 1 from local cache, falling back to web fetch."""
    if _ALLEN_CACHE_PATH.exists():
        try:
            return json.loads(_ALLEN_CACHE_PATH.read_text())
        except json.JSONDecodeError:
            pass  # Cache corrupt; re-fetch.
    # Fetch and cache
    with urllib.request.urlopen(_ALLEN_ONTOLOGY_URL, timeout=60) as r:
        raw = r.read().decode("utf-8")
    _ALLEN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ALLEN_CACHE_PATH.write_text(raw)
    return json.loads(raw)


@lru_cache(maxsize=1)
def _allen_name_to_id() -> dict[str, int]:
    """Map every Allen full NAME (and common variants) to its Structure ID.

    Includes the canonical name, comma-stripped version, lowercase, and
    lowercase-comma-stripped. Later variants don't overwrite earlier ones.
    """
    tree = _load_allen_ontology_raw()
    out: dict[str, int] = {}

    def walk(node: dict[str, Any]) -> None:
        nid = node.get("id")
        name = node.get("name")
        if nid is not None and name:
            # Canonical first; only fill in variants if not already set
            # (so a later region whose comma-stripped name collides doesn't win).
            out.setdefault(name, nid)
            ns = name.replace(",", "")
            out.setdefault(ns, nid)
            out.setdefault(name.lower(), nid)
            out.setdefault(ns.lower(), nid)
        for c in node.get("children") or []:
            walk(c)

    for root in tree.get("msg", []):
        walk(root)
    return out


def aba_label_to_allen_id(name: str | None) -> int | None:
    """Resolve an Allen Brain Atlas full NAME string to a Structure ID.

    Returns None for None/empty input or if the name doesn't resolve. The
    lookup is tolerant of Paul's comma-stripping (full names like
    ``"Anterior cingulate area, ventral part, layer 5"`` may arrive as
    ``"Anterior cingulate area ventral part layer 5"``) and to letter case.
    """
    if name is None:
        return None
    s = str(name).strip()
    if not s:
        return None
    table = _allen_name_to_id()
    if s in table:                       return table[s]
    if s.replace(",", "") in table:      return table[s.replace(",", "")]
    if s.lower() in table:               return table[s.lower()]
    if s.replace(",", "").lower() in table:
        return table[s.replace(",", "").lower()]
    return None


# ---------------------------------------------------------------------------
# DSURQE ontology (via Beauchamp 2022 tree)
# ---------------------------------------------------------------------------

def _load_dsurqe_tree_raw() -> dict[str, Any]:
    """Load the DSURQE_tree.json shipped with Beauchamp 2022. Raises if missing."""
    if not _DSURQE_TREE_PATH.exists():
        raise FileNotFoundError(
            f"DSURQE_tree.json not found at {_DSURQE_TREE_PATH}. "
            f"Clone Beauchamp 2022's MouseHumanTranscriptomicSimilarity into "
            f"data_external/ to enable DSURQE label lookups."
        )
    return json.loads(_DSURQE_TREE_PATH.read_text())


@lru_cache(maxsize=1)
def _dsurqe_name_to_labels() -> dict[str, list[int]]:
    """Map DSURQE region names (and variants) to their label IDs.

    Each name maps to a list because the DSURQE atlas assigns multiple
    voxel-level integer labels under one named region (e.g. left+right
    variants). Variants include the canonical name, comma-stripped,
    whitespace-collapsed, and lowercase combinations.
    """
    tree = _load_dsurqe_tree_raw()
    out: dict[str, list[int]] = {}

    def normlab(L: Any) -> list[int]:
        if not L:
            return []
        if isinstance(L, int):
            return [L]
        return [int(x) for x in L]

    def walk(node: dict[str, Any]) -> None:
        nm = node.get("name")
        labs = normlab(node.get("label"))
        if nm and labs:
            out.setdefault(nm, labs)
            ns = nm.replace(",", "")
            out.setdefault(ns, labs)
            out.setdefault(nm.lower(), labs)
            out.setdefault(ns.lower(), labs)
        for c in (node.get("children") or {}).values():
            walk(c)

    # The Beauchamp DSURQE_tree.json has structure {"msg": [root]} with root
    # holding the hierarchy. Children are dict-keyed (not list-keyed).
    for root in tree.get("msg", []):
        walk(root)
    return out


def dsurqe_label_to_id(name: str | None) -> list[int] | None:
    """Resolve a DSURQE atlas label string to its integer label IDs.

    Returns a list of label IDs (DSURQE assigns multiple voxel-level integers
    per named region). Returns None on None/empty input or if the name
    doesn't resolve.

    Tolerant of comma-stripping (Paul's strings sometimes drop commas) and
    letter case.
    """
    if name is None:
        return None
    s = str(name).strip()
    if not s:
        return None
    try:
        table = _dsurqe_name_to_labels()
    except FileNotFoundError:
        # If the DSURQE tree isn't available, return None rather than crash.
        # Callers that need the resolution can catch the
        # FileNotFoundError by calling _load_dsurqe_tree_raw() directly.
        return None
    if s in table:                       return list(table[s])
    if s.replace(",", "") in table:      return list(table[s.replace(",", "")])
    if s.lower() in table:               return list(table[s.lower()])
    if s.replace(",", "").lower() in table:
        return list(table[s.replace(",", "").lower()])
    return None


# ---------------------------------------------------------------------------
# Utilities for downstream consumers
# ---------------------------------------------------------------------------

def unravel_ns(idx: int | np.ndarray) -> np.ndarray:
    """Convenience: unravel a 0-based linear NS index → (i, j, k) ijk in NS grid.

    Returned shape: (3,) if input scalar; (n, 3) if input array. Always
    int64. Uses Fortran (column-major) order, the convention the parcel
    voxel indices follow.
    """
    from homer.data.io import _NS_SHAPE
    if np.isscalar(idx):
        ijk = np.unravel_index(int(idx), _NS_SHAPE, order="F")
        return np.array(ijk, dtype=np.int64)
    arr = np.asarray(idx, dtype=np.int64)
    return np.column_stack(np.unravel_index(arr, _NS_SHAPE, order="F")).astype(np.int64)


def unravel_ss(idx: int | np.ndarray) -> np.ndarray:
    """Convenience: unravel a 0-based linear SS index → (i, j, k) ijk in SS grid.

    Returned shape: (3,) if input scalar; (n, 3) if input array. Always
    int64. Uses Fortran (column-major) order.
    """
    from homer.data.io import _SS_SHAPE
    if np.isscalar(idx):
        ijk = np.unravel_index(int(idx), _SS_SHAPE, order="F")
        return np.array(ijk, dtype=np.int64)
    arr = np.asarray(idx, dtype=np.int64)
    return np.column_stack(np.unravel_index(arr, _SS_SHAPE, order="F")).astype(np.int64)
