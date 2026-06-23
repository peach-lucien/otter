"""Cross-check: live DSURQE atlas lookup vs. the precomputed region votes.

HOMER's anchor packs resolve "which mouse parcels are in DSURQE region R?"
with a **live atlas lookup** (production default): each parcel's centroid is
placed into the Beauchamp DSURQE label volume and the majority label in a
~1 mm neighbourhood is read off (see
``src/homer/data/anchor_packs/_dsurqe.py::assign_dsurqe_labels``).

The mouse parcel table also ships a **precomputed vote** per parcel
(``region_vote_ss_dsq``, the majority DSURQE label over the parcel's full
voxel set, computed upstream). This script quantifies how the two compare,
so the choice of default is grounded in numbers rather than intuition.

It is a maintainer diagnostic, intentionally **not** part of the user-facing
notebooks. Self-contained: needs only numpy / h5py / nibabel + stdlib (no
pandas / anndata / POT), so it runs without the full solver stack.

What it reports
---------------
* vote-resolution coverage, what fraction of votes can even be mapped to a
  DSURQE_tree.json node (via direct match, the Beauchamp CSV bridge, and the
  hand-authored ``_paul_vote_bridge`` table);
* per anchor-pack region query: |live set|, |vote set|, intersection, Jaccard;
* a CSV of the per-region numbers for inspection.

Usage
-----
    PYTHONPATH=src python experiments/dsurqe_lookup_crosscheck/compare_lookup_vs_votes.py \
        [--h5ad outputs/anndata/mouse.h5ad] [--out outputs/logs/lookup_vs_votes.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ATLAS = ROOT / "data_external/MouseHumanTranscriptomicSimilarity/AMBA/data"
TREE_PATH = ATLAS / "DSURQE_tree.json"
VOL_PATH = ATLAS / "imaging/DSURQE_CCFv3_labels_200um.mnc"
CSV_PATH = ATLAS / "imaging/DSURQE_40micron_R_mapping_long.csv"
PACKS_DIR = ROOT / "src/homer/data/anchor_packs"

# Mouse-coord -> DSURQE-volume offset, mirrored from _dsurqe.DSURQE_OFFSET_MM
# (kept inline so this diagnostic has no heavy imports; _dsurqe.py is the
# source of truth).
DSURQE_OFFSET_MM = np.array([-0.027, -2.334, 1.018])


def _norm(s: str) -> str:
    """Comma/case/whitespace-tolerant key, matching homer.data.labels."""
    return re.sub(r"\s+", " ", s.lower().replace(",", " ")).strip()


def load_tree(path: Path):
    tree = json.loads(path.read_text())

    def normlab(L):
        return [] if not L else ([int(L)] if isinstance(L, int) else [int(x) for x in L])

    nodes = []

    def walk(node):
        nodes.append({"name": node.get("name"), "labels": set(normlab(node.get("label")))})
        for c in (node.get("children") or {}).values():
            walk(c)

    walk(tree["msg"][0])
    name_to_lbl = {x["name"]: x["labels"] for x in nodes if x["labels"] and x["name"]}
    treenames = {x["name"] for x in nodes if x["name"]}
    return name_to_lbl, treenames


def load_mouse(h5ad: Path):
    with h5py.File(h5ad, "r") as f:
        def col(name):
            o = f[f"var/{name}"]
            if isinstance(o, h5py.Group):  # categorical
                cats = [c.decode() if isinstance(c, bytes) else c for c in o["categories"][()]]
                return np.array([cats[i] if i >= 0 else "" for i in o["codes"][()]])
            v = o[()]
            return np.array([x.decode() if isinstance(x, bytes) else x for x in v])
        xyz = np.column_stack([col("x"), col("y"), col("z")]).astype(float)
        votes = col("region_vote_ss_dsq")
    return xyz, votes


def live_labels(xyz: np.ndarray, radius: int = 2) -> np.ndarray:
    """Replicate _dsurqe.assign_dsurqe_labels: centroid + neighbourhood majority."""
    img = nib.load(str(VOL_PATH))
    lab = np.asarray(img.get_fdata()).astype(np.int32)
    sh = lab.shape
    inv = np.linalg.inv(img.affine)
    n = len(xyz)
    ijk = np.round((inv @ np.c_[xyz + DSURQE_OFFSET_MM, np.ones(n)].T).T[:, :3]).astype(int)
    out = np.zeros(n, dtype=np.int32)
    for p in range(n):
        i, j, k = ijk[p]
        blk = lab[max(0, i - radius):i + radius + 1,
                  max(0, j - radius):j + radius + 1,
                  max(0, k - radius):k + radius + 1].ravel()
        nz = blk[blk > 0]
        if len(nz):
            out[p] = Counter(nz.tolist()).most_common(1)[0][0]
    return out


def _load_bridge():
    """Load _paul_vote_bridge by file path, bypassing the package __init__
    (which imports the heavy solver stack). The bridge module itself is pure."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_paul_vote_bridge", PACKS_DIR / "_paul_vote_bridge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_vote_resolver(treenames):
    """Resolve a vote string -> DSURQE_tree node name (direct / CSV / hand)."""
    _b = _load_bridge()
    PAUL_TO_TREE_HAND_MAPPED = _b.PAUL_TO_TREE_HAND_MAPPED
    CEREBELLAR_VOTES_EXCLUDED = _b.CEREBELLAR_VOTES_EXCLUDED
    tnorm = {_norm(t): t for t in treenames}
    csvmap = {}
    if CSV_PATH.exists():
        with open(CSV_PATH) as fh:
            for r in csv.DictReader(fh):
                s = r["Structure"]
                for pre in ("left ", "right "):
                    if s.lower().startswith(pre):
                        s = s[len(pre):]
                if r["ABI"] in treenames:
                    csvmap.setdefault(_norm(s), r["ABI"])

    def resolve(v: str):
        if v in CEREBELLAR_VOTES_EXCLUDED:
            return None
        if v in treenames:
            return v
        if v in PAUL_TO_TREE_HAND_MAPPED:
            return PAUL_TO_TREE_HAND_MAPPED[v]
        if _norm(v) in tnorm:
            return tnorm[_norm(v)]
        if _norm(v) in csvmap:
            return csvmap[_norm(v)]
        return None

    return resolve


def discover_region_queries():
    """Scan the pack builders for the DSURQE region names they query."""
    qs = set()
    pat = re.compile(r"mouse_parcels_in_dsurqe_region\([A-Za-z_]+,\s*\"([^\"]+)\"")
    for p in PACKS_DIR.glob("*.py"):
        qs.update(pat.findall(p.read_text()))
    return sorted(qs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default=str(ROOT / "outputs/anndata/mouse.h5ad"))
    ap.add_argument("--out", default=str(ROOT / "outputs/logs/lookup_vs_votes.csv"))
    args = ap.parse_args()

    name_to_lbl, treenames = load_tree(TREE_PATH)
    xyz, votes = load_mouse(Path(args.h5ad))
    n = len(votes)
    print(f"parcels: {n}")

    live_lbl = live_labels(xyz)
    resolve = build_vote_resolver(treenames)
    vote_node = np.array([resolve(v) or "" for v in votes])
    cov = float((vote_node != "").mean())
    print(f"vote resolution coverage: {cov:.1%} "
          f"({int((vote_node == '').sum())} of {n} votes unmapped)")

    queries = discover_region_queries()
    rows = []
    print(f"\n{'region query':44s} {'live':>5s} {'vote':>5s} {'∩':>5s} {'Jacc':>6s}")
    for R in queries:
        rl = name_to_lbl.get(R)
        if not rl:
            print(f"{R:44s}  (not a labelled tree node)")
            continue
        live = set(np.where(np.isin(live_lbl, list(rl)))[0])
        vote = {i for i, vn in enumerate(vote_node)
                if vn and name_to_lbl.get(vn) and name_to_lbl[vn] <= rl}
        inter, uni = len(live & vote), len(live | vote)
        jac = inter / uni if uni else float("nan")
        print(f"{R:44s} {len(live):5d} {len(vote):5d} {inter:5d} {jac:6.2f}")
        rows.append({"region": R, "n_live": len(live), "n_vote": len(vote),
                     "n_intersection": inter, "jaccard": round(jac, 4)})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["region", "n_live", "n_vote",
                                           "n_intersection", "jaccard"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")
    js = [r["jaccard"] for r in rows if r["jaccard"] == r["jaccard"]]
    if js:
        zero = sum(1 for j in js if j == 0)
        print(f"median Jaccard {np.median(js):.2f}; {zero}/{len(js)} region queries "
              f"the votes cannot select at all (too coarse / unmapped).")


if __name__ == "__main__":
    main()
