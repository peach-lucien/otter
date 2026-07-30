"""Pipeline 05f, external validation against Beauchamp 2022's curated mouse↔human pairs.

Tests whether the production π predicts homologies that match published
cross-species region correspondences from Beauchamp et al. 2022 eLife.

Beauchamp's `create_neuro_pairs.R` hard-codes 36 canonical pairs. We use the
22 non-cerebellar pairs (we exclude cerebellum from our parcellation).

Mouse side: each Beauchamp mouse region name -> set of DSURQE label IDs ->
overlap with our 1864 parcels via spatial mapping into the
`DSURQE_CCFv3_labels_200um.mnc` volume (origin offset (-0.027, -2.334, +1.018)
estimated from 6 anchor pairs whose DSURQE leaf IDs are unambiguous).

Human side: each Beauchamp human region name -> our parcels whose `subregion`
string contains a curated keyword (e.g. "Heschl's gyrus" -> "Primary Auditory
Cortex"). Some Beauchamp regions (CA1/2/3, dentate gyrus, medulla) aren't in
our anchor vocabulary; those pairs are skipped with a note.

Output: outputs/logs/beauchamp_validation.json with per-pair recovery metrics:
  - n_mouse_parcels, n_human_parcels  (size of each region's parcel set)
  - top1, top5, top10                 (does our argmax / top-K include any
                                         parcel in the human region set?)
  - mean_xyz_dist                     (distance from our argmax to human
                                         region centroid)
  - mean_rank_in_region                (rank of best human-region parcel)

Aggregate report grouped by anchor-vs-novel pair.

Usage:
    PYTHONPATH=src python pipeline/05f_beauchamp_validation.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from homer.data import load_cached                                   # noqa: E402
from homer.data.anchors import get_anchor_index                       # noqa: E402

ANN  = ROOT / "outputs" / "anndata"
LOG  = ROOT / "outputs" / "logs"; LOG.mkdir(parents=True, exist_ok=True)
EXT  = ROOT / "data_external" / "MouseHumanTranscriptomicSimilarity"
COUP = ROOT / "outputs" / "coupling"


# Beauchamp's 22 non-cerebellar pairs (DSURQE mouse name, AHBA human name)
BEAUCHAMP_PAIRS = [
    ("Claustrum",                              "claustrum"),
    ("Piriform area",                          "piriform cortex"),
    ("Subiculum",                              "subiculum"),
    ("Field CA1",                              "CA1 field"),
    ("Field CA2",                              "CA2 field"),
    ("Field CA3",                              "CA3 field"),
    ("Dentate gyrus",                          "dentate gyrus"),
    ("Anterior cingulate area",                "cingulate gyrus"),
    ("Primary auditory area",                  "Heschl's gyrus"),
    ("Primary motor area",                     "precentral gyrus"),
    ("Primary somatosensory area",             "postcentral gyrus"),
    ("Visual areas",                           "cuneus"),
    ("Pallidum",                               "globus pallidus"),
    ("Striatum ventral region",                "nucleus accumbens"),
    ("Caudoputamen",                           "caudate nucleus"),
    ("Cortical subplate-other",                "amygdala"),
    ("Inferior colliculus",                    "inferior colliculus"),
    ("Superior colliculus, sensory related",   "superior colliculus"),
    ("Medulla",                                "myelencephalon"),
    ("Pons",                                   "pons"),
    ("Hypothalamus",                           "hypothalamus"),
    ("Thalamus",                               "thalamus"),
]

# Curated MNI152 centroid coordinates (mm) for each Beauchamp human region,
# given as (x, y, z) of the LEFT hemisphere centroid (right side mirrored
# automatically by negating x). Coordinates from standard published atlases
# (AAL3, Harvard-Oxford subcortical, AHBA structure ontology, hippocampal
# subfields atlas). Radius_mm controls the parcel-membership ball.
#
# `None` means the region isn't in our atlas (we excluded medulla; no specific
# centroid was needed for some subdivisions because we group them).
HUMAN_REGION_MNI = {
    # name:                    (x_left, y, z, radius_mm)
    "claustrum":               (-32,   0,   0,   8),
    "piriform cortex":         (-25,   5, -20,  10),
    "subiculum":               (-22, -32,  -8,   8),
    "CA1 field":               (-30, -25, -10,   8),
    "CA2 field":               (-25, -22, -10,   6),
    "CA3 field":               (-25, -22, -10,   8),
    "dentate gyrus":           (-25, -28, -10,   8),
    "cingulate gyrus":         ( -5,  25,  25,  15),    # anterior cingulate
    "Heschl's gyrus":          (-50, -20,   5,  10),    # primary auditory
    "precentral gyrus":        (-35, -20,  55,  15),    # M1
    "postcentral gyrus":       (-40, -25,  55,  15),    # S1
    "cuneus":                  (-10, -85,   5,  15),    # V1/V2 medial
    "globus pallidus":         (-20,  -5,   0,   8),
    "nucleus accumbens":       (-10,  10, -10,   6),
    "caudate nucleus":         (-15,  10,  10,  12),
    "amygdala":                (-25,  -5, -20,   8),
    "inferior colliculus":     ( -5, -35,  -8,   6),
    "superior colliculus":     ( -5, -30,  -2,   6),
    "myelencephalon":          None,                     # not in our atlas
    "pons":                    ( -5, -25, -35,  10),
    "hypothalamus":            ( -5,  -5, -15,   8),
    "thalamus":                (-10, -20,   5,  12),
}


# Estimated translation from our mouse coords to DSURQE world coords.
# Estimated from 6 anchor pairs (left/right Visual, Motor, Auditory) whose
# DSURQE leaf IDs are unambiguous. See script audit notes.
DSURQE_OFFSET_MM = np.array([-0.027, -2.334, +1.018])


def parse_dsurqe_tree(path: Path) -> dict[str, set[int]]:
    """Return {region_name: set(label_ids descended from that node)}."""
    tree = json.loads(Path(path).read_text())

    def normlab(L):
        return [] if not L else ([L] if isinstance(L, int) else [int(x) for x in L])

    def walk(node):
        out = [{"name": node.get("name"), "labels": normlab(node.get("label"))}]
        for c in (node.get("children") or {}).values():
            out.extend(walk(c))
        return out

    return {n["name"]: set(n["labels"]) for n in walk(tree["msg"][0]) if n["labels"]}


def assign_dsurqe_labels(M, dsurqe_volume_path: Path, *, radius: int = 2) -> np.ndarray:
    """Return (n_m,) array of DSURQE label IDs (0 if no overlap within radius)."""
    img = nib.load(str(dsurqe_volume_path))
    labels = np.asarray(img.get_fdata()).astype(np.int32)
    sh = labels.shape

    xyz = M.var[["x", "y", "z"]].to_numpy() + DSURQE_OFFSET_MM
    inv = np.linalg.inv(img.affine)
    voxels = (inv @ np.c_[xyz, np.ones(len(xyz))].T).T[:, :3]
    i, j, k = (voxels[:, ax].round().astype(int) for ax in range(3))

    out = np.zeros(len(xyz), dtype=np.int32)
    for p in range(len(xyz)):
        i0, i1 = max(0, i[p] - radius), min(sh[0], i[p] + radius + 1)
        j0, j1 = max(0, j[p] - radius), min(sh[1], j[p] + radius + 1)
        k0, k1 = max(0, k[p] - radius), min(sh[2], k[p] + radius + 1)
        block = labels[i0:i1, j0:j1, k0:k1].ravel()
        nz = block[block > 0]
        if len(nz):
            out[p] = Counter(nz.tolist()).most_common(1)[0][0]
    return out


def assign_human_region_membership(
    H, name_to_mni: dict[str, tuple | None]
) -> dict[str, np.ndarray]:
    """For each Beauchamp human name, return a boolean mask over our 2094 parcels.

    Membership = parcels whose xyz is within `radius_mm` of the named centroid
    on EITHER hemisphere (the `(x_left, y, z)` centroid is mirrored to the
    right hemisphere automatically).
    """
    xyz = H.var[["x", "y", "z"]].to_numpy()
    out = {}
    for hname, info in name_to_mni.items():
        if info is None:
            out[hname] = None
            continue
        x_left, y, z, radius = info
        # Left hemisphere centroid + right (mirror x)
        c_left  = np.array([x_left, y, z])
        c_right = np.array([-x_left, y, z])
        d_left  = np.linalg.norm(xyz - c_left,  axis=1)
        d_right = np.linalg.norm(xyz - c_right, axis=1)
        mask = (d_left <= radius) | (d_right <= radius)
        out[hname] = mask
    return out


def evaluate_pair(pi, m_mask, h_mask, h_xyz, *, k_top=10, h_centroid=None):
    """For each mouse parcel in m_mask, evaluate whether its top-K human picks
    fall in h_mask. Returns dict of metrics."""
    n = int(m_mask.sum())
    if n == 0 or h_mask.sum() == 0:
        return None
    h_idx_set = set(np.where(h_mask)[0].tolist())

    top_k_arr = np.argsort(-pi[m_mask, :], axis=1)[:, :k_top]
    # Per-mouse-parcel: did top-K hit any h_mask parcel?
    in_top1  = np.array([top_k_arr[i, 0]   in h_idx_set for i in range(n)])
    in_top5  = np.array([any(t in h_idx_set for t in top_k_arr[i, :5])  for i in range(n)])
    in_top10 = np.array([any(t in h_idx_set for t in top_k_arr[i, :10]) for i in range(n)])

    # Rank of best matching human parcel
    h_indices = np.where(h_mask)[0]
    pi_to_h = pi[m_mask][:, h_indices]               # (n, |h_mask|)
    best_h_in_region = h_indices[pi_to_h.argmax(axis=1)]   # (n,)
    # Overall rank of that best-matching parcel within all 2094
    ranks = np.array([
        (-pi[m_mask][i]).argsort().tolist().index(int(best_h_in_region[i])) + 1
        for i in range(n)
    ])

    # xyz distance from argmax to h_centroid
    if h_centroid is None:
        h_centroid = h_xyz[h_mask].mean(axis=0)
    argmax_h = top_k_arr[:, 0]
    argmax_xyz = h_xyz[argmax_h]
    dist = np.linalg.norm(argmax_xyz - h_centroid[None, :], axis=1)

    # Mean mass on the human region (sum of pi over h_mask)
    mass = pi[m_mask][:, h_mask].sum(axis=1) / pi[m_mask].sum(axis=1).clip(min=1e-12)

    return {
        "n_mouse_parcels":   int(n),
        "n_human_parcels":   int(h_mask.sum()),
        "top1":              float(in_top1.mean()),
        "top5":              float(in_top5.mean()),
        "top10":             float(in_top10.mean()),
        "mean_rank_in_region": float(ranks.mean()),
        "median_rank_in_region": float(np.median(ranks)),
        "mean_xyz_dist_mm":  float(dist.mean()),
        "median_xyz_dist_mm": float(np.median(dist)),
        "mean_mass_in_region": float(mass.mean()),
    }


def main(args):
    print(f"Loading π from {args.pi_file}")
    pi = np.load(COUP / args.pi_file).astype(np.float64)
    print(f"  shape={pi.shape}")

    M, _ = load_cached("mouse", cache_dir=ANN)
    H, _ = load_cached("human", cache_dir=ANN)
    assert pi.shape == (len(M.var), len(H.var)), \
        f"pi shape {pi.shape} != ({len(M.var)}, {len(H.var)})"

    # ---- Mouse side: parcel -> DSURQE label -> Beauchamp region membership
    #
    # NB: regions are resolved via the live DSURQE atlas volume, not the
    # parcel table's precomputed `region_vote_ss_dsq` labels, the vote
    # vocabulary uses different region names than the tree (striatum vs
    # Caudoputamen, British vs American spelling, etc.). See _dsurqe.py.
    name_to_dsurqe_labels = parse_dsurqe_tree(EXT / "AMBA/data/DSURQE_tree.json")
    parcel_dsurqe = assign_dsurqe_labels(
        M, EXT / "AMBA/data/imaging/DSURQE_CCFv3_labels_200um.mnc",
    )
    print(f"  Mouse parcels with DSURQE label assigned: "
          f"{(parcel_dsurqe>0).sum()}/{len(parcel_dsurqe)}")

    # ---- Human side: parcel -> Beauchamp region membership
    human_membership = assign_human_region_membership(H, HUMAN_REGION_MNI)
    h_xyz = H.var[["x", "y", "z"]].to_numpy()
    print("\nHuman region membership (parcels within radius of MNI centroid):")
    for hname, mask in human_membership.items():
        if mask is None:
            print(f"  {hname:25s} skipped")
        else:
            print(f"  {hname:25s} {int(mask.sum()):4d} parcels")

    # ---- Anchor-overlap tagging: hand-curated (Garin's 21 anchors largely
    # overlap Beauchamp's 22 canonical pairs, since both use well-known
    # cross-species homologies). Only hippocampal subfields and medulla are
    # "novel" w.r.t. our supervision.
    HIPPOCAMPAL_OR_MISSING = {
        "Subiculum",      # hippocampal formation, no Garin anchor
        "Field CA1", "Field CA2", "Field CA3", "Dentate gyrus",
        "Medulla",        # outside our parcellation
    }

    # ---- Run validation per pair
    results = {}
    for mname, hname in BEAUCHAMP_PAIRS:
        # Mouse mask: any parcel whose DSURQE label belongs to mname's branch
        if mname not in name_to_dsurqe_labels:
            results[f"{mname} -> {hname}"] = {"skip_reason": "mouse name not in DSURQE tree"}
            continue
        m_lbl_set = name_to_dsurqe_labels[mname]
        m_mask = np.isin(parcel_dsurqe, list(m_lbl_set))

        h_mask = human_membership[hname]
        if h_mask is None:
            results[f"{mname} -> {hname}"] = {
                "skip_reason": "human region not in our atlas vocabulary",
                "n_mouse_parcels": int(m_mask.sum()),
            }
            continue

        if m_mask.sum() == 0 or h_mask.sum() == 0:
            results[f"{mname} -> {hname}"] = {
                "skip_reason": ("no mouse parcels" if m_mask.sum() == 0
                                else "no human parcels"),
                "n_mouse_parcels": int(m_mask.sum()),
                "n_human_parcels": int(h_mask.sum()),
            }
            continue

        metrics = evaluate_pair(pi, m_mask, h_mask, h_xyz, k_top=10)
        metrics["is_anchor_overlapping"] = (mname not in HIPPOCAMPAL_OR_MISSING)
        results[f"{mname} -> {hname}"] = metrics

    # ---- Save + print
    out_path = LOG / "beauchamp_validation.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved {out_path}")

    # Summary
    print(f"\n{'pair':50s} {'n_m':>4s} {'n_h':>4s} {'top1':>6s} {'top5':>6s} "
          f"{'top10':>6s} {'rank':>8s} {'dist_mm':>8s} {'anchor?':>8s}")
    print("-" * 110)
    succ = []
    for pair, r in results.items():
        if "skip_reason" in r:
            n_m = r.get("n_mouse_parcels", "-")
            print(f"  {pair:50s} {str(n_m):>4s}                                     "
                  f"   ⚠ {r['skip_reason']}")
            continue
        succ.append(r)
        print(f"  {pair:50s} {r['n_mouse_parcels']:>4d} {r['n_human_parcels']:>4d} "
              f"{r['top1']:>6.0%} {r['top5']:>6.0%} {r['top10']:>6.0%} "
              f"{r['mean_rank_in_region']:>8.1f} {r['mean_xyz_dist_mm']:>8.1f}  "
              f"{'YES' if r['is_anchor_overlapping'] else 'no':>8s}")

    if succ:
        print("\n--- Aggregate (weighted by n_mouse_parcels) ---")
        wts = np.array([r["n_mouse_parcels"] for r in succ], dtype=float)
        wsum = wts.sum()
        for k in ("top1", "top5", "top10", "mean_rank_in_region", "mean_xyz_dist_mm"):
            v = sum(wts[i] * succ[i][k] for i in range(len(succ))) / wsum
            print(f"  {k:30s} = {v:.3f}")

        # ---- Chance baseline: if argmax were uniformly random over 2094 parcels,
        # the expected top-K hit rate for a region with n_h parcels is
        # 1 - C(N-n_h, K)/C(N, K) ≈ 1 - (1 - n_h/N)^K (for K << N).
        N_h = pi.shape[1]
        chance_top1 = chance_top5 = chance_top10 = 0.0
        for r in succ:
            w = r["n_mouse_parcels"] / wsum
            n_h = r["n_human_parcels"]
            for K, acc in [(1, "chance_top1"), (5, "chance_top5"), (10, "chance_top10")]:
                p = 1 - (1 - n_h / N_h) ** K
                if K == 1:  chance_top1  += w * p
                if K == 5:  chance_top5  += w * p
                if K == 10: chance_top10 += w * p
        print(f"\n--- Chance baseline (uniform random argmax over {N_h} parcels) ---")
        print(f"  chance_top1                    = {chance_top1:.3f}")
        print(f"  chance_top5                    = {chance_top5:.3f}")
        print(f"  chance_top10                   = {chance_top10:.3f}")
        actual_top1  = sum(wts[i] * succ[i]["top1"]  for i in range(len(succ))) / wsum
        actual_top5  = sum(wts[i] * succ[i]["top5"]  for i in range(len(succ))) / wsum
        actual_top10 = sum(wts[i] * succ[i]["top10"] for i in range(len(succ))) / wsum
        print(f"  enrichment top1                = {actual_top1/max(chance_top1,1e-9):.1f}x")
        print(f"  enrichment top5                = {actual_top5/max(chance_top5,1e-9):.1f}x")
        print(f"  enrichment top10               = {actual_top10/max(chance_top10,1e-9):.1f}x")

        # Anchor vs novel breakdown
        anc = [r for r in succ if r["is_anchor_overlapping"]]
        nov = [r for r in succ if not r["is_anchor_overlapping"]]
        for label, group in [("anchor-overlapping", anc), ("novel", nov)]:
            if not group: continue
            wts_g = np.array([r["n_mouse_parcels"] for r in group], dtype=float)
            wsum_g = wts_g.sum()
            top1 = sum(wts_g[i] * group[i]["top1"] for i in range(len(group))) / wsum_g
            top5 = sum(wts_g[i] * group[i]["top5"] for i in range(len(group))) / wsum_g
            top10 = sum(wts_g[i] * group[i]["top10"] for i in range(len(group))) / wsum_g
            ch1 = sum((wts_g[i]/wsum_g) * (1 - (1 - group[i]["n_human_parcels"]/N_h))
                      for i in range(len(group)))
            ch5 = sum((wts_g[i]/wsum_g) * (1 - (1 - group[i]["n_human_parcels"]/N_h)**5)
                      for i in range(len(group)))
            print(f"  [{label:20s}] n_pairs={len(group)} n_parcels={int(wsum_g)} "
                  f"top1={top1:.0%} (chance {ch1:.1%}; {top1/max(ch1,1e-9):.1f}x) "
                  f"top5={top5:.0%} (chance {ch5:.1%}; {top5/max(ch5,1e-9):.1f}x) "
                  f"top10={top10:.0%}")

        # Save aggregate to JSON for downstream
        results["__aggregate__"] = {
            "n_pairs_succeeded": len(succ),
            "n_pairs_anchor": len(anc),
            "n_pairs_novel": len(nov),
            "n_mouse_parcels_total": int(wsum),
            "top1": actual_top1, "top5": actual_top5, "top10": actual_top10,
            "chance_top1": chance_top1,
            "chance_top5": chance_top5,
            "chance_top10": chance_top10,
            "enrichment_top1": actual_top1/max(chance_top1,1e-9),
            "enrichment_top5": actual_top5/max(chance_top5,1e-9),
            "enrichment_top10": actual_top10/max(chance_top10,1e-9),
        }
        out_path.write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi-file", default="pi_canonical.npy",
                    help="filename in outputs/coupling/")
    main(ap.parse_args())
