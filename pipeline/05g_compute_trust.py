"""Pipeline 05g, compute per-parcel trust score for the production π.

Two independent trust signals, both saved as outputs/coupling/trust_score_<config>.npz:

1. **Model-confidence trust** (continuous):
   bootstrap row-stability + argmax mass concentration + FC similarity to
   nearest anchor → composite score in [0, 1] + tier {high, medium, low}.

2. **Regional-empirical trust** (discrete by validation region):
   per Beauchamp 2022 region, what is the actual top-1 accuracy of the
   model in this region? Each parcel gets the accuracy of its region. Tier
   is set by absolute thresholds (high ≥ 15%, low < 3%, else medium,
   `unknown` if not in any validated region).

Saves:
    outputs/coupling/trust_score_<config>.npz with:
        - trust              : (n_m,) model-confidence composite, [0, 1]
        - tier               : (n_m,) {high, medium, low}
        - bootstrap          : per-row bootstrap argmax stability
        - concentration      : argmax mass / row sum
        - fc_sim             : Pearson r to nearest anchor's FC profile
        - regional_trust     : per-parcel empirical Beauchamp top-1 (NaN if unknown)
        - regional_tier      : (n_m,) {high, medium, low, unknown}
        - parcel_to_region   : Beauchamp region label per parcel ('' if not in one)
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
from homer.data import load_cached                                      # noqa: E402
from homer.eval.trust_score import (                                    # noqa: E402
    compute_trust_score,
    regional_empirical_accuracy,
    assign_regional_trust,
)

ANN  = ROOT / "outputs" / "anndata"
COUP = ROOT / "outputs" / "coupling"
EXT  = ROOT / "data_external" / "MouseHumanTranscriptomicSimilarity"


# Beauchamp pair list, kept in lockstep with `pipeline/05f_beauchamp_validation.py`
# HUMAN_REGION_MNI. Format: (mouse_DSURQE_region_name, x_left, y, z, radius_mm).
BEAUCHAMP_REGIONS = [
    ("Anterior cingulate area",            -5,  25,  25, 15),
    ("Primary motor area",                 -35, -20, 55, 15),
    ("Primary somatosensory area",         -40, -25, 55, 15),
    ("Visual areas",                       -10, -85,  5, 15),
    ("Pallidum",                           -20,  -5,  0,  8),
    ("Caudoputamen",                       -15,  10, 10, 12),
    ("Cortical subplate-other",            -25,  -5,-20,  8),
    ("Pons",                                -5, -25,-35, 10),
    ("Hypothalamus",                        -5,  -5,-15,  8),
    ("Thalamus",                           -10, -20,  5, 12),
    ("Piriform area",                      -25,   5,-20, 10),
    ("Inferior colliculus",                 -5, -35, -8,  6),
    ("Superior colliculus, sensory related",-5, -30, -2,  6),
    ("Striatum ventral region",            -10,  10,-10,  6),
    ("Primary auditory area",              -50, -20,  5, 10),
    # Hippocampal, match centroids used in pipeline/05f_*.py exactly
    ("Field CA1",                          -30, -25, -10,  8),
    ("Field CA3",                          -25, -22, -10,  8),
    ("Dentate gyrus",                      -25, -28, -10,  8),
    ("Subiculum",                          -22, -32,  -8,  8),
]


def parcel_to_dsurqe_region(M, dsurqe_volume_path):
    """Return (n_m,) array of DSURQE Beauchamp region names per parcel
    (or '' if not in any of our 19 evaluable regions)."""
    img = nib.load(str(dsurqe_volume_path))
    labels = np.asarray(img.get_fdata()).astype(np.int32); sh = labels.shape
    xyz_m = M.var[["x","y","z"]].to_numpy() + np.array([-0.027, -2.334, +1.018])
    inv = np.linalg.inv(img.affine)
    voxels = (inv @ np.c_[xyz_m, np.ones(len(xyz_m))].T).T[:, :3]
    i, j, k = (voxels[:, ax].round().astype(int) for ax in range(3))

    def lookup(p, r=2):
        i0, i1 = max(0, i[p]-r), min(sh[0], i[p]+r+1)
        j0, j1 = max(0, j[p]-r), min(sh[1], j[p]+r+1)
        k0, k1 = max(0, k[p]-r), min(sh[2], k[p]+r+1)
        block = labels[i0:i1, j0:j1, k0:k1].ravel()
        nz = block[block > 0]
        return Counter(nz.tolist()).most_common(1)[0][0] if len(nz) else 0
    parcel_lbl = np.array([lookup(p, 2) for p in range(len(xyz_m))])

    # DSURQE tree: name → label IDs
    tree = json.loads((EXT / "AMBA/data/DSURQE_tree.json").read_text())
    def normlab(L):
        return [] if not L else ([L] if isinstance(L, int) else [int(x) for x in L])
    def walk(n):
        nodes = [{"name": n.get("name"), "labels": normlab(n.get("label"))}]
        for c in (n.get("children") or {}).values():
            nodes.extend(walk(c))
        return nodes
    name_to_lbls = {n["name"]: set(n["labels"]) for n in walk(tree["msg"][0]) if n["labels"]}

    parcel_to_region = np.array([""] * len(xyz_m), dtype=object)
    for region_name, *_ in BEAUCHAMP_REGIONS:
        if region_name not in name_to_lbls: continue
        mask = np.isin(parcel_lbl, list(name_to_lbls[region_name]))
        parcel_to_region[mask & (parcel_to_region == "")] = region_name
    return parcel_to_region


def build_expected_h_indices(parcel_to_region, H_var):
    """For each mouse parcel in a Beauchamp region, the set of human parcels
    within that region's MNI sphere."""
    h_xyz = H_var[["x","y","z"]].to_numpy()
    expected = {}
    for region_name, x_L, y, z, r in BEAUCHAMP_REGIONS:
        mask_m = (parcel_to_region == region_name)
        if not mask_m.any(): continue
        cL = np.array([x_L, y, z]); cR = np.array([-x_L, y, z])
        h_in = (np.linalg.norm(h_xyz - cL[None,:], axis=1) <= r) | \
                (np.linalg.norm(h_xyz - cR[None,:], axis=1) <= r)
        h_set = set(int(p) for p in np.where(h_in)[0])
        if not h_set: continue
        for m in np.where(mask_m)[0]:
            expected[int(m)] = h_set
    return expected


def main(args):
    M, _ = load_cached("mouse", cache_dir=ANN)
    H, _ = load_cached("human", cache_dir=ANN)
    pi = np.load(COUP / args.pi_file).astype(np.float64)
    print(f"Loaded {args.pi_file} ({pi.shape})")

    # 1. Model-confidence trust
    boot_path = COUP / args.bootstrap_file
    if not boot_path.exists():
        print(f"  ⚠ {boot_path} missing, bootstrap component will be 0.5")
        boot_path = None
    ts = compute_trust_score(M, H, pi, bootstrap_path=boot_path)
    print(f"\nModel-confidence trust:")
    print(f"  range=[{ts['trust'].min():.3f}, {ts['trust'].max():.3f}], mean={ts['trust'].mean():.3f}")
    for t in ["high", "medium", "low"]:
        n = (ts["tier"] == t).sum()
        print(f"  {t}: {n}/{len(ts['tier'])}")

    # 2. Regional-empirical trust
    parcel_to_region = parcel_to_dsurqe_region(
        M, EXT / "AMBA/data/imaging/DSURQE_CCFv3_labels_200um.mnc",
    )
    expected_h = build_expected_h_indices(parcel_to_region, H.var)
    reg_acc = regional_empirical_accuracy(parcel_to_region, pi, expected_h)
    reg_trust, reg_tier = assign_regional_trust(
        len(M.var), parcel_to_region, reg_acc,
        high_threshold=args.high_threshold, low_threshold=args.low_threshold,
    )
    print(f"\nRegional-empirical trust (per Beauchamp validation):")
    for region, info in sorted(reg_acc.items(), key=lambda kv: -kv[1]["top1_accuracy"]):
        n = info["n"]; acc = info["top1_accuracy"]
        marker = "🟢" if acc >= args.high_threshold else "🔴" if acc < args.low_threshold else "🟡"
        print(f"  {marker} {region:42s} n={n:>3d}  top-1={acc:.0%}")
    n_unknown = (reg_tier == "unknown").sum()
    print(f"\n  Tier distribution:")
    for t in ["high", "medium", "low", "unknown"]:
        n = (reg_tier == t).sum()
        print(f"    {t:>8s}: {n:>4d}/{len(reg_tier)}")

    # Save combined output
    cfg_name = args.pi_file.replace("pi_", "").replace(".npy", "")
    out_path = COUP / f"trust_score_{cfg_name}.npz"
    np.savez(
        out_path,
        # model-confidence trust
        trust=ts["trust"].astype(np.float32),
        tier=np.array([str(t) for t in ts["tier"]]),
        bootstrap=ts["bootstrap"].astype(np.float32),
        concentration=ts["concentration"].astype(np.float32),
        fc_sim=ts["fc_sim"].astype(np.float32),
        weights=np.asarray(ts["weights"], dtype=np.float64),
        # regional empirical trust
        regional_trust=reg_trust.astype(np.float32),
        regional_tier=np.array([str(t) for t in reg_tier]),
        parcel_to_region=np.array([str(r) for r in parcel_to_region]),
    )
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi-file",        default="pi_fc_plus_SC.npy")
    ap.add_argument("--bootstrap-file", default="bootstrap_aggregate_fc_plus_SC.npz")
    ap.add_argument("--high-threshold", type=float, default=0.15)
    ap.add_argument("--low-threshold",  type=float, default=0.03)
    main(ap.parse_args())
