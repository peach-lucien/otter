"""Compute uncalibrated stability summaries for the point-anchor configurations defined by the low-level pipeline.

The historical output field names are retained for compatibility and must not be interpreted as calibrated confidence."""
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
from otter.data import load_cached                                      # noqa: E402
from otter.eval.trust_score import (                                    # noqa: E402
    compute_trust_score,
    regional_empirical_accuracy,
    assign_regional_trust,
)

# Reuse the Beauchamp region table from 05f, which is the single source of
# truth for the pair list and the human MNI centroids.
# 05j_region_level_eval.py loads the same tables the same way.
sys.path.insert(0, str(ROOT / "pipeline"))
from importlib import import_module                                     # noqa: E402
_beau = import_module("05f_beauchamp_validation")

ANN  = ROOT / "outputs" / "anndata"
COUP = ROOT / "outputs" / "coupling"
EXT  = ROOT / "data_external" / "MouseHumanTranscriptomicSimilarity"


# Beauchamp region table, derived from `pipeline/05f_beauchamp_validation.py`
# rather than retyped, so the two cannot drift apart.
# Format: (mouse_DSURQE_region_name, x_left, y, z, radius_mm), the tuple shape
# the two helpers below unpack.
# Pairs whose human region has no curated MNI centroid in 05f (the entry is
# None, currently Medulla only) carry no sphere and are not evaluable, so they
# are dropped here. Order follows 05f's BEAUCHAMP_PAIRS, which is the order
# `parcel_to_dsurqe_region` resolves ties in when a parcel's DSURQE label
# belongs to more than one Beauchamp region.
BEAUCHAMP_REGIONS = [
    (mouse_name, *_beau.HUMAN_REGION_MNI[human_name])
    for mouse_name, human_name in _beau.BEAUCHAMP_PAIRS
    if _beau.HUMAN_REGION_MNI.get(human_name) is not None
]


def parcel_to_dsurqe_region(M, dsurqe_volume_path):
    """Return (n_m,) array of DSURQE Beauchamp region names per parcel
    (or '' if not in any evaluable region of `BEAUCHAMP_REGIONS`)."""
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

    # 1. Internal stability composite.
    boot_path = COUP / args.bootstrap_file
    if not boot_path.exists():
        print(f"  ⚠ {boot_path} missing, bootstrap component will be 0.5")
        boot_path = None
    ts = compute_trust_score(M, H, pi, bootstrap_path=boot_path)
    print(f"\nInternal stability composite:")
    print(f"  range=[{ts['trust'].min():.3f}, {ts['trust'].max():.3f}], mean={ts['trust'].mean():.3f}")
    for t in ["high", "medium", "low"]:
        n = (ts["tier"] == t).sum()
        print(f"  {t}: {n}/{len(ts['tier'])}")

    # 2. Regional empirical summary.
    parcel_to_region = parcel_to_dsurqe_region(
        M, EXT / "AMBA/data/imaging/DSURQE_CCFv3_labels_200um.mnc",
    )
    expected_h = build_expected_h_indices(parcel_to_region, H.var)
    reg_acc = regional_empirical_accuracy(parcel_to_region, pi, expected_h)
    reg_trust, reg_tier = assign_regional_trust(
        len(M.var), parcel_to_region, reg_acc,
        high_threshold=args.high_threshold, low_threshold=args.low_threshold,
    )
    print(f"\nRegional empirical top-1 summary:")
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
        # Internal stability fields; names retained for file compatibility.
        trust=ts["trust"].astype(np.float32),
        tier=np.array([str(t) for t in ts["tier"]]),
        bootstrap=ts["bootstrap"].astype(np.float32),
        concentration=ts["concentration"].astype(np.float32),
        fc_sim=ts["fc_sim"].astype(np.float32),
        weights=np.asarray(ts["weights"], dtype=np.float64),
        # Regional empirical summary.
        regional_trust=reg_trust.astype(np.float32),
        regional_tier=np.array([str(t) for t in reg_tier]),
        parcel_to_region=np.array([str(r) for r in parcel_to_region]),
    )
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi-file",        default="pi_canonical.npy")
    ap.add_argument("--bootstrap-file", default="bootstrap_aggregate_fc_plus_SC.npz")
    ap.add_argument("--high-threshold", type=float, default=0.15)
    ap.add_argument("--low-threshold",  type=float, default=0.03)
    main(ap.parse_args())
