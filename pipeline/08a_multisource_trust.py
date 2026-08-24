"""Build the compatibility file containing explorer display metadata.

The output combines anchor membership, benchmark-region membership and internal
stability summaries. Its historical field names are retained because the
self-contained explorer reads them, but the categories are interface metadata:
they are not calibrated confidence estimates or additional validation results.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from otter.data import load_cached                                       # noqa: E402
from otter.data.anchor_packs import (                                    # noqa: E402
    DEFAULT_PACK_NAMES,
    build_default_pack_entries,
)
from otter.data.anchor_packs._dsurqe import (                            # noqa: E402
    assign_dsurqe_labels,
    parse_dsurqe_tree,
)
from otter.eval.trust_score import compute_multisource_trust             # noqa: E402

ANN  = ROOT / "outputs" / "anndata"
COUP = ROOT / "outputs" / "coupling"
LOG  = ROOT / "outputs" / "logs"
EXT  = ROOT / "data_external" / "MouseHumanTranscriptomicSimilarity"

DSURQE_TREE   = EXT / "AMBA/data/DSURQE_tree.json"
DSURQE_VOLUME = EXT / "AMBA/data/imaging/DSURQE_CCFv3_labels_200um.mnc"


def main(args):
    # ---- Inputs
    pi_path = COUP / args.pi_file
    if not pi_path.exists():
        raise SystemExit(
            f"π not found: {pi_path}\n"
            f"  download the release bundle or pass an existing --pi-file."
        )
    M, _ = load_cached("mouse", cache_dir=ANN)
    H, _ = load_cached("human", cache_dir=ANN)
    pi = np.load(pi_path).astype(np.float64)
    print(f"Loaded {args.pi_file} ({pi.shape})")
    assert pi.shape == (len(M.var), len(H.var)), \
        f"π shape {pi.shape} != ({len(M.var)}, {len(H.var)})"

    boot_path = COUP / args.bootstrap_file if args.bootstrap_file else None
    if boot_path is not None and not boot_path.exists():
        print(f"  ⚠ bootstrap aggregate {boot_path.name} missing, "
              f"bootstrap component will be a constant 0.5")
        boot_path = None

    # ---- Region-anchor packs (external supervision signal #1).
    # The registry defines the canonical regional-entry composition.
    print(f"\nBuilding default region-anchor packs "
          f"({', '.join(DEFAULT_PACK_NAMES)}) ...")
    entries = build_default_pack_entries(M.var, H.var, atlas_root=ROOT)
    n_pack_parcels = len({i for e in entries for i in e.mouse_indices})
    print(f"  {len(entries)} region-anchor entries, "
          f"{n_pack_parcels} distinct pack-anchored mouse parcels")

    # ---- Beauchamp validation (external supervision signal #2)
    beau_path = LOG / args.beauchamp_file
    if not beau_path.exists():
        raise SystemExit(
            f"Benchmark-region metadata not found: {beau_path}"
        )
    beauchamp_per_pair = json.loads(beau_path.read_text())
    print(f"\nLoaded {args.beauchamp_file} "
          f"({len([k for k in beauchamp_per_pair if not k.startswith('_')])} pairs)")

    # Per-parcel DSURQE label + region→label map, so each parcel can be
    # attached to its Beauchamp validation pair.
    if not (DSURQE_TREE.exists() and DSURQE_VOLUME.exists()):
        raise SystemExit(
            f"Beauchamp 2022 DSURQE atlas missing under {EXT}\n"
            f"  needed to attach parcels to Beauchamp validation regions."
        )
    print("Assigning DSURQE labels to mouse parcels ...")
    mouse_dsurqe_labels = assign_dsurqe_labels(M.var, DSURQE_VOLUME)
    region_to_dsurqe = parse_dsurqe_tree(DSURQE_TREE)
    print(f"  {(mouse_dsurqe_labels > 0).sum()}/{len(mouse_dsurqe_labels)} "
          f"parcels assigned a DSURQE label")

    # ---- Explorer display metadata
    print("\nComputing explorer display metadata ...")
    out = compute_multisource_trust(
        M, H, pi,
        bootstrap_path=boot_path,
        region_anchor_entries=entries,
        beauchamp_per_pair=beauchamp_per_pair,
        mouse_dsurqe_labels=mouse_dsurqe_labels,
        beauchamp_region_to_mouse_dsurqe=region_to_dsurqe,
    )

    # ---- Report
    print(f"\nInternal stability composite:")
    print(f"  range=[{out['trust'].min():.3f}, {out['trust'].max():.3f}], "
          f"mean={out['trust'].mean():.3f}")
    print(f"\nDisplay-category distribution ({len(out['evidence_tier'])} parcels):")
    tier_counts = Counter(str(t) for t in out["evidence_tier"])
    for tier in ("anchored_and_validated", "anchored_only", "validated_only",
                 "structural", "low_evidence"):
        n = tier_counts.get(tier, 0)
        print(f"  {tier:24s}: {n:>4d}  ({n / len(out['evidence_tier']):.0%})")
    print(f"\n  garin-anchored : {int(out['garin_anchored'].sum()):>4d} parcels")
    print(f"  pack-anchored  : {int(out['pack_anchored'].sum()):>4d} parcels")
    n_validated = int(np.nansum(out["beauchamp_top1"] > 0))
    print(f"  Beauchamp top-1 > 0 : {n_validated:>4d} parcels")

    # ---- Save (keys consumed by otter.viz.gui._node_records)
    out_path = COUP / args.output
    np.savez(
        out_path,
        trust=out["trust"].astype(np.float64),
        tier=np.array([str(t) for t in out["tier"]]),
        bootstrap=out["bootstrap"].astype(np.float64),
        concentration=out["concentration"].astype(np.float64),
        fc_sim=out["fc_sim"].astype(np.float64),
        weights=np.asarray(out["weights"], dtype=np.float64),
        garin_anchored=out["garin_anchored"],
        pack_anchored=out["pack_anchored"],
        beauchamp_top1=out["beauchamp_top1"].astype(np.float64),
        evidence_tier=np.array([str(t) for t in out["evidence_tier"]]),
    )
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pi-file", default="pi_canonical.npy",
                    help="coupling filename in outputs/coupling/")
    ap.add_argument("--beauchamp-file", default="beauchamp_validation_canonical.json",
                    help="benchmark-region metadata filename in outputs/logs/")
    ap.add_argument("--bootstrap-file", default=None,
                    help="matching bootstrap aggregate in outputs/coupling/; omitted by default")
    ap.add_argument("--output", default="trust_multisource_canonical.npz",
                    help="output filename in outputs/coupling/ read by the explorer")
    main(ap.parse_args())
