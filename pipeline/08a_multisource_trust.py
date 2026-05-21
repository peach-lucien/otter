"""Pipeline step 08a — multi-source per-parcel trust map for the recommended π.

This step is the missing link between the trust machinery and the GUI.

  * ``05g_compute_trust.py`` writes ``trust_score_<config>.npz`` — the
    *internal* composite (bootstrap stability + argmax concentration + FC
    similarity) plus the regional-empirical Beauchamp accuracy. Useful, but
    not the file the GUI reads.
  * ``08_build_gui.py`` reads ``trust_multisource_all_packs.npz`` — the
    *multi-source evidence map* (five evidence tiers) produced by
    :func:`homer.eval.trust_score.compute_multisource_trust`.

Until now nothing wrote that file inside the documented pipeline:
``compute_multisource_trust`` had no caller, and the GUI silently fell back
to "unknown" trust tiers when the file was absent or stale. This step is
that caller.

It layers external supervision (anchor-pack membership, Beauchamp region
validation) on top of the internal composite and classifies every mouse
parcel into one of five evidence tiers:

    anchored_and_validated  — in an anchor pack AND Beauchamp top-1 > 0
    anchored_only           — in an anchor pack, no Beauchamp validation
    validated_only          — Beauchamp top-1 > 0, not in any anchor pack
    structural              — high internal trust, no external evidence
    low_evidence            — none of the above (use predictions with caution)

Outputs:
    outputs/coupling/trust_multisource_all_packs.npz with:
        trust          : (n_m,) internal composite, [0, 1]
        tier           : (n_m,) {high, medium, low}  (internal composite tier)
        bootstrap      : (n_m,) per-row bootstrap argmax stability
        concentration  : (n_m,) argmax mass / row sum
        fc_sim         : (n_m,) Pearson r to nearest anchor FC profile
        weights        : (3,)   the (boot, concentration, fc) component weights
        garin_anchored : (n_m,) bool — one of the Garin point anchors
        pack_anchored  : (n_m,) bool — in any default region-anchor pack
        beauchamp_top1 : (n_m,) float — its Beauchamp pair's top-1 (NaN if N/A)
        evidence_tier  : (n_m,) {anchored_and_validated, anchored_only,
                                 validated_only, structural, low_evidence}

Usage:
    PYTHONPATH=src python pipeline/08a_multisource_trust.py
    PYTHONPATH=src python pipeline/08a_multisource_trust.py \
        --pi-file pi_fc_plus_SC_with_all_packs.npy \
        --beauchamp-file beauchamp_validation_all_packs.json \
        --bootstrap-file bootstrap_aggregate_fc_plus_SC.npz
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

from homer.data import load_cached                                       # noqa: E402
from homer.data.anchor_packs import (                                    # noqa: E402
    DEFAULT_PACK_NAMES,
    build_default_pack_entries,
)
from homer.data.anchor_packs._dsurqe import (                            # noqa: E402
    assign_dsurqe_labels,
    parse_dsurqe_tree,
)
from homer.eval.trust_score import compute_multisource_trust             # noqa: E402

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
            f"  run experiments/anchor_packs/compose_all.py first."
        )
    M, _ = load_cached("mouse", cache_dir=ANN)
    H, _ = load_cached("human", cache_dir=ANN)
    pi = np.load(pi_path).astype(np.float64)
    print(f"Loaded {args.pi_file} ({pi.shape})")
    assert pi.shape == (len(M.var), len(H.var)), \
        f"π shape {pi.shape} != ({len(M.var)}, {len(H.var)})"

    boot_path = COUP / args.bootstrap_file
    if not boot_path.exists():
        print(f"  ⚠ bootstrap aggregate {boot_path.name} missing — "
              f"bootstrap component will be a constant 0.5")
        boot_path = None

    # ---- Region-anchor packs (external supervision signal #1).
    # The pack registry is the single source of truth for the recommended
    # composition — this stays in lockstep with compose_all.py automatically.
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
            f"Beauchamp validation log not found: {beau_path}\n"
            f"  run experiments/anchor_packs/compose_all.py (it writes "
            f"beauchamp_validation_all_packs.json) first."
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

    # ---- Multi-source trust  (the previously-orphaned function)
    print("\nComputing multi-source trust map ...")
    out = compute_multisource_trust(
        M, H, pi,
        bootstrap_path=boot_path,
        region_anchor_entries=entries,
        beauchamp_per_pair=beauchamp_per_pair,
        mouse_dsurqe_labels=mouse_dsurqe_labels,
        beauchamp_region_to_mouse_dsurqe=region_to_dsurqe,
    )

    # ---- Report
    print(f"\nInternal composite trust:")
    print(f"  range=[{out['trust'].min():.3f}, {out['trust'].max():.3f}], "
          f"mean={out['trust'].mean():.3f}")
    print(f"\nEvidence-tier distribution ({len(out['evidence_tier'])} parcels):")
    tier_counts = Counter(str(t) for t in out["evidence_tier"])
    for tier in ("anchored_and_validated", "anchored_only", "validated_only",
                 "structural", "low_evidence"):
        n = tier_counts.get(tier, 0)
        print(f"  {tier:24s}: {n:>4d}  ({n / len(out['evidence_tier']):.0%})")
    print(f"\n  garin-anchored : {int(out['garin_anchored'].sum()):>4d} parcels")
    print(f"  pack-anchored  : {int(out['pack_anchored'].sum()):>4d} parcels")
    n_validated = int(np.nansum(out["beauchamp_top1"] > 0))
    print(f"  Beauchamp top-1 > 0 : {n_validated:>4d} parcels")

    # ---- Save (keys consumed by homer.viz.gui._node_records)
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
    ap.add_argument("--pi-file", default="pi_fc_plus_SC_with_all_packs.npy",
                    help="recommended π filename in outputs/coupling/")
    ap.add_argument("--beauchamp-file", default="beauchamp_validation_all_packs.json",
                    help="Beauchamp validation log filename in outputs/logs/")
    ap.add_argument("--bootstrap-file", default="bootstrap_aggregate_fc_plus_SC.npz",
                    help="bootstrap aggregate filename in outputs/coupling/")
    ap.add_argument("--output", default="trust_multisource_all_packs.npz",
                    help="output filename in outputs/coupling/")
    main(ap.parse_args())
