#!/usr/bin/env python3
"""Held-out three-config comparison of the supervision terms.

For each of the 19 Beauchamp homology pairs, that region's own curation is withheld (its Garin
point anchor, every region pack whose mouse-side set touches it, and that landmark's
contribution to the thin-plate-spline warp behind the spatial cost), the model is refitted, and
recovery of the held-out region is scored under three configurations:

    both        connectivity + anchor-warped space, at production settings
    xyz_only    space only (alpha = 0)
    conn_only   connectivity only, no cross-species spatial cost

No configuration wins everywhere. Connectivity alone recovers the structures
whose position differs most between the species, superior colliculus and the hippocampal
subfields, and fails elsewhere. The comparison is read through the minimax and regret summary
computed by 06_regret.py from this log.

Writes outputs/logs/out_a1b_loro.json. outputs/logs/heldout_three_config_canonical.json holds the
same per-region values under the canonical name; --also-canonical writes both so they cannot
drift apart.

disentangle_loro.py runs the same comparison at epsilon = 5e-3 and xyz_weight = 0.5 rather than
the canonical epsilon = 0.05 and xyz_weight = 0.25, so its log
loro_disentangle_connectivity_vs_xyz.json is not comparable with this one.

57 fits. Resumable: each region is written as it completes, and re-running skips what is done.
Because the warp is now refitted per held-out region, every value differs from a run that
shared one warp across regions; --force --rewrite is needed once to replace such a log.

    conda activate retune
    cd otter && python3 experiments/section2_supervision/05_heldout_three_config.py --check
    cd otter && python3 experiments/section2_supervision/05_heldout_three_config.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]                       # .../otter
sys.path.insert(0, str(ROOT / "src"))

from otter.repro import (ALPHA, EPSILON, FC_WEIGHT, SC_WEIGHT, XYZ_WEIGHT,   # noqa: E402
                         anchor_warped_xyz, beauchamp_scorer, fit_coupling,
                         load_inputs, refit_provenance, stamp)

OUT = ROOT / "outputs" / "logs" / "out_a1b_loro.json"
MIRROR = ROOT / "outputs" / "logs" / "heldout_three_config_canonical.json"
# Resume state, not the output log. Keying resume off the output would let a second
# invocation find nothing to do, fit nothing, and still rewrite the log with no provenance.
PROGRESS = ROOT / "outputs" / "logs" / ".05_heldout_progress.json"

# The mirror's consumers, recorded in the log so the dependency is visible.
CONSUMERS = "notebooks/; experiments/section2_supervision/06_regret.py"

# The canonical recipe with one term switched off at a time. Everything not named here comes
# from otter.repro, so a change to the recipe reaches this script rather than being restated.
CONFIGS: dict[str, dict] = {
    "both":      dict(alpha=ALPHA, xyz_weight=XYZ_WEIGHT),
    "xyz_only":  dict(alpha=0.0,   xyz_weight=XYZ_WEIGHT),
    "conn_only": dict(alpha=ALPHA, xyz_weight=0.0),
}

WHAT = ("Held-out three-config disentanglement. For each of the 19 Beauchamp homology pairs, "
        "that region's own curation (its Garin anchor and any overlapping region pack) is "
        "withheld, the model is re-fitted, and recovery of the held-out region is scored under "
        "three configurations. cdist_mm is the displacement of the routed centroid from the "
        "expected human homologue; rand_mm is the mean distance from that homologue to the human "
        "parcel cloud, which is where a coupling carrying no spatial information would route. "
        "Produced by experiments/section2_supervision/05_heldout_three_config.py.")

SETTINGS = (f"epsilon = {EPSILON}, xyz_weight = {XYZ_WEIGHT}, alpha = {ALPHA}, "
            f"FC/SC {FC_WEIGHT}/{SC_WEIGHT} (canonical)")

FLOAT_RTOL = 1e-6      # refits are not bit-reproducible; see --check


def score(pi: np.ndarray, m_mask, h_mask, h_xyz) -> dict:
    """The five quantities the committed log records, scored exactly as beauchamp_battery does."""
    hidx = np.where(h_mask)[0]
    true_c = h_xyz[hidx].mean(0)
    block = pi[m_mask]
    tot = block.sum(0)
    s = tot.sum()
    if s <= 0:
        return dict(cdist_mm=float("nan"), mass=0.0, top1=0.0, auroc=float("nan"),
                    rand_mm=float(np.linalg.norm(h_xyz - true_c[None, :], axis=1).mean()))
    totn = tot / s
    pred_c = (totn[:, None] * h_xyz).sum(0)
    return dict(
        cdist_mm=float(np.linalg.norm(pred_c - true_c)),
        mass=float(tot[hidx].sum() / s),
        top1=float(np.isin(block.argmax(1), hidx).mean()),
        auroc=float(roc_auc_score(h_mask, tot)),
        rand_mm=float(np.linalg.norm(h_xyz - true_c[None, :], axis=1).mean()),
    )


def withhold(M, H, entries, entry_mouse_idx, m_mask, garin_M, garin_H, apid_m, apid_h):
    """Mask this region's Garin anchor, and drop every pack whose mouse set touches it.

    Returns the surviving pack entries. The caller restores the Garin columns afterwards. A pack
    is dropped on any overlap rather than on majority overlap, so the held-out region keeps no
    curation of its own even partially.
    """
    finite = np.isfinite(apid_m)
    pids = set(apid_m[garin_M & m_mask & finite].astype(int).tolist()) or {-999}
    drop_m = garin_M & np.isin(np.nan_to_num(apid_m, nan=-1).astype(int), list(pids))
    drop_h = garin_H & np.isin(np.nan_to_num(apid_h, nan=-1).astype(int), list(pids))
    M.var["garin_anchor"] = garin_M & ~drop_m
    H.var["garin_anchor"] = garin_H & ~drop_h
    surviving = [e for e, mi in zip(entries, entry_mouse_idx) if not m_mask[mi].any()]
    return surviving, {p for p in pids if p != -999}


def compare(built: dict, committed: dict) -> list[str]:
    """Regions whose displacement differs from the committed log beyond FLOAT_RTOL."""
    out = []
    for key in sorted(k for k in built if not k.startswith("_")):
        if key not in committed:
            out.append(f"{key}: absent from the committed log")
            continue
        for config in CONFIGS:
            a, b = built[key][config]["cdist_mm"], committed[key][config]["cdist_mm"]
            if not (abs(a - b) <= FLOAT_RTOL * max(abs(a), abs(b))):
                out.append(f"{key} / {config}: cdist_mm {a:.6f} against {b:.6f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="score without writing, and report any drift from the committed log")
    ap.add_argument("--force", action="store_true", help="rescore regions already in the log")
    ap.add_argument("--also-canonical", action="store_true",
                    help="write heldout_three_config_canonical.json with the same values")
    ap.add_argument("--rewrite", action="store_true",
                    help="write even when the values differ from the committed log. Needed once, "
                         "because refitting the warp per held-out region changes every value.")
    args = ap.parse_args()

    M, H, costs, entries = load_inputs()
    BB = beauchamp_scorer()
    pairs, _, _, h_xyz, _, _ = BB.build(M, H)

    garin_M = M.var["garin_anchor"].fillna(False).to_numpy().astype(bool)
    garin_H = H.var["garin_anchor"].fillna(False).to_numpy().astype(bool)
    apid_m = pd.to_numeric(M.var["anchor_pair_id"], errors="coerce").to_numpy(
        dtype="float64", na_value=np.nan)
    apid_h = pd.to_numeric(H.var["anchor_pair_id"], errors="coerce").to_numpy(
        dtype="float64", na_value=np.nan)
    entry_mouse_idx = [np.asarray(e.mouse_indices, dtype=int) for e in entries]

    committed = json.loads(OUT.read_text()) if OUT.exists() else {}
    use_progress = PROGRESS.exists() and not args.force and not args.check
    resumed = json.loads(PROGRESS.read_text()) if use_progress else {}
    out = {k: v for k, v in resumed.items() if not k.startswith("_")}
    todo = [k for k in pairs if k not in out]
    if resumed and todo:
        print(f"resuming from {PROGRESS.name}, {len(out)} region(s) already fitted")
    print(f"{len(pairs)} pairs, {len(todo)} to score, {len(CONFIGS)} fits each")

    last_pi = None
    for key in todo:
        m_mask, h_mask = pairs[key]
        start = time.time()
        try:
            surviving, held_pids = withhold(M, H, entries, entry_mouse_idx, m_mask,
                                            garin_M, garin_H, apid_m, apid_h)
            # The warp is refitted without the held-out landmark, so its position cannot
            # re-enter through the spatial term.
            M_xyz = anchor_warped_xyz(M, H, exclude_pids=held_pids)
            row = {}
            for name, config in CONFIGS.items():
                pi = fit_coupling(M, H, costs, surviving, M_xyz,
                                  garin=True, packs=True, epsilon=EPSILON, **config)
                row[name] = score(pi, m_mask, h_mask, h_xyz)
                if name == "both":
                    last_pi = pi
        finally:
            M.var["garin_anchor"] = garin_M
            H.var["garin_anchor"] = garin_H

        out[key] = row
        if not args.check:
            PROGRESS.write_text(json.dumps(out, indent=2, default=float))
        print(f"  {key.split(' -> ')[0][:26]:26s} both={row['both']['cdist_mm']:6.1f} "
              f"xyz={row['xyz_only']['cdist_mm']:6.1f} conn={row['conn_only']['cdist_mm']:6.1f} "
              f"rand={row['both']['rand_mm']:5.1f}mm  ({time.time() - start:.0f}s)", flush=True)

    if committed and not args.rewrite:
        drift = compare(out, committed)
        if drift:
            print(f"\nDIFFERS from the committed log in {len(drift)} place(s):", file=sys.stderr)
            for d in drift[:15]:
                print(f"  {d}", file=sys.stderr)
            print("\nNot written; the committed values stand.", file=sys.stderr)
            return 1
        print(f"\nreproduces the committed {OUT.name} on all "
              f"{len(CONFIGS) * len([k for k in out if not k.startswith('_')])} displacements")

    if not args.check:
        if last_pi is None:
            print("\nNothing was fitted in this run, so there is no coupling to record and the "
                  "log would go out unstamped. Re-run with --force to refit all 19 regions.",
                  file=sys.stderr)
            return 1
        # The canonical arm is a refit, so the record includes its recipe and measured distance.
        # from the release rather than a coupling sha this run never opened.
        prov = refit_provenance(last_pi, recipe={**CONFIGS["both"], "epsilon": EPSILON,
                                                 "garin": True, "packs": True,
                                                 "note": "last held-out arm; curation withheld "
                                                         "for one region, so agreement with the "
                                                         "release is expected to be high but not "
                                                         "exact"}) if last_pi is not None else {}
        payload = stamp({"_what": WHAT, "_settings": SETTINGS, **out}, **prov)
        OUT.write_text(json.dumps(payload, indent=2, default=float))
        print(f"wrote {OUT.relative_to(ROOT)}")
        if args.also_canonical:
            mirror = dict(payload)
            mirror["_consumers"] = CONSUMERS
            MIRROR.write_text(json.dumps(mirror, indent=2, default=float))
            print(f"wrote {MIRROR.relative_to(ROOT)}")
        PROGRESS.unlink(missing_ok=True)
        print("\nNow run: python3 experiments/section2_supervision/06_regret.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
