"""COMPOSE-ALL: fit MultimodalFGW with all default anchor packs at once.

The set of default packs is defined once in
:mod:`homer.data.anchor_packs.registry` (``DEFAULT_PACK_NAMES``) and built
here via :func:`build_default_pack_entries` — this script no longer keeps
its own copy of the list. Currently the default packs are:

  - biccn_motor (pid 30, 31) — Bakken 2021
  - tectum      (pid 32, 33) — May 2006; Schreiner & Winer 2007
  - olfactory   (pid 34, 35) — Mori 2014
  - amygdala    (pid 38)     — Janak & Tye 2015
  - hippocampal (pid 39-42)  — Strange et al. 2014

These are layered on top of the 21 Garin point anchors. Reports Beauchamp
top-K across all 19 evaluable pairs, region-level eval (Beauchamp-22
candidate set), and the per-pair before/after comparison.

Outputs:
  - outputs/coupling/pi_fc_plus_SC_with_all_packs.npy
  - outputs/logs/beauchamp_validation_all_packs.json
  - outputs/logs/region_level_eval_all_packs.json

Usage:
    PYTHONPATH=src python experiments/compose_all/01_compose_all_default_packs.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from homer.data import load_cached                                  # noqa: E402
from homer.data.anchor_packs import (                                # noqa: E402
    DEFAULT_PACK_NAMES,
    build_default_pack_entries,
)
from homer.models import MultimodalFGW                              # noqa: E402

ANN  = ROOT / "outputs" / "anndata"
COUP = ROOT / "outputs" / "coupling"
LOG  = ROOT / "outputs" / "logs"


def main():
    M, _ = load_cached("mouse", cache_dir=ANN)
    H, _ = load_cached("human", cache_dir=ANN)
    costs = np.load(ANN / "full_costs.npz")
    print(f"Mouse parcels: {len(M.var)}, human parcels: {len(H.var)}")

    # Compose all default packs (registry is the single source of truth).
    print(f"\nBuilding all {len(DEFAULT_PACK_NAMES)} default anchor packs "
          f"({', '.join(DEFAULT_PACK_NAMES)}) ...")
    entries = build_default_pack_entries(M.var, H.var, atlas_root=ROOT)
    print(f"\n{len(entries)} region-anchor entries:")
    for e in entries:
        print(f"  pid={e.pair_id:>3d}  {e.label!r:60s} "
              f"|m|={len(e.mouse_indices):3d} |h|={len(e.human_indices):3d}")

    # Detect any shared human parcels (composition overlaps)
    print("\nHuman-side overlap audit:")
    n_overlap = 0
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            shared = set(entries[i].human_indices) & set(entries[j].human_indices)
            if shared:
                n_overlap += 1
                print(f"  pid={entries[i].pair_id} ∩ pid={entries[j].pair_id}: "
                      f"{len(shared)} shared parcels")
    if not n_overlap:
        print("  (none)")

    # Fit
    print("\nFitting MultimodalFGW with all 11 region anchors ...")
    model = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7,
                           epsilon=5e-3, xyz_weight=0.5, lam_anchor=1.0)
    model.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"],
               region_anchors=entries)
    out = COUP / "pi_fc_plus_SC_with_all_packs.npy"
    np.save(out, model.pi)
    print(f"  saved {out}  loss={model.fit_info_.loss:.4g}  "
          f"converged={model.fit_info_.converged}")

    # Beauchamp validation
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    print("\nRunning Beauchamp validation ...")
    subprocess.run(["python", str(ROOT / "pipeline" / "05f_beauchamp_validation.py"),
                     "--pi-file", out.name], env=env, check=True, capture_output=True)
    d_all = json.loads((LOG / "beauchamp_validation.json").read_text())
    (LOG / "beauchamp_validation.json").rename(LOG / "beauchamp_validation_all_packs.json")
    # Re-run production
    subprocess.run(["python", str(ROOT / "pipeline" / "05f_beauchamp_validation.py"),
                     "--pi-file", "pi_fc_plus_SC.npy"], env=env, check=True, capture_output=True)
    d_prod = json.loads((LOG / "beauchamp_validation.json").read_text())

    # Region-level eval
    print("\nRunning region-level eval ...")
    subprocess.run(["python", str(ROOT / "pipeline" / "05j_region_level_eval.py"),
                     "--pi-file", out.name, "--output", "region_level_eval_all_packs.json",
                     "--skip-nulls"],
                    env=env, check=True, capture_output=True)
    rl_all = json.loads((LOG / "region_level_eval_all_packs.json").read_text())
    rl_prod = json.loads((LOG / "region_level_eval.json").read_text())

    # ---- Full Beauchamp comparison
    key_pairs = [
        ('Mot', 'Primary motor area -> precentral gyrus'),
        ('SC',  'Superior colliculus, sensory related -> superior colliculus'),
        ('IC',  'Inferior colliculus -> inferior colliculus'),
        ('Pir', 'Piriform area -> piriform cortex'),
        ('Amg', 'Cortical subplate-other -> amygdala'),
        ('Sub', 'Subiculum -> subiculum'),
        ('CA1', 'Field CA1 -> CA1 field'),
        ('CA3', 'Field CA3 -> CA3 field'),
        ('DG',  'Dentate gyrus -> dentate gyrus'),
        # Not in any pack but anchor-overlapping
        ('Thal', 'Thalamus -> thalamus'),
        ('Aud', "Primary auditory area -> Heschl's gyrus"),
        ('S1',  'Primary somatosensory area -> postcentral gyrus'),
        ('ACG', 'Anterior cingulate area -> cingulate gyrus'),
        ('V1',  'Visual areas -> cuneus'),
        ('NAc', 'Striatum ventral region -> nucleus accumbens'),
        ('Cau', 'Caudoputamen -> caudate nucleus'),
        ('Hyp', 'Hypothalamus -> hypothalamus'),
        ('Pal', 'Pallidum -> globus pallidus'),
        ('Pon', 'Pons -> pons'),
    ]
    pack_anchored = {'Mot','SC','IC','Pir','Amg','Sub','CA1','CA3','DG'}

    print("\n" + "=" * 75)
    print(f"{'region':6s} {'prod':>7s} {'+all packs':>12s} {'Δ':>7s} {'note':>20s}")
    print("-" * 75)
    for short, name in key_pairs:
        p = d_prod.get(name, {}).get('top1')
        a = d_all.get(name, {}).get('top1')
        if p is None or a is None: continue
        d_pp = (a - p) * 100
        note = "  ← pack-anchored" if short in pack_anchored else ""
        flag = "  ↑" if d_pp > 1 else ("  ↓" if d_pp < -1 else "")
        print(f"  {short:4s} {p*100:>6.0f}%  {a*100:>10.0f}%  {d_pp:>+5.0f}pp{note}{flag}")

    # ---- Aggregate Beauchamp metrics (anchor-overlapping pairs)
    print("\n--- Aggregate Beauchamp metrics (anchor-overlapping, 15 pairs / 927 parcels) ---")
    def agg(d, key, anchor_only=True):
        keys = [k for k in d if not k.startswith('_') and 'skip_reason' not in d[k]]
        group = [d[k] for k in keys if d[k].get('is_anchor_overlapping') is (True if anchor_only else None) or not anchor_only]
        anchor = [d[k] for k in keys if d[k].get('is_anchor_overlapping')]
        w = np.array([g['n_mouse_parcels'] for g in anchor], dtype=float)
        return float(((np.array([g[key] for g in anchor]) * w).sum()) / w.sum())
    print(f"  top-1:     prod = {agg(d_prod,'top1'):.1%}     +all packs = {agg(d_all,'top1'):.1%}")
    print(f"  top-5:     prod = {agg(d_prod,'top5'):.1%}     +all packs = {agg(d_all,'top5'):.1%}")
    print(f"  top-10:    prod = {agg(d_prod,'top10'):.1%}     +all packs = {agg(d_all,'top10'):.1%}")
    print(f"  mean rank: prod = {agg(d_prod,'mean_rank_in_region'):.0f}      +all packs = {agg(d_all,'mean_rank_in_region'):.0f}")

    # ---- Region-level eval
    print("\n--- Region-level eval (Beauchamp-22 candidate set) ---")
    for d, label in [(rl_prod, 'prod'), (rl_all, '+all packs')]:
        if 'anchor_overlapping' not in d: continue
        d2 = d['anchor_overlapping']
        print(f"  {label}: qualified top-1 = {d2['qualified_top_k']['1']:.1%},   "
              f"top-3 = {d2['qualified_top_k']['3']:.1%},   "
              f"top-5 = {d2['qualified_top_k']['5']:.1%}    "
              f"mean fold = {d2['mean_fold_enrichment']:.1f}x")


if __name__ == "__main__":
    main()
