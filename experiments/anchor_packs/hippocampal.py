"""HIPPOCAMPAL-1: Add Subiculum + CA1 + CA3 + Dentate gyrus region anchors.

All five hippocampal subfields show 0 % Beauchamp top-1 under production
point-anchor π — HOMER's cleanest documented failure region. Earlier
EXP-1 / SPLIT-1 added four hippocampal *point* anchors and moved 3 of 4
from 0 → 7-9 % top-1. This experiment is the region-anchor analogue:
each subfield's full DSURQE parcel set is forced into the matching human
subfield MNI ball.

  pid 39: Subiculum (29 mouse, 8 human)
  pid 40: CA1       (15 mouse, 6 human)
  pid 41: CA3       (26 mouse, 4 human)
  pid 42: Dentate gyrus (22 mouse, 4 human)

Outputs:
  - outputs/coupling/pi_fc_plus_SC_with_hippocampal.npy
  - outputs/coupling/pi_fc_plus_SC_with_hippocampal_subi_only.npy   (held-out)
  - outputs/logs/beauchamp_validation_hippocampal.json
  - outputs/logs/beauchamp_validation_hippocampal_subi_only.json

Usage:
    PYTHONPATH=src python experiments/hippocampal/01_add_hippocampal_anchors.py
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

from homer.data import load_cached                                       # noqa: E402
from homer.data.anchor_packs import build_hippocampal_region_anchors    # noqa: E402
from homer.models import MultimodalFGW                                   # noqa: E402

ANN  = ROOT / "outputs" / "anndata"
COUP = ROOT / "outputs" / "coupling"
LOG  = ROOT / "outputs" / "logs"


def fit_and_validate(M, H, costs, *, region_anchors, pi_filename, val_filename):
    model = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7,
                           epsilon=5e-3, xyz_weight=0.5, lam_anchor=1.0)
    model.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"],
               region_anchors=region_anchors)
    out_pi = COUP / pi_filename
    np.save(out_pi, model.pi)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(
        ["python", str(ROOT / "pipeline" / "05f_beauchamp_validation.py"),
         "--pi-file", out_pi.name],
        env=env, check=True, capture_output=True,
    )
    src = LOG / "beauchamp_validation.json"
    dst = LOG / val_filename
    if src != dst:
        src.rename(dst)
    return json.loads(dst.read_text())


def main():
    M, _ = load_cached("mouse", cache_dir=ANN)
    H, _ = load_cached("human", cache_dir=ANN)
    costs = np.load(ANN / "full_costs.npz")
    print(f"Mouse parcels: {len(M.var)}, human parcels: {len(H.var)}")

    entries = build_hippocampal_region_anchors(M.var, H.var, atlas_root=ROOT)
    print("\nHippocampal anchors:")
    for e in entries:
        print(f"  pid={e.pair_id:>3d}  {e.label!r}  |mouse|={len(e.mouse_indices)}  "
              f"|human|={len(e.human_indices)}")

    print("\n[1/3] Fitting production + hippocampal pack (4 subfields) ...")
    d_full = fit_and_validate(
        M, H, costs, region_anchors=entries,
        pi_filename="pi_fc_plus_SC_with_hippocampal.npy",
        val_filename="beauchamp_validation_hippocampal.json",
    )

    print("\n[2/3] Fitting production + Subiculum anchor only (CA1/CA3/DG held out) ...")
    d_subi = fit_and_validate(
        M, H, costs, region_anchors=[entries[0]],   # Subiculum only
        pi_filename="pi_fc_plus_SC_with_hippocampal_subi_only.npy",
        val_filename="beauchamp_validation_hippocampal_subi_only.json",
    )

    print("\n[3/3] Re-running Beauchamp on production π for comparison ...")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(["python", str(ROOT / "pipeline" / "05f_beauchamp_validation.py"),
                     "--pi-file", "pi_fc_plus_SC.npy"], env=env, check=True, capture_output=True)
    d_prod = json.loads((LOG / "beauchamp_validation.json").read_text())

    key_pairs = [
        ('Sub',   'Subiculum -> subiculum'),
        ('CA1',   'Field CA1 -> CA1 field'),
        ('CA3',   'Field CA3 -> CA3 field'),
        ('DG',    'Dentate gyrus -> dentate gyrus'),
        ('Thal',  'Thalamus -> thalamus'),
        ('Mot',   'Primary motor area -> precentral gyrus'),
        ('Aud',   "Primary auditory area -> Heschl's gyrus"),
        ('ACG',   'Anterior cingulate area -> cingulate gyrus'),
        ('S1',    'Primary somatosensory area -> postcentral gyrus'),
        ('Hyp',   'Hypothalamus -> hypothalamus'),
    ]

    print("\n" + "=" * 90)
    print(f"{'region':6s} {'prod':>7s} {'+hippo':>9s} {'+Subi only':>12s} "
          f"{'Δ full':>8s} {'Δ Subi-only':>12s}  notes")
    print("-" * 90)
    for short, name in key_pairs:
        p = d_prod.get(name, {}).get('top1')
        f = d_full.get(name, {}).get('top1')
        s = d_subi.get(name, {}).get('top1')
        if None in (p, f, s): continue
        d_full_pp = (f - p) * 100
        d_subi_pp = (s - p) * 100
        marker = "  ←hippo" if short in ('Sub','CA1','CA3','DG') else ""
        flag = "  ↑" if d_full_pp > 1 else ("  ↓" if d_full_pp < -1 else "")
        print(f"  {short:4s} {p*100:>6.0f}%  {f*100:>7.0f}%  {s*100:>10.0f}%  "
              f"{d_full_pp:>+6.0f}pp  {d_subi_pp:>+10.0f}pp{marker}{flag}")


if __name__ == "__main__":
    main()
