"""Add the amygdala / cortical subplate region anchor.

This pack covers the remaining 0 % Beauchamp top-1 failure pair without
dedicated sub-region supervision ("Cortical subplate-other → amygdala").

  pid 38: Mouse Cortical subplate (54 parcels) ↔ Human amygdala (6 parcels)

Single-entry pack. DSURQE does not distinguish amygdala sub-nuclei, so no
held-out test is possible: with one entry there is nothing to hold out and
compare against the FC/SC-driven recovery.

Outputs:
  - outputs/coupling/pi_fc_plus_SC_with_amygdala.npy
  - outputs/logs/beauchamp_validation_amygdala.json

Usage:
    PYTHONPATH=src python experiments/anchor_packs/amygdala.py
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

from otter.data import load_cached                                  # noqa: E402
from otter.data.anchor_packs import build_amygdala_region_anchors  # noqa: E402
from otter.models import MultimodalFGW                              # noqa: E402

ANN  = ROOT / "outputs" / "anndata"
COUP = ROOT / "outputs" / "coupling"
LOG  = ROOT / "outputs" / "logs"


def main():
    M, _ = load_cached("mouse", cache_dir=ANN)
    H, _ = load_cached("human", cache_dir=ANN)
    costs = np.load(ANN / "full_costs.npz")
    print(f"Mouse parcels: {len(M.var)}, human parcels: {len(H.var)}")

    entries = build_amygdala_region_anchors(M.var, H.var, atlas_root=ROOT)
    print("\nAmygdala anchor:")
    for e in entries:
        print(f"  pid={e.pair_id:>3d}  {e.label!r}  |mouse|={len(e.mouse_indices)}  "
              f"|human|={len(e.human_indices)}")

    print("\nFitting production + amygdala pack ...")
    model = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7,
                           epsilon=5e-3, xyz_weight=0.5, lam_anchor=1.0)
    model.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"],
               region_anchors=entries)
    out = COUP / "pi_fc_plus_SC_with_amygdala.npy"
    np.save(out, model.pi)

    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(
        ["python", str(ROOT / "pipeline" / "05f_beauchamp_validation.py"),
         "--pi-file", out.name],
        env=env, check=True, capture_output=True,
    )
    d_amg = json.loads((LOG / "beauchamp_validation.json").read_text())
    (LOG / "beauchamp_validation.json").rename(LOG / "beauchamp_validation_amygdala.json")

    # Re-run production
    subprocess.run(["python", str(ROOT / "pipeline" / "05f_beauchamp_validation.py"),
                     "--pi-file", "pi_fc_plus_SC.npy"], env=env, check=True, capture_output=True)
    d_prod = json.loads((LOG / "beauchamp_validation.json").read_text())

    key_pairs = [
        ('Amg',   'Cortical subplate-other -> amygdala'),
        ('Mot',   'Primary motor area -> precentral gyrus'),
        ('SC',    'Superior colliculus, sensory related -> superior colliculus'),
        ('IC',    'Inferior colliculus -> inferior colliculus'),
        ('Pir',   'Piriform area -> piriform cortex'),
        ('Sub',   'Subiculum -> subiculum'),
        ('Aud',   "Primary auditory area -> Heschl's gyrus"),
        ('S1',    'Primary somatosensory area -> postcentral gyrus'),
        ('Thal',  'Thalamus -> thalamus'),
        ('Hyp',   'Hypothalamus -> hypothalamus'),
        ('ACG',   'Anterior cingulate area -> cingulate gyrus'),
    ]

    print("\n" + "=" * 70)
    print(f"{'region':6s} {'prod':>7s} {'+amyg':>8s} {'Δ':>7s}  notes")
    print("-" * 70)
    for short, name in key_pairs:
        p = d_prod.get(name, {}).get('top1')
        a = d_amg.get(name, {}).get('top1')
        if p is None or a is None: continue
        d_pp = (a - p) * 100
        marker = "  ←amygdala" if short == 'Amg' else ""
        flag = "  ↑" if d_pp > 1 else ("  ↓" if d_pp < -1 else "")
        print(f"  {short:4s} {p*100:>6.0f}%  {a*100:>6.0f}%  {d_pp:>+5.0f}pp{marker}{flag}")


if __name__ == "__main__":
    main()
