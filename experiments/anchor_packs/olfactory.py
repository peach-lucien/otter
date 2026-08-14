"""OLFACTORY-1: Add Piriform + AON sub-region anchors.

Olfactory cortex is a documented OTTER failure region. Beauchamp gives
0 % top-1 for "Piriform area → piriform cortex" under the production
point-anchor π (mean rank 657 / 2094). The Garin pair_id 11 (Olfactory
cortex) is a single point anchor in the Piriform parcel.

This experiment adds the two olfactory anchor-pack entries:

  pid 34: Mouse Piriform area ↔ Human Piriform cortex (Mori 2014; Carlén 2017)
  pid 35: Mouse Anterior olfactory nucleus ↔ Human AON (Mori 2014)

Tests three configurations on Beauchamp top-1:

  1. Production fc_plus_SC point-anchor π          (baseline)
  2. Production + olfactory pack                   (full)
  3. Production + Piriform anchor only (AON held)  (held-out generalization)

Outputs:
  - outputs/coupling/pi_fc_plus_SC_with_olfactory.npy
  - outputs/coupling/pi_fc_plus_SC_with_olfactory_pir_only.npy
  - outputs/logs/beauchamp_validation_olfactory.json
  - outputs/logs/beauchamp_validation_olfactory_pir_only.json

Usage:
    PYTHONPATH=src python experiments/olfactory/01_add_olfactory_anchors.py
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

from otter.data import load_cached                                    # noqa: E402
from otter.data.anchor_packs import build_olfactory_region_anchors   # noqa: E402
from otter.models import MultimodalFGW                                # noqa: E402

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

    entries = build_olfactory_region_anchors(M.var, H.var, atlas_root=ROOT)
    print("\nOlfactory anchors:")
    for e in entries:
        print(f"  pid={e.pair_id:>3d}  {e.label!r}  |mouse|={len(e.mouse_indices)}  "
              f"|human|={len(e.human_indices)}")

    print("\n[1/3] Fitting production + olfactory pack (Piriform + AON) ...")
    d_full = fit_and_validate(
        M, H, costs, region_anchors=entries,
        pi_filename="pi_fc_plus_SC_with_olfactory.npy",
        val_filename="beauchamp_validation_olfactory.json",
    )

    print("\n[2/3] Fitting production + Piriform anchor only (AON held out) ...")
    d_pir = fit_and_validate(
        M, H, costs, region_anchors=[entries[0]],   # Piriform only
        pi_filename="pi_fc_plus_SC_with_olfactory_pir_only.npy",
        val_filename="beauchamp_validation_olfactory_pir_only.json",
    )

    print("\n[3/3] Re-running Beauchamp on production π for comparison ...")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(["python", str(ROOT / "pipeline" / "05f_beauchamp_validation.py"),
                     "--pi-file", "pi_fc_plus_SC.npy"], env=env, check=True, capture_output=True)
    d_prod = json.loads((LOG / "beauchamp_validation.json").read_text())

    key_pairs = [
        ('Pir',   'Piriform area -> piriform cortex'),
        ('Mot',   'Primary motor area -> precentral gyrus'),
        ('SC',    'Superior colliculus, sensory related -> superior colliculus'),
        ('IC',    'Inferior colliculus -> inferior colliculus'),
        ('Aud',   "Primary auditory area -> Heschl's gyrus"),
        ('Thal',  'Thalamus -> thalamus'),
        ('Hyp',   'Hypothalamus -> hypothalamus'),
        ('Pon',   'Pons -> pons'),
        ('ACG',   'Anterior cingulate area -> cingulate gyrus'),
        ('V1',    'Visual areas -> cuneus'),
        ('S1',    'Primary somatosensory area -> postcentral gyrus'),
    ]

    print("\n" + "=" * 90)
    print(f"{'region':6s} {'prod':>7s} {'+olfact':>10s} {'+Pir only':>11s} "
          f"{'Δ full':>8s} {'Δ Pir-only':>11s}  notes")
    print("-" * 90)
    for short, name in key_pairs:
        p = d_prod.get(name, {}).get('top1')
        f = d_full.get(name, {}).get('top1')
        pir = d_pir.get(name, {}).get('top1')
        if None in (p, f, pir): continue
        d_full_pp = (f - p) * 100
        d_pir_pp = (pir - p) * 100
        marker = "  ←olfactory" if short == 'Pir' else ""
        flag_full = "  ↑" if d_full_pp > 1 else ("  ↓" if d_full_pp < -1 else "")
        print(f"  {short:4s} {p*100:>6.0f}%  {f*100:>8.0f}%  {pir*100:>9.0f}%  "
              f"{d_full_pp:>+6.0f}pp  {d_pir_pp:>+9.0f}pp{marker}{flag_full}")

    print(f"\n{'-' * 90}")
    print("Δ full = production → +olfactory (both anchors).")
    print("Δ Pir-only = production → +Piriform anchor only (AON held out).")


if __name__ == "__main__":
    main()
