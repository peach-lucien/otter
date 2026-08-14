"""Add subgenual ACC + Retrosplenial region anchors (Vogt 2019).

Beauchamp validation gives 13 % top-1 for "Anterior cingulate area →
cingulate gyrus", above the other failure regions. This pack adds Vogt
2019's two best-conserved cingulate sub-domains. Its mouse-side set differs
from the corresponding Beauchamp validation set, so the Beauchamp
comparison is informative rather than tautological.

  pid 36: Mouse ACA ventral (15 parcels)   ↔ Human subgenual ACC (6 parcels)
  pid 37: Mouse Retrosplenial (27 parcels) ↔ Human RSC (8 parcels)

Beauchamp's ACG validation uses the full mouse ACA (23 parcels, including
the dorsal part omitted here) and a different human centroid (pregenual ACC
at ±5,25,25 r=15). This pack uses mouse ACA-ventral (15 of 23) and human
subgenual at ±5,10,35 r=10. The intersection is small, so the Beauchamp ACG
metric is a partly-independent measurement.

Outputs:
  - outputs/coupling/pi_fc_plus_SC_with_cingulate.npy
  - outputs/coupling/pi_fc_plus_SC_with_cingulate_rsc_only.npy
  - outputs/logs/beauchamp_validation_cingulate.json
  - outputs/logs/beauchamp_validation_cingulate_rsc_only.json

Usage:
    PYTHONPATH=src python experiments/anchor_packs/cingulate.py
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
from otter.data.anchor_packs import build_cingulate_region_anchors   # noqa: E402
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

    entries = build_cingulate_region_anchors(M.var, H.var, atlas_root=ROOT)
    print("\nCingulate anchors:")
    for e in entries:
        print(f"  pid={e.pair_id:>3d}  {e.label!r}  |mouse|={len(e.mouse_indices)}  "
              f"|human|={len(e.human_indices)}")

    print("\n[1/3] Fitting production + cingulate pack (ACC subgenual + RSC) ...")
    d_full = fit_and_validate(
        M, H, costs, region_anchors=entries,
        pi_filename="pi_fc_plus_SC_with_cingulate.npy",
        val_filename="beauchamp_validation_cingulate.json",
    )

    print("\n[2/3] Fitting production + RSC anchor only (ACC held out) ...")
    d_rsc = fit_and_validate(
        M, H, costs, region_anchors=[entries[1]],   # RSC only
        pi_filename="pi_fc_plus_SC_with_cingulate_rsc_only.npy",
        val_filename="beauchamp_validation_cingulate_rsc_only.json",
    )

    print("\n[3/3] Re-running Beauchamp on production π for comparison ...")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(["python", str(ROOT / "pipeline" / "05f_beauchamp_validation.py"),
                     "--pi-file", "pi_fc_plus_SC.npy"], env=env, check=True, capture_output=True)
    d_prod = json.loads((LOG / "beauchamp_validation.json").read_text())

    key_pairs = [
        ('ACG',   'Anterior cingulate area -> cingulate gyrus'),
        ('Mot',   'Primary motor area -> precentral gyrus'),
        ('SC',    'Superior colliculus, sensory related -> superior colliculus'),
        ('IC',    'Inferior colliculus -> inferior colliculus'),
        ('Aud',   "Primary auditory area -> Heschl's gyrus"),
        ('S1',    'Primary somatosensory area -> postcentral gyrus'),
        ('V1',    'Visual areas -> cuneus'),
        ('Thal',  'Thalamus -> thalamus'),
        ('Hyp',   'Hypothalamus -> hypothalamus'),
        ('Pir',   'Piriform area -> piriform cortex'),
    ]

    print("\n" + "=" * 90)
    print(f"{'region':6s} {'prod':>7s} {'+cingul':>9s} {'+RSC only':>11s} "
          f"{'Δ full':>8s} {'Δ RSC-only':>11s}  notes")
    print("-" * 90)
    for short, name in key_pairs:
        p = d_prod.get(name, {}).get('top1')
        f = d_full.get(name, {}).get('top1')
        r = d_rsc.get(name, {}).get('top1')
        if None in (p, f, r): continue
        d_full_pp = (f - p) * 100
        d_rsc_pp = (r - p) * 100
        marker = "  ←cingul" if short == 'ACG' else ""
        flag = "  ↑" if d_full_pp > 1 else ("  ↓" if d_full_pp < -1 else "")
        print(f"  {short:4s} {p*100:>6.0f}%  {f*100:>7.0f}%  {r*100:>9.0f}%  "
              f"{d_full_pp:>+6.0f}pp  {d_rsc_pp:>+9.0f}pp{marker}{flag}")


if __name__ == "__main__":
    main()
