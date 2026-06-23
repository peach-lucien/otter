"""BICCN-MOTOR-1: Add motor sub-region anchors and test for Motor recovery.

Bakken et al. 2021 (Nature; the BICCN Motor Cortex Consortium) identifies two
strongly-conserved motor sub-region homologies across mouse and human:

  - Mouse Primary motor area (M1)  ↔ Human Area 4 / BA4 (primary motor)
  - Mouse Secondary motor area (M2) ↔ Human Area 6 dorsal premotor (PMd)

Our current 21 Garin anchors include one Motor anchor (pair_id 2) at a single
mouse parcel that the colleague's preprocessing labels as "Motor and premotor"
(containing only Secondary motor area / M2 anatomically) mapped to one human
parcel that lumps Primary Motor + Premotor + FEF + SMA + others. The
Beauchamp validation shows this single anchor gives Motor → precentral
top-1 = 0 %.

This experiment adds the two BICCN-aligned region anchors:

  pair_id 30: M1  (53 mouse Primary motor parcels, via DSURQE atlas overlay)
              ↔  human BA4 (parcels within 10 mm of MNI ±37,-22,55)
  pair_id 31: M2  (48 mouse Secondary motor parcels)
              ↔  human PMd (parcels within 12 mm of MNI ±28,-5,62)

Soft constraint (lam_outside=0.15 default) so the anchors push without
forbidding alternatives, see docs/archive/iteration_log.md §5.6.0a.

Outputs:
  - outputs/coupling/pi_fc_plus_SC_with_biccn_motor.npy
  - outputs/logs/beauchamp_validation_biccn_motor.json

Usage:
    PYTHONPATH=src python experiments/biccn_motor/01_add_motor_subregion_anchors.py
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
sys.path.insert(0, str(ROOT / "pipeline"))

from homer.data import load_cached                                    # noqa: E402
from homer.data.anchor_packs import build_biccn_motor_region_anchors  # noqa: E402
from homer.models import MultimodalFGW                                # noqa: E402

ANN  = ROOT / "outputs" / "anndata"
COUP = ROOT / "outputs" / "coupling"
LOG  = ROOT / "outputs" / "logs"
EXT  = ROOT / "data_external" / "MouseHumanTranscriptomicSimilarity"


def main():
    M, _ = load_cached("mouse", cache_dir=ANN)
    H, _ = load_cached("human", cache_dir=ANN)
    costs = np.load(ANN / "full_costs.npz")
    print(f"Mouse parcels: {len(M.var)}, human parcels: {len(H.var)}")

    entries = build_biccn_motor_region_anchors(M.var, H.var, atlas_root=ROOT)
    print("\nBICCN region anchors:")
    for e in entries:
        print(f"  pair_id={e.pair_id:>3d}  {e.label!r}  |mouse|={len(e.mouse_indices)}  "
              f"|human|={len(e.human_indices)}")

    print("\nFitting MultimodalFGW with point anchors + 2 BICCN region anchors ...")
    model = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7,
                           epsilon=5e-3, xyz_weight=0.5, lam_anchor=1.0)
    model.fit(M, H,
              Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"],
              region_anchors=entries)        # default region_lam_outside=0.15 (soft)
    out = COUP / "pi_fc_plus_SC_with_biccn_motor.npy"
    np.save(out, model.pi)
    print(f"  saved {out}")
    print(f"  loss={model.fit_info_.loss:.4g}  converged={model.fit_info_.converged}")

    # Beauchamp validation
    print("\nRunning Beauchamp validation ...")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(["python", str(ROOT / "pipeline" / "05f_beauchamp_validation.py"),
                     "--pi-file", out.name], env=env, check=True, capture_output=True)
    d_biccn = json.loads((LOG / "beauchamp_validation.json").read_text())
    (LOG / "beauchamp_validation.json").rename(LOG / "beauchamp_validation_biccn_motor.json")

    # Re-run production for fresh comparison
    subprocess.run(["python", str(ROOT / "pipeline" / "05f_beauchamp_validation.py"),
                     "--pi-file", "pi_fc_plus_SC.npy"], env=env, check=True, capture_output=True)
    d_prod = json.loads((LOG / "beauchamp_validation.json").read_text())

    key_pairs = [
        ('Mot',   'Primary motor area -> precentral gyrus'),
        ('S1',    'Primary somatosensory area -> postcentral gyrus'),
        ('Aud',   "Primary auditory area -> Heschl's gyrus"),
        ('ACG',   'Anterior cingulate area -> cingulate gyrus'),
        ('V1',    'Visual areas -> cuneus'),
        ('Cau',   'Caudoputamen -> caudate nucleus'),
        ('NAc',   'Striatum ventral region -> nucleus accumbens'),
        ('Hyp',   'Hypothalamus -> hypothalamus'),
        ('Pal',   'Pallidum -> globus pallidus'),
        ('Pir',   'Piriform area -> piriform cortex'),
        ('IC',    'Inferior colliculus -> inferior colliculus'),
        ('SC',    'Superior colliculus, sensory related -> superior colliculus'),
        ('Pon',   'Pons -> pons'),
        ('Thal',  'Thalamus -> thalamus'),
        ('Sub',   'Subiculum -> subiculum'),
    ]

    print("\n" + "=" * 80)
    print(f"{'region':6s} {'prod':>7s} {'+BICCN':>9s} {'Δ':>7s} {'top5 prod→biccn':>17s}")
    print("-" * 80)
    for short, name in key_pairs:
        p = d_prod.get(name, {}); b = d_biccn.get(name, {})
        t1p, t1b = p.get('top1'), b.get('top1')
        t5p, t5b = p.get('top5'), b.get('top5')
        if None in (t1p, t1b): continue
        marker = "  ←motor" if short == 'Mot' else ""
        d1 = (t1b - t1p) * 100
        flag = "  ↑" if d1 > 1 else ("  ↓" if d1 < -1 else "")
        print(f"  {short:4s} {t1p*100:>6.0f}%  {t1b*100:>7.0f}%  {d1:>+5.0f}pp  "
              f"{t5p*100:>5.0f}% → {t5b*100:>3.0f}%{marker}{flag}")


if __name__ == "__main__":
    main()
