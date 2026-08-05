"""ENTORHINAL-1: Add entorhinal cortex anchor (Franjic 2021).

Single-entry pack. DSURQE doesn't expose medial entorhinal. The whole-EC
anchor captures the broad cross-species EC homology documented by Franjic
2021 (transcriptomic taxonomy across human, macaque, pig).

  pid 49: Mouse Entorhinal area (84 parcels) ↔ Human entorhinal cortex (6 parcels)

Outputs:
  - outputs/coupling/pi_fc_plus_SC_with_entorhinal.npy
  - outputs/logs/beauchamp_validation_entorhinal.json

Beauchamp doesn't have an entorhinal validation pair, so the direct
effect isn't measurable; the experiment confirms zero off-target effects
on other Beauchamp pairs.

Usage:
    PYTHONPATH=src python experiments/anchor_packs/entorhinal.py
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

from otter.data import load_cached                                       # noqa: E402
from otter.data.anchor_packs import build_entorhinal_region_anchors     # noqa: E402
from otter.models import MultimodalFGW                                   # noqa: E402

ANN = ROOT / "outputs" / "anndata"
COUP = ROOT / "outputs" / "coupling"
LOG = ROOT / "outputs" / "logs"


def main():
    M, _ = load_cached("mouse", cache_dir=ANN)
    H, _ = load_cached("human", cache_dir=ANN)
    costs = np.load(ANN / "full_costs.npz")
    entries = build_entorhinal_region_anchors(M.var, H.var, atlas_root=ROOT)
    print("Entorhinal anchor:")
    for e in entries:
        print(f"  pid={e.pair_id:>3d}  {e.label!r}  |m|={len(e.mouse_indices)}  "
              f"|h|={len(e.human_indices)}")

    model = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7,
                          epsilon=5e-3, xyz_weight=0.5, lam_anchor=1.0)
    model.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"],
              region_anchors=entries)
    out = COUP / "pi_fc_plus_SC_with_entorhinal.npy"
    np.save(out, model.pi)

    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(
        ["python", str(ROOT / "pipeline" / "05f_beauchamp_validation.py"),
         "--pi-file", out.name],
        env=env, check=True, capture_output=True,
    )
    src = LOG / "beauchamp_validation.json"
    dst = LOG / "beauchamp_validation_entorhinal.json"
    if src != dst:
        src.rename(dst)
    d_ent = json.loads(dst.read_text())

    subprocess.run(["python", str(ROOT / "pipeline" / "05f_beauchamp_validation.py"),
                    "--pi-file", "pi_fc_plus_SC.npy"], env=env, check=True, capture_output=True)
    d_prod = json.loads((LOG / "beauchamp_validation.json").read_text())

    print(f"\nOff-target check (no Beauchamp entorhinal pair exists):")
    print(f"{'region':6s} {'prod':>7s} {'+entorh':>10s} {'Δ':>7s}")
    for short, name in [
        ('Sub', 'Subiculum -> subiculum'),
        ('CA1', 'Field CA1 -> CA1 field'),
        ('CA3', 'Field CA3 -> CA3 field'),
        ('Thal','Thalamus -> thalamus'),
        ('Cau','Caudoputamen -> caudate nucleus'),
        ('Aud',"Primary auditory area -> Heschl's gyrus"),
        ('S1', 'Primary somatosensory area -> postcentral gyrus'),
    ]:
        p = d_prod.get(name, {}).get('top1')
        e = d_ent.get(name, {}).get('top1')
        if p is None or e is None: continue
        print(f"  {short:4s} {p*100:>6.0f}%  {e*100:>8.0f}%  {(e-p)*100:>+5.0f}pp")


if __name__ == "__main__":
    main()
