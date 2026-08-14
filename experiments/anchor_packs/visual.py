"""Add mouse LM ↔ human V2 region anchor (Wang & Burkhalter 2007).

Single-entry pack. Mouse Lateral visual area (LM, 9 parcels) ↔ human V2
at MNI(±20, –85, 10) r=10 mm (12 parcels).

Beauchamp validates "Visual areas → cuneus" using all 54 mouse Visual
parcels → human cuneus at (±10, -85, 5). This pack uses a subset (LM, 9 of
54) → a different human target (V2 lateral, not cuneus), so its effect on
Beauchamp Visual→cuneus is null, 7 % → 7 %. It makes the LM↔V2
correspondence explicit.

Usage:
    PYTHONPATH=src python experiments/anchor_packs/visual.py
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

from otter.data import load_cached                                   # noqa: E402
from otter.data.anchor_packs import build_visual_region_anchors     # noqa: E402
from otter.models import MultimodalFGW                               # noqa: E402

ANN = ROOT / "outputs" / "anndata"
COUP = ROOT / "outputs" / "coupling"
LOG = ROOT / "outputs" / "logs"


def main():
    M, _ = load_cached("mouse", cache_dir=ANN)
    H, _ = load_cached("human", cache_dir=ANN)
    costs = np.load(ANN / "full_costs.npz")
    entries = build_visual_region_anchors(M.var, H.var, atlas_root=ROOT)
    for e in entries:
        print(f"  pid={e.pair_id}  {e.label!r}  |m|={len(e.mouse_indices)}  "
              f"|h|={len(e.human_indices)}")

    model = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7,
                          epsilon=5e-3, xyz_weight=0.5, lam_anchor=1.0)
    model.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"],
              region_anchors=entries)
    out = COUP / "pi_fc_plus_SC_with_visual.npy"
    np.save(out, model.pi)

    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(["python", str(ROOT / "pipeline" / "05f_beauchamp_validation.py"),
                    "--pi-file", out.name], env=env, check=True, capture_output=True)
    src = LOG / "beauchamp_validation.json"
    dst = LOG / "beauchamp_validation_visual.json"
    if src != dst:
        src.rename(dst)
    d_vis = json.loads(dst.read_text())
    subprocess.run(["python", str(ROOT / "pipeline" / "05f_beauchamp_validation.py"),
                    "--pi-file", "pi_fc_plus_SC.npy"], env=env, check=True, capture_output=True)
    d_prod = json.loads((LOG / "beauchamp_validation.json").read_text())

    for short, name in [
        ('V1',  'Visual areas -> cuneus'),
        ('Mot', 'Primary motor area -> precentral gyrus'),
        ('Aud', "Primary auditory area -> Heschl's gyrus"),
        ('S1',  'Primary somatosensory area -> postcentral gyrus'),
    ]:
        p = d_prod.get(name, {}).get('top1'); v = d_vis.get(name, {}).get('top1')
        if p is None or v is None: continue
        print(f"  {short}: {p*100:.0f}% → {v*100:.0f}% ({(v-p)*100:+.0f}pp)")


if __name__ == "__main__":
    main()
