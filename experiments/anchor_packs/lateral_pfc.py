"""Add OFC + dlPFC region anchors.

Coverage of lateral PFC regions that Beauchamp does not validate but that
matter for downstream cognitive-neuroscience use cases (decision-making,
working memory, executive control).

  pid 45: Mouse Orbital area lateral ↔ Human OFC BA11/47 (Wallis 2012)
  pid 46: Mouse Prelimbic ↔ Human dlPFC BA9/46 (Carlén 2017; *contested*)

dlPFC homology is opt-in, see docs/04_anchor_packs.md for the
Preuss 1995 vs Carlén 2017 debate. The script fits with both anchors
together but logs them separately so a user can drop the dlPFC entry
if they reject the homology.

Outputs:
  - outputs/coupling/pi_fc_plus_SC_with_lateral_pfc.npy
  - outputs/logs/beauchamp_validation_lateral_pfc.json

Usage:
    PYTHONPATH=src python experiments/anchor_packs/lateral_pfc.py
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
from otter.data.anchor_packs import build_lateral_pfc_region_anchors    # noqa: E402
from otter.models import MultimodalFGW                                   # noqa: E402

ANN  = ROOT / "outputs" / "anndata"
COUP = ROOT / "outputs" / "coupling"
LOG  = ROOT / "outputs" / "logs"


def main():
    M, _ = load_cached("mouse", cache_dir=ANN)
    H, _ = load_cached("human", cache_dir=ANN)
    costs = np.load(ANN / "full_costs.npz")
    entries = build_lateral_pfc_region_anchors(M.var, H.var, atlas_root=ROOT)
    print("Lateral PFC anchors:")
    for e in entries:
        print(f"  pid={e.pair_id:>3d}  {e.label!r}  |m|={len(e.mouse_indices)}  "
              f"|h|={len(e.human_indices)}")

    model = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7,
                           epsilon=5e-3, xyz_weight=0.5, lam_anchor=1.0)
    model.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"],
               region_anchors=entries)
    out = COUP / "pi_fc_plus_SC_with_lateral_pfc.npy"
    np.save(out, model.pi)
    print(f"\nSaved {out}")

    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(
        ["python", str(ROOT / "pipeline" / "05f_beauchamp_validation.py"),
         "--pi-file", out.name],
        env=env, check=True, capture_output=True,
    )
    src = LOG / "beauchamp_validation.json"
    dst = LOG / "beauchamp_validation_lateral_pfc.json"
    if src != dst:
        src.rename(dst)
    print(f"Saved {dst}")

    # No Beauchamp pair for OFC or dlPFC, confirm zero off-target side effect
    d_pfc = json.loads(dst.read_text())
    subprocess.run(["python", str(ROOT / "pipeline" / "05f_beauchamp_validation.py"),
                     "--pi-file", "pi_fc_plus_SC.npy"], env=env, check=True, capture_output=True)
    d_prod = json.loads((LOG / "beauchamp_validation.json").read_text())

    print("\nOff-target check (no direct gain expected, no OFC/dlPFC in Beauchamp):")
    for short, name in [
        ('Mot', 'Primary motor area -> precentral gyrus'),
        ('ACG', 'Anterior cingulate area -> cingulate gyrus'),
        ('S1',  'Primary somatosensory area -> postcentral gyrus'),
        ('Thal','Thalamus -> thalamus'),
    ]:
        p = d_prod.get(name, {}).get('top1'); a = d_pfc.get(name, {}).get('top1')
        if p is None or a is None: continue
        print(f"  {short}: {p*100:.0f}% → {a*100:.0f}%  ({(a-p)*100:+.0f}pp)")


if __name__ == "__main__":
    main()
