"""Add periaqueductal gray anchor (Ezra 2015; Kingsbury 2011).

Single-entry pack. Mouse PAG (16 parcels) ↔ Human PAG at MNI(±5, –30,
–10) r=6 mm (4 parcels).

Beauchamp has no PAG validation pair, so the direct effect is not
measurable. NAc gains +4 pp top-1, possibly via mass redistribution along
midbrain → forebrain projections; all other Beauchamp pairs are unchanged.

Usage:
    PYTHONPATH=src python experiments/anchor_packs/pag.py
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
from otter.data.anchor_packs import build_pag_region_anchors        # noqa: E402
from otter.models import MultimodalFGW                               # noqa: E402

ANN = ROOT / "outputs" / "anndata"
COUP = ROOT / "outputs" / "coupling"
LOG = ROOT / "outputs" / "logs"


def main():
    M, _ = load_cached("mouse", cache_dir=ANN)
    H, _ = load_cached("human", cache_dir=ANN)
    costs = np.load(ANN / "full_costs.npz")
    entries = build_pag_region_anchors(M.var, H.var, atlas_root=ROOT)
    for e in entries:
        print(f"  pid={e.pair_id}  {e.label!r}  |m|={len(e.mouse_indices)}  "
              f"|h|={len(e.human_indices)}")

    model = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7,
                          epsilon=5e-3, xyz_weight=0.5, lam_anchor=1.0)
    model.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"],
              region_anchors=entries)
    out = COUP / "pi_fc_plus_SC_with_pag.npy"
    np.save(out, model.pi)

    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(["python", str(ROOT / "pipeline" / "05f_beauchamp_validation.py"),
                    "--pi-file", out.name], env=env, check=True, capture_output=True)
    src = LOG / "beauchamp_validation.json"
    dst = LOG / "beauchamp_validation_pag.json"
    if src != dst:
        src.rename(dst)
    d_pag = json.loads(dst.read_text())
    subprocess.run(["python", str(ROOT / "pipeline" / "05f_beauchamp_validation.py"),
                    "--pi-file", "pi_fc_plus_SC.npy"], env=env, check=True, capture_output=True)
    d_prod = json.loads((LOG / "beauchamp_validation.json").read_text())

    for short, name in [
        ('SC',  'Superior colliculus, sensory related -> superior colliculus'),
        ('IC',  'Inferior colliculus -> inferior colliculus'),
        ('NAc', 'Striatum ventral region -> nucleus accumbens'),
        ('Thal','Thalamus -> thalamus'),
        ('Hyp', 'Hypothalamus -> hypothalamus'),
    ]:
        p = d_prod.get(name, {}).get('top1'); g = d_pag.get(name, {}).get('top1')
        if p is None or g is None: continue
        print(f"  {short}: {p*100:.0f}% → {g*100:.0f}% ({(g-p)*100:+.0f}pp)")


if __name__ == "__main__":
    main()
