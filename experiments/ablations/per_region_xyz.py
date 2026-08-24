"""Per-region xyz ablation report.

Tests whether the failure of motor/tectum/piriform under production
(spatial-topology inversion) can be fixed by zeroing the xyz cost for parcels
in those regions.

Two configurations are fit and compared against production:
  1. ``xyz=0 globally``, xyz cost removed entirely
  2. ``xyz=0 per-region``, xyz removed only for parcels nearest
                                     pair_ids {2 (Motor), 11 (Piriform),
                                     21 (Tectum)}

Result (saved to outputs/logs/beauchamp_validation_xyz_zero.json and
beauchamp_validation_per_region_xyz_v2.json):

  - **Global xyz=0** has *mixed* effects per Beauchamp pair:
      hurts Thalamus (-28pp), S1 (-17pp), Caudate (-12pp), Hyp (-8pp),
      NAc (-8pp), ACG (-4pp); helps Piriform (+13pp), Tectum/SC (+6pp),
      Motor (+4pp). Net: xyz overall *helps* more than it hurts.
  - **Per-region xyz=0** targeting {Motor, Piriform, Tectum} does *not*
      reproduce the global effect, those regions stay at 0% top-1.
      The xyz contribution interacts non-locally via the FGW equilibrium,
      so per-row weighting cannot replicate a global xyz change.

Per-region xyz weighting therefore does not fix topology-inverted regions.
The ``xyz_weight_per_mouse_parcel`` infrastructure is kept in the model API
as a general tool.

Usage:
    PYTHONPATH=src python experiments/ablations/per_region_xyz.py
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
from otter.data import load_cached, get_anchor_index, build_xyz_weight_array  # noqa: E402
from otter.models import MultimodalFGW                                          # noqa: E402

ANN  = ROOT / "outputs" / "anndata"
COUP = ROOT / "outputs" / "coupling"
LOG  = ROOT / "outputs" / "logs"
LOG.mkdir(parents=True, exist_ok=True)


# Topology-inversion candidates
XYZ_OFF_PAIR_IDS = {2, 11, 21}    # Motor, Piriform/Olfactory, Tectum
KEY_PAIRS = [
    ('Hyp',  'Hypothalamus -> hypothalamus'),
    ('ACG',  'Anterior cingulate area -> cingulate gyrus'),
    ('Aud',  "Primary auditory area -> Heschl's gyrus"),
    ('S1',   'Primary somatosensory area -> postcentral gyrus'),
    ('V1',   'Visual areas -> cuneus'),
    ('NAc',  'Striatum ventral region -> nucleus accumbens'),
    ('Cau',  'Caudoputamen -> caudate nucleus'),
    ('Pir',  'Piriform area -> piriform cortex'),
    ('Mot',  'Primary motor area -> precentral gyrus'),
    ('Thal', 'Thalamus -> thalamus'),
    ('IC',   'Inferior colliculus -> inferior colliculus'),
    ('SC',   'Superior colliculus, sensory related -> superior colliculus'),
    ('Pal',  'Pallidum -> globus pallidus'),
    ('Pon',  'Pons -> pons'),
    ('Sub',  'Subiculum -> subiculum'),
]


def beauchamp_topk(pi_filename: str, save_as: str) -> dict:
    """Run 05f_beauchamp_validation.py on a saved π, save under a unique name,
    return the dict of top-1 per pair. If save_as == 'beauchamp_validation.json'
    the file is left in place (canonical output)."""
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(
        ["python", str(ROOT / "pipeline" / "05f_beauchamp_validation.py"),
         "--pi-file", pi_filename],
        env=env, check=True, capture_output=True,
    )
    src = LOG / "beauchamp_validation.json"
    dst = LOG / save_as
    if src != dst:
        src.rename(dst)
    return json.loads(dst.read_text())


def main():
    M, _ = load_cached("mouse", cache_dir=ANN)
    H, _ = load_cached("human", cache_dir=ANN)
    costs = np.load(ANN / "full_costs.npz")
    print(f"Mouse parcels: {len(M.var)}, human parcels: {len(H.var)}")

    # ---- Config 1: xyz = 0 globally
    print("\n[1/2] Fitting fc_plus_SC with xyz_weight=0 globally ...")
    m_global = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7,
                              epsilon=5e-3, xyz_weight=0.0, lam_anchor=1.0)
    m_global.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"])
    f_global = COUP / "pi_fc_plus_SC_xyz_zero.npy"
    np.save(f_global, m_global.pi)
    d_global = beauchamp_topk(f_global.name, "beauchamp_validation_xyz_zero.json")

    # ---- Config 2: per-region xyz = 0
    print("\n[2/2] Fitting fc_plus_SC with xyz_weight=0 for pair_ids "
          f"{sorted(XYZ_OFF_PAIR_IDS)} ...")
    idx_m = get_anchor_index(M.var)
    xyz_w = build_xyz_weight_array(
        M.var, idx_m,
        weights_per_pair_id={p: 0.0 for p in XYZ_OFF_PAIR_IDS},
        default_weight=0.5,
    )
    print(f"  zeroed xyz for {int((xyz_w == 0.0).sum())} of {len(xyz_w)} parcels")
    m_perreg = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7,
                              epsilon=5e-3, xyz_weight=0.5, lam_anchor=1.0)
    m_perreg.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"],
                  xyz_weight_per_mouse_parcel=xyz_w)
    f_perreg = COUP / "pi_fc_plus_SC_per_region_xyz_v2.npy"
    np.save(f_perreg, m_perreg.pi)
    d_perreg = beauchamp_topk(f_perreg.name, "beauchamp_validation_per_region_xyz_v2.json")

    # ---- Point-anchor comparator.
    d_prod = beauchamp_topk("pi_fc_plus_SC.npy", "beauchamp_validation.json")

    # ---- Compare
    print(f"\n{'='*75}")
    print(f"{'region':6s} {'prod':>6s} {'xyz=0 global':>13s} {'per-region':>11s}  delta-pp  notes")
    print("-" * 75)
    for short, name in KEY_PAIRS:
        p = d_prod.get(name, {}).get('top1')
        g = d_global.get(name, {}).get('top1')
        r = d_perreg.get(name, {}).get('top1')
        if None in (p, g, r): continue
        gd, rd = (g - p) * 100, (r - p) * 100
        targ = "  ← targeted" if short.lower() in ('mot', 'sc', 'pir') else ""
        print(f"  {short:4s} {p*100:>5.0f}%  {g*100:>9.0f}%      {r*100:>5.0f}%     "
              f"g={gd:+4.0f}  r={rd:+4.0f}{targ}")

    print(f"\n{'-'*75}")
    print("Conclusion:")
    print("  - Global xyz=0 is mixed (net negative); xyz helps most regions overall.")
    print("  - Per-region xyz=0 targeting topology-inversion candidates does NOT")
    print("    reproduce the (modest) global gains on those regions. The xyz effect")
    print("    interacts non-locally via the FGW equilibrium.")
    print("  - Per-row xyz weighting does not reproduce the global xyz effect.")


if __name__ == "__main__":
    main()
