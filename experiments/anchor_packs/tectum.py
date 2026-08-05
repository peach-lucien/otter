"""TECTUM-1: Add Superior + Inferior Colliculus sub-region anchors.

The tectum (midbrain colliculi) is one of OTTER's documented failure
regions, both SC and IC have 0 % Beauchamp top-1 under the production
point-anchor π. ``docs/archive/diagnostics.md`` calls out tectum's spatial
inversion as the failure mechanism: mouse SC is dorsal whereas human SC
is ventral in MNI space, so the xyz cross-species cost actively misleads
non-anchor tectum parcels.

This experiment adds the two tectum-pack region anchors:

  pid 32: Mouse Superior Colliculus ↔ Human SC (Mai/Paxinos, May 2006)
  pid 33: Mouse Inferior Colliculus ↔ Human IC (Mai/Paxinos, Schreiner 2007)

Tests three configurations on Beauchamp top-1 (with the same caveat
as BICCN motor, see docs/archive/iteration_log.md §5.12.2):

  1. Production fc_plus_SC point-anchor π        (baseline)
  2. Production + tectum pack                     (full)
  3. Production + SC anchor only (IC held out)    (held-out generalization)

Outputs:
  - outputs/coupling/pi_fc_plus_SC_with_tectum.npy
  - outputs/coupling/pi_fc_plus_SC_with_tectum_sc_only.npy
  - outputs/logs/beauchamp_validation_tectum.json
  - outputs/logs/beauchamp_validation_tectum_sc_only.json

Usage:
    PYTHONPATH=src python experiments/tectum/01_add_tectum_anchors.py
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
from otter.data.anchor_packs import build_tectum_region_anchors       # noqa: E402
from otter.models import MultimodalFGW                                # noqa: E402

ANN  = ROOT / "outputs" / "anndata"
COUP = ROOT / "outputs" / "coupling"
LOG  = ROOT / "outputs" / "logs"


def fit_and_validate(M, H, costs, *, region_anchors, pi_filename, val_filename):
    """Fit, save π, run Beauchamp validation, save under val_filename. Return dict."""
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

    entries = build_tectum_region_anchors(M.var, H.var, atlas_root=ROOT)
    print("\nTectum anchors:")
    for e in entries:
        print(f"  pid={e.pair_id:>3d}  {e.label!r}  |mouse|={len(e.mouse_indices)}  "
              f"|human|={len(e.human_indices)}")

    # Config 1: Full tectum pack
    print("\n[1/3] Fitting production + tectum pack (SC + IC) ...")
    d_full = fit_and_validate(
        M, H, costs, region_anchors=entries,
        pi_filename="pi_fc_plus_SC_with_tectum.npy",
        val_filename="beauchamp_validation_tectum.json",
    )

    # Config 2: SC only, hold IC out for generalization test
    print("\n[2/3] Fitting production + SC anchor only (IC held out) ...")
    d_sc_only = fit_and_validate(
        M, H, costs, region_anchors=[entries[0]],   # SC entry only
        pi_filename="pi_fc_plus_SC_with_tectum_sc_only.npy",
        val_filename="beauchamp_validation_tectum_sc_only.json",
    )

    # Config 3: Production baseline (re-run for consistency)
    print("\n[3/3] Re-running Beauchamp on production π for comparison ...")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(["python", str(ROOT / "pipeline" / "05f_beauchamp_validation.py"),
                     "--pi-file", "pi_fc_plus_SC.npy"], env=env, check=True, capture_output=True)
    d_prod = json.loads((LOG / "beauchamp_validation.json").read_text())

    key_pairs = [
        ('IC',    'Inferior colliculus -> inferior colliculus'),
        ('SC',    'Superior colliculus, sensory related -> superior colliculus'),
        ('Mot',   'Primary motor area -> precentral gyrus'),
        ('Aud',   "Primary auditory area -> Heschl's gyrus"),
        ('Thal',  'Thalamus -> thalamus'),
        ('Hyp',   'Hypothalamus -> hypothalamus'),
        ('Pon',   'Pons -> pons'),
        ('ACG',   'Anterior cingulate area -> cingulate gyrus'),
        ('V1',    'Visual areas -> cuneus'),
        ('S1',    'Primary somatosensory area -> postcentral gyrus'),
    ]

    print("\n" + "=" * 90)
    print(f"{'region':6s} {'prod':>7s} {'+tectum':>10s} {'+SC only':>10s} "
          f"{'Δ full':>8s} {'Δ SC-only':>10s}  notes")
    print("-" * 90)
    for short, name in key_pairs:
        p = d_prod.get(name, {}).get('top1')
        f = d_full.get(name, {}).get('top1')
        sc = d_sc_only.get(name, {}).get('top1')
        if None in (p, f, sc): continue
        d_full_pp = (f - p) * 100
        d_sc_pp = (sc - p) * 100
        marker = "  ←tectum" if short in ('SC', 'IC') else ""
        flag_full = "  ↑" if d_full_pp > 1 else ("  ↓" if d_full_pp < -1 else "")
        flag_sc = "  ↑" if d_sc_pp > 1 else ("  ↓" if d_sc_pp < -1 else "")
        print(f"  {short:4s} {p*100:>6.0f}%  {f*100:>8.0f}%  {sc*100:>8.0f}%  "
              f"{d_full_pp:>+6.0f}pp  {d_sc_pp:>+8.0f}pp{marker}{flag_full}")

    print(f"\n{'-' * 90}")
    print("Reading: Δ full = production → +tectum (both anchors).")
    print("         Δ SC-only = production → +SC anchor only (IC held out).")
    print("Generalization signal: 'SC-only' delta for IC.")


if __name__ == "__main__":
    main()
