#!/usr/bin/env python3
"""Clinical-network reverse translation: which human therapeutic networks a mouse can model.

Instead of point seeds, which missed small subcortical/ventromedial targets on the coarse
human parcellation, this routes the published whole-brain DBS/TMS optimal-connectivity
network maps, distributed maps derived from a ~1,000-subject normative connectome. That is
the same kind of distributed input used in the functional reverse-translation (12/12
spin-significant).

For each therapeutic network it reports where it lands in mouse and how confidently, and
flags "no adequate mouse target" when a human network does not route to any specific mouse
structure.

HUMAN NETWORK MAPS (put MNI152 volumes in clinical_maps/, named <stem>.nii.gz)
  depression_tms.nii.gz   convergent depression TMS circuit  Siddiqi 2021 NHB (n=713)
  tms_anxdys.nii.gz       dysphoric/anxiosomatic TMS atlas   Siddiqi 2020 AJP  (n=111)
  ptsd_circuit.nii.gz     PTSD circuit                       Siddiqi 2024 NatNeuro (n=193)
  ms_depression.nii.gz    MS-depression circuit              Siddiqi 2023 NMH  (n=281)
These are directly downloadable from NeuroVault collection 13075; see
clinical_maps/DATA_SOURCES.md for the exact per-file URLs and a curl block.
Any MNI152 volume works, since the script resamples it, so DBS voxel maps can be added.

Run: cd otter && PYTHONPATH=src python experiments/reverse_translation/03_clinical_networks.py
Read-only; writes outputs/logs/reverse_translation_clinical_networks.json
"""
from __future__ import annotations
import json, sys, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached, load_pi, pi_provenance      # noqa: E402

# reuse the validated helpers from script 01 (resampling, mouse acronyms, spin null)
_s = importlib.util.spec_from_file_location("rt01", ROOT / "experiments/reverse_translation/01_validate.py")
rt01 = importlib.util.module_from_spec(_s); _s.loader.exec_module(rt01)

MAPDIR = ROOT / "experiments/reverse_translation/clinical_maps"
N_SPINS = 1000
MIN_PARCELS = 5

# stem -> (class, note). Expected structures are not hard-coded; the script reads out where
# each map lands and how confidently rather than grading against a preset answer.
NETWORKS = {
    "depression_tms": ("cortico-limbic", "convergent depression TMS circuit (Siddiqi 2021 NHB)"),
    "tms_anxdys":     ("cortico-limbic", "dysphoric/anxiosomatic TMS atlas (Siddiqi 2020 AJP)"),
    "ptsd_circuit":   ("cortico-limbic", "PTSD circuit (Siddiqi 2024 Nat Neurosci)"),
    "ms_depression":  ("cortico-limbic", "MS-depression circuit (Siddiqi 2023 NMH)"),
    # add DBS voxel maps here once Lead-DBS fiber targets are converted to MNI volumes:
    # "ocd_dbs": ("conserved-subcortical", "OCD DBS optimal network (Li/Baldermann)"),
}


def main():
    prov = pi_provenance(); print(f"pi: {prov['pi_file']} sha256={prov['pi_sha256'][:12]}")
    if not MAPDIR.exists() or not any(MAPDIR.glob("*.nii*")):
        print(f"\nNO NETWORK MAPS in {MAPDIR}. See clinical_maps/DATA_SOURCES.md."); return
    pi = load_pi()
    acr = rt01.mouse_acr() if hasattr(rt01, "mouse_acr") else _acr()
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    hxyz = H.var[["x", "y", "z"]].to_numpy(float)
    parc, numids = rt01.build_node_voxel_map(H.var)
    rown = pi / pi.sum(1, keepdims=True).clip(1e-12)
    spins = rt01.spin_indices(hxyz, N_SPINS)
    structs = [s for s in set(acr.tolist()) if s != "NA" and (acr == s).sum() >= MIN_PARCELS]

    rows = []
    for stem, (klass, note) in NETWORKS.items():
        cand = list(MAPDIR.glob(f"{stem}.nii*"))
        if not cand:
            rows.append({"network": stem, "status": "map_missing"}); continue
        v_h = rt01.sample_to_nodes(cand[0], parc, numids)
        ok = np.isfinite(v_h); v_h = np.where(ok, v_h, np.nanmean(v_h[ok]))
        v_m = rown @ v_h
        sc = {s: float(np.nanmean(v_m[acr == s])) for s in structs}
        ranked = sorted(sc, key=sc.get, reverse=True)
        vals = np.array([sc[s] for s in ranked]); vals = vals - vals.min()
        peak = float(vals[0] / vals.sum()) if vals.sum() > 0 else 0.0
        p = (vals / vals.sum()) if vals.sum() > 0 else np.ones(len(vals)) / len(vals)
        effN = float(1.0 / np.sum(p ** 2))
        top = ranked[0]
        null = np.array([float(np.nanmean((rown @ v_h[perm])[acr == top])) for perm in spins])
        spin_p = float((np.sum(null >= sc[top]) + 1) / (N_SPINS + 1))
        # verdict: whether it routes to a specific mouse structure at all
        verdict = "MOUSE HOMOLOG (routes specifically)" if spin_p < 0.05 else \
                  "NO ADEQUATE MOUSE TARGET (does not route specifically)"
        rows.append({"network": stem, "klass": klass, "note": note,
                     "top3": [(s, round(sc[s], 4)) for s in ranked[:3]],
                     "concentration": round(peak, 3), "effN": round(effN, 1),
                     "spin_p": spin_p, "verdict": verdict})

    out = {**prov, "n_spins": N_SPINS, "networks": rows}
    (ROOT / "outputs/logs/reverse_translation_clinical_networks.json").write_text(json.dumps(out, indent=1))
    print("\n=============  CLINICAL-NETWORK REVERSE TRANSLATION  =============")
    for r in rows:
        if r.get("status") == "map_missing":
            print(f"  {r['network']:16s} MAP MISSING"); continue
        print(f"\n{r['network']}  [{r['klass']}]  {r['note']}")
        print("   top mouse -> " + ", ".join(f"{s} {v:.3f}" for s, v in r["top3"]))
        print(f"   concentration={r['concentration']}  effN={r['effN']}  spin p={r['spin_p']:.3f}")
        print(f"   VERDICT: {r['verdict']}")
    print("\nwrote outputs/logs/reverse_translation_clinical_networks.json")


def _acr():
    mm = json.loads((ROOT / "data_external/mouse_sc_meta.json").read_text())
    return np.array([mm["structure_acronyms"][i] if i >= 0 else "NA" for i in mm["node_struct_idx"]])


if __name__ == "__main__":
    main()
