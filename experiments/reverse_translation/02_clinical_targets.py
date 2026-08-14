#!/usr/bin/env python3
"""Clinical-target reverse translation — which human therapeutic targets can a mouse model?

Routes established human neuromodulation targets (DBS / TMS sites) through the coupling to
mouse, and reports not just WHERE each lands but HOW CONFIDENTLY -- so the output is a
prescription with a trust flag: "target the mouse X" or "no adequate mouse homolog, do not
model this in a mouse". The expected contrast:

  conserved subcortical/limbic targets (nucleus accumbens, subthalamic nucleus) -> a sharp,
      significant mouse prescription;
  a contested medial-frontal target (subgenual Cg25) -> mouse medial PFC (IL/PL), the debated
      but existing homolog;
  a primate-elaborated target (dorsolateral PFC, the antidepressant TMS site) -> diffuse, low
      confidence, no adequate mouse home -- consistent with the dlPFC connectional
      reorganisation reported in `docs/03_results.md`.

The human target is defined as the K human parcels nearest the published MNI coordinate; the
reverse operator is row-normalised pi (human -> mouse), aggregated to mouse structures.

Coordinates below are REPRESENTATIVE published targets; confirm against your preferred source
before publication. The method contrast is robust to a few mm.

Run: cd otter && PYTHONPATH=src python experiments/reverse_translation/02_clinical_targets.py
Read-only; writes outputs/logs/reverse_translation_clinical.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached, load_pi, pi_provenance      # noqa: E402

K_NEAREST = 8          # human parcels forming the target ROI
MIN_PARCELS = 2        # allow small nuclei (STN) — parcel counts are printed
N_SPINS = 1000

# name: (MNI xyz, expected mouse structures, class, source)
CLINICAL_TARGETS = {
    "NAcc DBS (OCD/depression)":   ((-8, 10, -8),  ["ACB"],            "conserved",  "ventral striatum DBS"),
    "STN DBS (Parkinson's)":       ((-12, -14, -7),["STN"],            "conserved",  "subthalamic DBS"),
    "Subgenual Cg25 DBS (depr.)":  ((-4, 22, -8),  ["ILA", "PL", "ACAv"], "contested", "Mayberg subgenual DBS"),
    "dlPFC TMS (depression)":      ((-42, 44, 30), None,               "elaborated", "Fox antidepressant TMS target"),
}


def mouse_acr():
    mm = json.loads((ROOT / "data_external/mouse_sc_meta.json").read_text())
    return np.array([mm["structure_acronyms"][i] if i >= 0 else "NA" for i in mm["node_struct_idx"]])


def evidence_tier_by_parcel():
    try:
        z = np.load(ROOT / "outputs/coupling/trust_multisource_canonical.npz", allow_pickle=True)
        for key in ("evidence_tier", "tier"):
            if key in z:
                return np.array([str(t) for t in z[key]])
    except Exception:
        pass
    return None


def spin_indices(xyz, n, seed=0):
    from scipy.spatial import cKDTree
    from otter.eval.nulls import _haar_rotation
    c = xyz - xyz.mean(0); s = c / np.linalg.norm(c, axis=1, keepdims=True)
    t = cKDTree(s); rng = np.random.default_rng(seed)
    return [t.query(s @ _haar_rotation(rng).T)[1] for _ in range(n)]


def main():
    prov = pi_provenance(); print(f"pi: {prov['pi_file']} sha256={prov['pi_sha256'][:12]}")
    pi = load_pi()
    acr = mouse_acr()
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    hxyz = H.var[["x", "y", "z"]].to_numpy(float)
    rown = pi / pi.sum(1, keepdims=True).clip(1e-12)
    tiers = evidence_tier_by_parcel()
    spins = spin_indices(hxyz, N_SPINS)

    structs = [s for s in set(acr.tolist()) if s != "NA" and (acr == s).sum() >= MIN_PARCELS]

    def route(v_h):
        v_m = rown @ v_h
        return {s: float(np.nanmean(v_m[acr == s])) for s in structs}, v_m

    rows = []
    for name, (xyz, expected, klass, src) in CLINICAL_TARGETS.items():
        d = np.linalg.norm(hxyz - np.array(xyz), axis=1)
        seed = np.argsort(d)[:K_NEAREST]
        v_h = np.zeros(pi.shape[1]); v_h[seed] = 1.0
        sc, v_m = route(v_h)
        ranked = sorted(sc, key=sc.get, reverse=True)
        vals = np.array([sc[s] for s in ranked])
        peak = float(vals[0] / vals.sum()) if vals.sum() > 0 else 0.0     # concentration on top structure
        p = vals / vals.sum(); effN = float(1.0 / np.sum(p ** 2))         # diffuseness over structures
        # spin null on the top structure's score
        top = ranked[0]
        null = []
        for perm in spins:
            vm = rown @ v_h[perm]
            null.append(float(np.nanmean(vm[acr == top])))
        spin_p = float((np.sum(np.array(null) >= sc[top]) + 1) / (N_SPINS + 1))
        # evidence tier at the top mouse structure (modal)
        tier = None
        if tiers is not None:
            tt = tiers[acr == top]
            if len(tt):
                u, c = np.unique(tt, return_counts=True); tier = str(u[c.argmax()])
        hit = None if expected is None else any(e in ranked[:3] for e in expected)
        # verdict
        if spin_p < 0.05 and peak >= 0.12 and effN <= 8:
            verdict = "CONFIDENT PRESCRIPTION"
        elif spin_p >= 0.05 or effN > 15:
            verdict = "NO ADEQUATE MOUSE TARGET"
        else:
            verdict = "WEAK / PARTIAL"
        rows.append(dict(target=name, klass=klass, source=src, mni=list(xyz),
                         expected=expected, hit_top3=hit,
                         top3=[(s, round(sc[s], 4)) for s in ranked[:3]],
                         peak_concentration=round(peak, 3), effN_structures=round(effN, 1),
                         spin_p=spin_p, evidence_tier_at_top=tier, verdict=verdict))

    out = {**prov, "params": dict(K_NEAREST=K_NEAREST, N_SPINS=N_SPINS), "targets": rows}
    (ROOT / "outputs/logs/reverse_translation_clinical.json").write_text(json.dumps(out, indent=1))

    print("\n=================  CLINICAL-TARGET REVERSE TRANSLATION  =================")
    for r in rows:
        exp = "primate-elaborated (expect none)" if r["expected"] is None else "/".join(r["expected"])
        print(f"\n{r['target']}   [{r['klass']}]  expect: {exp}")
        print("   top mouse -> " + ", ".join(f"{s} {v:.3f}" for s, v in r["top3"]))
        print(f"   concentration={r['peak_concentration']}  effN={r['effN_structures']}  "
              f"spin p={r['spin_p']:.3f}  tier@top={r['evidence_tier_at_top']}")
        print(f"   VERDICT: {r['verdict']}"
              + ("" if r["hit_top3"] is None else f"   (expected structure in top-3: {r['hit_top3']})"))
    print("\nwrote outputs/logs/reverse_translation_clinical.json")


if __name__ == "__main__":
    main()
