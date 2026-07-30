#!/usr/bin/env python3
"""Beat 3 gate: do the two depression TMS circuits reverse-translate to DIFFERENT mouse
substrates?  Preregistered, a priori:
  * dysphoric circuit (negative side of the Siddiqi anxdys atlas)  -> mouse medial prefrontal
    (PL, ILA, ACAd, ACAv, ORBm, FRP)      [human dysphoria <-> subgenual/medial PFC]
  * anxiosomatic circuit (positive side)                          -> mouse amygdala + insula
    (BLA, BMA, CEA, LA, AId, AIv, AIp)    [human somatic anxiety <-> amygdala/insula]

It is NOT enough that each circuit is individually spin-significant; the claim is a DISSOCIATION.
Test statistic: prefrontal-minus-amygdala "bias" of each routed circuit, and the between-circuit
contrast  C = bias(dysphoric) - bias(anxiosomatic)  (expected > 0). Significance from a spatial
spin null that rotates the human map, re-splits by sign, re-routes and re-computes C -- so the
null preserves each circuit's spatial smoothness and the sign split.

Needs clinical_maps/tms_anxdys.nii.gz. Run:
  cd homer && PYTHONPATH=src python experiments/reverse_translation/07_symptom_dissociation.py
Read-only; writes outputs/logs/reverse_translation_symptom_dissociation.json
"""
from __future__ import annotations
import json, sys, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from homer.data import load_cached, load_pi, pi_provenance      # noqa: E402

_s = importlib.util.spec_from_file_location("rt01", ROOT / "experiments/reverse_translation/01_validate.py")
rt01 = importlib.util.module_from_spec(_s); _s.loader.exec_module(rt01)

ATLAS = ROOT / "experiments/reverse_translation/clinical_maps/tms_anxdys.nii.gz"
FRAC = 0.10
N_SPINS = 2000

DYS_SET = ["PL", "ILA", "ACAd", "ACAv", "ORBm", "FRP"]                 # medial prefrontal
ANX_SET = ["BLA", "BMA", "CEA", "LA", "AId", "AIv", "AIp"]            # amygdala + insula


def focal(v_raw, sign, frac=FRAC):
    v = sign * np.asarray(v_raw, float)
    ok = np.isfinite(v); v = np.where(ok, v, np.nanmin(v[ok]))
    thr = np.quantile(v, 1.0 - frac)
    return np.clip(v - thr, 0.0, None)


def struct_dist(v_m, acr, structs):
    s = np.array([max(np.nanmean(v_m[acr == a]), 0.0) for a in structs])
    return s / s.sum() if s.sum() > 0 else s


def main():
    prov = pi_provenance(); print(f"pi: {prov['pi_file']} sha256={prov['pi_sha256'][:12]}")
    if not ATLAS.exists():
        print("missing", ATLAS); return
    pi = load_pi(); acr = rt01.mouse_acr()
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    hxyz = H.var[["x", "y", "z"]].to_numpy(float)
    parc, numids = rt01.build_node_voxel_map(H.var)
    rown = pi / pi.sum(1, keepdims=True).clip(1e-12)
    structs = [s for s in set(acr.tolist()) if s != "NA" and (acr == s).sum() >= 5]
    dys_present = [s for s in DYS_SET if s in structs]
    anx_present = [s for s in ANX_SET if s in structs]
    axis = np.array([(1.0 if s in dys_present else (-1.0 if s in anx_present else 0.0)) for s in structs])

    v_raw = rt01.sample_to_nodes(ATLAS, parc, numids)

    def contrast_from(vr):
        s_dys = struct_dist(rown @ focal(vr, -1.0), acr, structs)   # dysphoric = negative side
        s_anx = struct_dist(rown @ focal(vr, +1.0), acr, structs)   # anxiosomatic = positive side
        return float(axis @ s_dys - axis @ s_anx), s_dys, s_anx

    obs_C, s_dys, s_anx = contrast_from(v_raw)

    spins = rt01.spin_indices(hxyz, N_SPINS)
    null = np.array([contrast_from(v_raw[perm])[0] for perm in spins])
    p = float((np.sum(null >= obs_C) + 1) / (N_SPINS + 1))

    def top(sd, k=5):
        order = np.argsort(sd)[::-1]
        return [(structs[i], round(float(sd[i]), 4)) for i in order[:k]]

    bias_dys = float(axis @ s_dys); bias_anx = float(axis @ s_anx)
    out = {**prov, "frac": FRAC, "n_spins": N_SPINS, "contrast_C": obs_C, "spin_p": p,
           "bias_dysphoric": bias_dys, "bias_anxiosomatic": bias_anx,
           "dys_set": dys_present, "anx_set": anx_present,
           "top_dysphoric": top(s_dys), "top_anxiosomatic": top(s_anx)}
    (ROOT / "outputs/logs/reverse_translation_symptom_dissociation.json").write_text(json.dumps(out, indent=1))

    print("\n==============  SYMPTOM DISSOCIATION (beat 3 gate)  ==============")
    print(f"prefrontal-minus-amygdala bias:  dysphoric {bias_dys:+.3f}   anxiosomatic {bias_anx:+.3f}")
    print(f"between-circuit contrast C = {obs_C:+.4f}   spin p = {p:.4f}   (C>0 & p<0.05 = dissociation)")
    print("  dysphoric top:    " + ", ".join(f"{s}" for s, _ in out["top_dysphoric"]))
    print("  anxiosomatic top: " + ", ".join(f"{s}" for s, _ in out["top_anxiosomatic"]))
    verdict = ("DISSOCIATION CONFIRMED" if obs_C > 0 and p < 0.05 else
               "NO SIGNIFICANT DISSOCIATION (drop beat 3)")
    print("VERDICT:", verdict)
    print("wrote outputs/logs/reverse_translation_symptom_dissociation.json")


if __name__ == "__main__":
    main()
