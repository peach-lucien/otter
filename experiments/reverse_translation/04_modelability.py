#!/usr/bin/env python3
"""Mouse-modelability contrast: do conserved functional systems reverse-translate to a
FOCAL mouse home, while human clinical neuromodulation circuits SMEAR across the mouse brain?

This turns the reverse-translation coupling into a quantitative "when is a mouse model
adequate" axis. For every human map we route it to mouse and measure how CONCENTRATED the
mouse prediction is (effective number of structures, effN = 1/sum p^2; and top-structure
concentration). Low effN = a sharp mouse home (modelable); high effN = distributed, no
adequate single mouse substrate (primate-elaborated).

THE CONFOUND, AND THE CONTROL
-----------------------------
Clinical maps are smooth whole-brain connectivity t-maps; Neurosynth functional maps are
focal. A raw effN gap could just inherit that input-smoothness difference. So we EQUALISE
input focality: every map is thresholded to the same top-FRAC of human parcels before
routing. If clinical circuits still smear in mouse after inputs are made equally focal, the
diffuseness is a property of the cross-species mapping, not the input. We report several FRAC
values for robustness, plus the raw (unmatched) numbers for completeness.

Groups:
  conserved-functional : the 12 GROUND_TRUTH Neurosynth maps in human_maps/ (script 01)
  clinical             : depression_tms, ptsd_circuit, ms_depression in clinical_maps/,
                         plus the anxdys atlas SPLIT by sign into anxiosomatic(+)/dysphoric(-)

Outputs a per-map table, a group contrast (median effN + label-permutation p), and the
anxiosomatic-vs-dysphoric dissociation.

Run: cd homer && PYTHONPATH=src python experiments/reverse_translation/04_modelability.py
Read-only; writes outputs/logs/reverse_translation_modelability.json
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

HUMAN = ROOT / "experiments/reverse_translation/human_maps"
CLIN = ROOT / "experiments/reverse_translation/clinical_maps"
FRACS = [0.05, 0.10, 0.20]        # matched input focality levels
PRIMARY = 0.10
MIN_PARCELS = 5
N_SPINS = 500


def focal_input(v_raw, frac, sign=1.0):
    """Rectify to the top-`frac` human parcels (sign +1 or -1), weighted above threshold.
    Guarantees a non-negative input of matched sparsity across all maps."""
    v = sign * np.asarray(v_raw, float)
    ok = np.isfinite(v)
    v = np.where(ok, v, np.nanmin(v[ok]))
    thr = np.quantile(v, 1.0 - frac)
    return np.clip(v - thr, 0.0, None)


def diffuseness(v_m, acr, structs):
    """effN (=1/sum p^2) and top-structure concentration of a non-negative mouse map."""
    s = np.array([max(np.nanmean(v_m[acr == a]), 0.0) for a in structs])
    tot = s.sum()
    if tot <= 0:
        return len(structs), 0.0, [(structs[i], 0.0) for i in range(min(3, len(structs)))]
    p = s / tot
    effN = float(1.0 / np.sum(p ** 2))
    order = np.argsort(s)[::-1]
    conc = float(s[order[0]] / tot)
    top3 = [(structs[i], round(float(s[i]), 4)) for i in order[:3]]
    return effN, conc, top3


def main():
    prov = pi_provenance(); print(f"pi: {prov['pi_file']} sha256={prov['pi_sha256'][:12]}")
    pi = load_pi()
    acr = rt01.mouse_acr()
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    hxyz = H.var[["x", "y", "z"]].to_numpy(float)
    parc, numids = rt01.build_node_voxel_map(H.var)
    rown = pi / pi.sum(1, keepdims=True).clip(1e-12)
    structs = [s for s in set(acr.tolist()) if s != "NA" and (acr == s).sum() >= MIN_PARCELS]
    spins = rt01.spin_indices(hxyz, N_SPINS)

    # assemble the map list: (name, group, path, sign, ground_truth_or_None)
    items = []
    for fn, (term, gt) in rt01.GROUND_TRUTH.items():
        cand = list(HUMAN.glob(f"{term}.nii*")) + list(HUMAN.glob(f"{fn}.nii*"))
        if cand:
            items.append((fn, "functional", cand[0], 1.0, gt))
    for stem in ["depression_tms", "ptsd_circuit", "ms_depression"]:
        cand = list(CLIN.glob(f"{stem}.nii*"))
        if cand:
            items.append((stem, "clinical", cand[0], 1.0, None))
    anx = list(CLIN.glob("tms_anxdys.nii*"))
    if anx:
        items.append(("anxiosomatic(+)", "clinical", anx[0], 1.0, None))
        items.append(("dysphoric(-)",    "clinical", anx[0], -1.0, None))

    # precompute sampled human maps once
    sampled = {}
    for name, grp, path, sign, gt in items:
        key = str(path)
        if key not in sampled:
            sampled[key] = rt01.sample_to_nodes(path, parc, numids)

    rows = []
    for name, grp, path, sign, gt in items:
        v_raw = sampled[str(path)]
        rec = {"name": name, "group": grp, "ground_truth": gt, "by_frac": {}}
        for frac in FRACS:
            w = focal_input(v_raw, frac, sign)
            v_m = rown @ w
            effN, conc, top3 = diffuseness(v_m, acr, structs)
            entry = {"effN": round(effN, 1), "concentration": round(conc, 3), "top3": top3}
            if frac == PRIMARY:
                # spin null on the top structure + GT rank (functional only)
                top = top3[0][0]
                obs = np.nanmean(v_m[acr == top])
                null = np.array([np.nanmean((rown @ w[perm])[acr == top]) for perm in spins])
                entry["spin_p"] = float((np.sum(null >= obs) + 1) / (N_SPINS + 1))
                if gt:
                    ranked = [s for s, _ in sorted(
                        ((a, np.nanmean(v_m[acr == a])) for a in structs),
                        key=lambda kv: kv[1], reverse=True)]
                    present = [g for g in gt if g in ranked]
                    entry["gt_rank"] = None if not present else min(ranked.index(g) for g in present) + 1
                    entry["gt_hit_top3"] = bool(present and entry["gt_rank"] <= 3)
            rec["by_frac"][f"{frac}"] = entry
        rows.append(rec)

    # group contrast at PRIMARY frac
    fx = lambda r: r["by_frac"][f"{PRIMARY}"]["effN"]
    fun = [fx(r) for r in rows if r["group"] == "functional"]
    cli = [fx(r) for r in rows if r["group"] == "clinical"]
    # label-permutation test on median-effN difference
    obs_d = float(np.median(cli) - np.median(fun))
    allv = np.array(fun + cli); nfun = len(fun)
    rng = np.random.default_rng(0)
    perm = []
    for _ in range(10000):
        rng.shuffle(allv)
        perm.append(np.median(allv[nfun:]) - np.median(allv[:nfun]))
    perm = np.array(perm)
    p_perm = float((np.sum(perm >= obs_d) + 1) / (len(perm) + 1))

    summary = {"frac_primary": PRIMARY,
               "functional_effN": {"median": float(np.median(fun)), "vals": sorted(fun)},
               "clinical_effN": {"median": float(np.median(cli)), "vals": sorted(cli)},
               "median_gap": round(obs_d, 1), "perm_p_clinical_gt_functional": p_perm,
               "n_functional": len(fun), "n_clinical": len(cli)}

    out = {**prov, "fracs": FRACS, "n_spins": N_SPINS, "summary": summary, "maps": rows}
    (ROOT / "outputs/logs/reverse_translation_modelability.json").write_text(json.dumps(out, indent=1))

    print("\n==================  MOUSE-MODELABILITY CONTRAST  ==================")
    print(f"(input focality matched at top-{int(PRIMARY*100)}% human parcels)\n")
    print(f"{'map':16s} {'group':11s} {'effN':>6s} {'conc':>6s} {'spin_p':>7s}  top mouse / GT")
    for r in sorted(rows, key=lambda r: (r["group"], fx(r))):
        e = r["by_frac"][f"{PRIMARY}"]
        gt = ""
        if r["ground_truth"]:
            gt = f"  GT rank {e.get('gt_rank')}" + (" HIT" if e.get("gt_hit_top3") else "")
        tops = ", ".join(s for s, _ in e["top3"])
        print(f"{r['name']:16s} {r['group']:11s} {e['effN']:6.1f} {e['concentration']:6.3f} "
              f"{e.get('spin_p', float('nan')):7.3f}  {tops}{gt}")
    s = summary
    print(f"\nfunctional effN median {s['functional_effN']['median']:.1f}  "
          f"vs clinical {s['clinical_effN']['median']:.1f}  "
          f"(gap {s['median_gap']}, perm p={s['perm_p_clinical_gt_functional']:.4f})")
    print("robustness across fracs (median effN functional | clinical):")
    for frac in FRACS:
        ff = [r["by_frac"][f"{frac}"]["effN"] for r in rows if r["group"] == "functional"]
        cc = [r["by_frac"][f"{frac}"]["effN"] for r in rows if r["group"] == "clinical"]
        print(f"  top-{int(frac*100):2d}%: {np.median(ff):5.1f} | {np.median(cc):5.1f}")
    print("\nSYMPTOM SPLIT (anxdys atlas):")
    for r in rows:
        if r["name"] in ("anxiosomatic(+)", "dysphoric(-)"):
            e = r["by_frac"][f"{PRIMARY}"]
            print(f"  {r['name']:16s} effN {e['effN']:.1f}  ->  " + ", ".join(s for s, _ in e["top3"]))
    print("\nwrote outputs/logs/reverse_translation_modelability.json")


if __name__ == "__main__":
    main()
