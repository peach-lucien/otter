#!/usr/bin/env python3
"""Subcortical-coverage disease-substrate reverse translation (the retry).

Neurosynth disease maps failed because they don't image the subcortex/midbrain, so the
disease substrate wasn't in the input (Parkinson landed in thalamus, SNc unsampled). This
uses the neuromaps molecular atlas (Hansen 2022) instead: PET neurotransmitter-system maps,
MNI152 volumetric, with real striatal/midbrain coverage.

KEY TEST (Parkinson): the human DOPAMINE system map (DAT / dopamine synthesis / D1/D2) is the
substrate that mouse PD models target (nigrostriatal). Does it reverse-translate to the mouse
dopamine system (CP, ACB, SNc, SNr, VTA)?  Serotonin and mu-opioid maps are carried as
reference systems for specificity.

Requires neuromaps:  pip install neuromaps
Run: cd homer && PYTHONPATH=src python experiments/reverse_translation/06_neuromaps_substrate.py
Read-only w.r.t. the coupling; caches maps in neuromaps_cache/, writes
outputs/logs/reverse_translation_neuromaps.json
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

CACHE = ROOT / "experiments/reverse_translation/neuromaps_cache"; CACHE.mkdir(parents=True, exist_ok=True)
N_SPINS = 1000
TOPK = 3

# neurotransmitter system -> (exact desc tracer tokens, mouse ground-truth structures, anchored?)
# NOTE: match on the DESC (tracer) field only, with explicit tracer names. Do NOT use short
# substrings like "d1"/"d2" -- they spuriously match years/other tracers (bedarD2019,
# fmpepD2=CB1, norgaarD2021=GABA), contaminating the group.
SYSTEMS = {
    "dopamine":  (["fdopa", "fallypride", "flb457", "raclopride", "sch23390", "pe2i",
                   "methylphenidate", "ioflupane", "fpcit"],
                  ["CP", "ACB", "SNc", "SNr", "VTA"], True),   # Parkinson substrate (nigrostriatal)
    "serotonin": (["dasb", "way100635", "cimbi", "altanserin", "sb207145", "p943", "az10419369"],
                  ["DR", "CP", "ACB"], False),                 # reference (raphe/striatal)
    "mu_opioid": (["carfentanil"],
                  ["CP", "ACB", "PAG"], False),                # reference (striatal/PAG)
}


def discover_mni_annotations():
    """Return list of (source, desc, space, den) for MNI152 volumetric neuromaps annotations."""
    from neuromaps.datasets import available_annotations
    out = []
    for ann in available_annotations():
        # ann is (source, desc, space, den)
        if len(ann) >= 4 and str(ann[2]).upper().startswith("MNI"):
            out.append(tuple(ann))
    return out


def fetch_one(source, desc, space, den):
    from neuromaps.datasets import fetch_annotation
    got = fetch_annotation(source=source, desc=desc, space=space, den=den, data_dir=str(CACHE))
    # fetch_annotation may return a path, list, or dict
    if isinstance(got, dict):
        got = list(got.values())
    if isinstance(got, (list, tuple)):
        got = got[0]
    return str(got)


def main():
    prov = pi_provenance(); print(f"pi: {prov['pi_file']} sha256={prov['pi_sha256'][:12]}")
    try:
        anns = discover_mni_annotations()
    except Exception as e:
        print("neuromaps not available or failed:", e, "\n-> pip install neuromaps"); return
    print(f"neuromaps: {len(anns)} MNI152 volumetric annotations available")

    pi = load_pi()
    acr = rt01.mouse_acr()
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    hxyz = H.var[["x", "y", "z"]].to_numpy(float)
    parc, numids = rt01.build_node_voxel_map(H.var)
    rown = pi / pi.sum(1, keepdims=True).clip(1e-12)
    spins = rt01.spin_indices(hxyz, N_SPINS)
    structs = [s for s in set(acr.tolist()) if s != "NA" and (acr == s).sum() >= 5]

    rows = []
    for sysname, (kws, gt, anchored) in SYSTEMS.items():
        # pick MNI annotations whose DESC (tracer) matches an explicit token (not source/year)
        picks = [a for a in anns if any(k in str(a[1]).lower() for k in kws)]
        # prefer coarser densities (faster, fine for parcel means); one per (source,desc)
        seen, chosen = set(), []
        for a in sorted(picks, key=lambda a: str(a[3])):
            key = (a[0], a[1])
            if key not in seen:
                seen.add(key); chosen.append(a)
        print(f"\n[{sysname}] {len(chosen)} maps: " + ", ".join(f"{a[0]}/{a[1]}" for a in chosen))
        for a in chosen:
            try:
                path = fetch_one(*a)
                v_h = rt01.sample_to_nodes(path, parc, numids)
            except Exception as e:
                print(f"    {a[0]}/{a[1]} FETCH/SAMPLE FAILED: {e}"); continue
            ok = np.isfinite(v_h); v_h = np.where(ok, v_h, np.nanmean(v_h[ok]))
            v_m = rown @ v_h
            sc = {s: float(np.nanmean(v_m[acr == s])) for s in structs}
            ranked = sorted(sc, key=sc.get, reverse=True)
            gt_present = [g for g in gt if g in sc]
            best = min((ranked.index(g) for g in gt_present), default=None)
            p = np.nan
            if gt_present:
                obs = max(sc[g] for g in gt_present)
                null = np.array([max(float(np.nanmean((rown @ v_h[perm])[acr == g])) for g in gt_present)
                                 for perm in spins])
                p = float((np.sum(null >= obs) + 1) / (N_SPINS + 1))
            h3 = any(g in ranked[:TOPK] for g in gt_present)
            rows.append({"system": sysname, "map": f"{a[0]}/{a[1]}", "anchored": anchored,
                         "ground_truth": gt_present, "best_gt_rank": None if best is None else best + 1,
                         "hit_top3": bool(h3), "spin_p": p,
                         "top3": [(s, round(sc[s], 4)) for s in ranked[:TOPK]]})
            sp = f"{p:.3f}" if p == p else "NA"
            mark = "OK " if h3 else "MISS"
            print(f"    {a[0]}/{a[1]:16s} {mark} GT rank {best+1 if best is not None else None} "
                  f"spin p={sp}  top: " + ", ".join(s for s, _ in rows[-1]["top3"]))

    # Parkinson verdict = fraction of dopamine maps hitting nigrostriatal top-3
    da = [r for r in rows if r["system"] == "dopamine"]
    da_hit = sum(r["hit_top3"] for r in da)
    da_sig = sum((r["spin_p"] == r["spin_p"]) and r["spin_p"] < 0.05 for r in da)
    out = {**prov, "n_spins": N_SPINS, "dopamine_maps": len(da), "dopamine_hit_top3": da_hit,
           "dopamine_spin_sig": da_sig, "results": rows}
    (ROOT / "outputs/logs/reverse_translation_neuromaps.json").write_text(json.dumps(out, indent=1))
    print("\n=================  NEUROMAPS SUBSTRATE REVERSE TRANSLATION  =================")
    if da:
        print(f"PARKINSON (dopamine system): {da_hit}/{len(da)} maps route to mouse nigrostriatal "
              f"(CP/ACB/SNc/SNr/VTA) top-3;  {da_sig}/{len(da)} spin-significant.")
        print("  -> anchored clinical hit" if da_hit >= max(1, len(da)//2) else
              "  -> still not landing on the dopamine substrate")
    else:
        print("no dopamine MNI maps matched — inspect the printed available-annotation keywords.")
    print("wrote outputs/logs/reverse_translation_neuromaps.json")


if __name__ == "__main__":
    main()
