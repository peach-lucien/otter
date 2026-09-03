#!/usr/bin/env python3
"""Subcortical-coverage disease-substrate reverse translation.

Neurosynth disease maps do not image the subcortex/midbrain, so the disease substrate is
absent from the input (Parkinson lands in thalamus, SNc unsampled). This script uses the
neuromaps molecular atlas (Hansen 2022) instead: PET neurotransmitter-system maps, MNI152
volumetric, with striatal and midbrain coverage.

Key test (Parkinson): the human dopamine system map (DAT / dopamine synthesis / D1/D2) is the
substrate that mouse PD models target (nigrostriatal). The test is whether it
reverse-translates to the mouse dopamine system (CP, ACB, SNc, SNr, VTA). Serotonin,
mu-opioid, cannabinoid CB1 and GABA-A benzodiazepine maps are carried as reference systems
for specificity. The per-map striatal mass fraction is computed and
written here rather than being reconstructed downstream by the figure code.

Requires neuromaps:  pip install neuromaps
Run: cd otter && PYTHONPATH=src python experiments/reverse_translation/06_neuromaps_substrate.py
Read-only w.r.t. the coupling; caches maps in neuromaps_cache/, writes
outputs/logs/reverse_translation_neuromaps.json
"""
from __future__ import annotations
import hashlib
import json
import sys
import importlib.util
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached, load_pi, pi_provenance      # noqa: E402

_s = importlib.util.spec_from_file_location("rt01", ROOT / "experiments/reverse_translation/01_validate.py")
rt01 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(rt01)

CACHE = ROOT / "experiments/reverse_translation/neuromaps_cache"
CACHE.mkdir(parents=True, exist_ok=True)
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
    # fmpepD2 is the tracer name for the CB1 ligand [18F]FMPEP-d2; it is not a D2-receptor map.
    "cannabinoid_cb1": (["fmpepd2"], [], False),               # cortical reference
    "gaba_a": (["flumazenil"], [], False),                     # cortical reference
}

STRIATUM = ("CP", "ACB", "OT")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def striatal_mass_fraction(v_m, acr, structs):
    """Share of positive structure-level translated mass in CP, ACB and OT."""
    positive = {s: max(float(np.nanmean(v_m[acr == s])), 0.0) for s in structs}
    total = float(sum(positive.values()))
    return float(sum(positive.get(s, 0.0) for s in STRIATUM) / total) if total else 0.0


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
    prov = pi_provenance()
    print(f"pi: {prov['pi_file']} sha256={prov['pi_sha256'][:12]}")
    try:
        anns = discover_mni_annotations()
    except Exception as e:
        print("neuromaps not available or failed:", e, "\n-> pip install neuromaps")
        return
    print(f"neuromaps: {len(anns)} MNI152 volumetric annotations available")

    pi = load_pi()
    acr = rt01.mouse_acr()
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    hxyz = H.var[["x", "y", "z"]].to_numpy(float)
    parc, numids = rt01.build_node_voxel_map(H.var)
    rown = pi / pi.sum(1, keepdims=True).clip(1e-12)
    spins = rt01.spin_indices(hxyz, N_SPINS)
    structs = [s for s in set(acr.tolist()) if s != "NA" and (acr == s).sum() >= 5]

    rows, failures = [], []
    for sysname, (kws, gt, anchored) in SYSTEMS.items():
        # pick MNI annotations whose DESC (tracer) matches an explicit token (not source/year)
        picks = [a for a in anns if any(k in str(a[1]).lower() for k in kws)]
        # prefer coarser densities (faster, fine for parcel means); one per (source,desc)
        seen, chosen = set(), []
        for a in sorted(picks, key=lambda a: str(a[3])):
            key = (a[0], a[1])
            if key not in seen:
                seen.add(key)
                chosen.append(a)
        print(f"\n[{sysname}] {len(chosen)} maps: " + ", ".join(f"{a[0]}/{a[1]}" for a in chosen))
        for a in chosen:
            try:
                path = fetch_one(*a)
                v_h = rt01.sample_to_nodes(path, parc, numids)
            except Exception as e:
                message = f"{type(e).__name__}: {e}"
                failures.append({"system": sysname, "map": f"{a[0]}/{a[1]}", "error": message})
                print(f"    {a[0]}/{a[1]} FETCH/SAMPLE FAILED: {message}")
                continue
            ok = np.isfinite(v_h)
            v_h = np.where(ok, v_h, np.nanmean(v_h[ok]))
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
            frac = striatal_mass_fraction(v_m, acr, structs)
            rows.append({"system": sysname, "map": f"{a[0]}/{a[1]}", "anchored": anchored,
                         "annotation": {"source": str(a[0]), "desc": str(a[1]),
                                        "space": str(a[2]), "density": str(a[3])},
                         "source_file": str(Path(path).resolve().relative_to(ROOT.resolve())),
                         "source_sha256": sha256(path),
                         "ground_truth": gt_present, "best_gt_rank": None if best is None else best + 1,
                         "hit_top3": bool(h3), "spin_p": p,
                         "striatal_mass_fraction": frac,
                         "top3": [(s, round(sc[s], 4)) for s in ranked[:TOPK]]})
            sp = f"{p:.3f}" if p == p else "NA"
            mark = "OK " if h3 else "MISS"
            print(f"    {a[0]}/{a[1]:16s} {mark} GT rank {best+1 if best is not None else None} "
                  f"spin p={sp} striatal fraction={frac:.3f}  top: "
                  + ", ".join(s for s, _ in rows[-1]["top3"]))

    # Parkinson verdict = fraction of dopamine maps hitting nigrostriatal top-3
    da = [r for r in rows if r["system"] == "dopamine"]
    da_hit = sum(r["hit_top3"] for r in da)
    da_sig = sum((r["spin_p"] == r["spin_p"]) and r["spin_p"] < 0.05 for r in da)
    summaries = {}
    for system in SYSTEMS:
        fractions = [r["striatal_mass_fraction"] for r in rows if r["system"] == system]
        summaries[system] = {
            "n_maps": len(fractions),
            "striatal_mass_fractions": fractions,
            "mean": float(np.mean(fractions)) if fractions else None,
            "median": float(np.median(fractions)) if fractions else None,
        }
    reference = [r["striatal_mass_fraction"] for r in rows if r["system"] != "dopamine"]
    specificity = None
    if da and reference:
        da_fraction = [r["striatal_mass_fraction"] for r in da]
        u = mannwhitneyu(da_fraction, reference, alternative="greater", method="exact")
        specificity = {
            "contrast": "dopamine > pooled non-dopamine reference systems",
            "test": "one-sided exact Mann-Whitney U",
            "u": float(u.statistic),
            "p": float(u.pvalue),
            "n_dopamine": len(da_fraction),
            "n_reference": len(reference),
        }
    out = {**prov, "n_spins": N_SPINS, "striatal_structures": list(STRIATUM),
           "striatal_fraction_definition": ("sum of positive translated structure means in "
                                             "CP, ACB and OT divided by total positive "
                                             "translated structure mass"),
           "dopamine_maps": len(da), "dopamine_hit_top3": da_hit,
           "dopamine_spin_sig": da_sig, "system_summary": summaries,
           "specificity_test": specificity, "results": rows, "failures": failures}
    (ROOT / "outputs/logs/reverse_translation_neuromaps.json").write_text(json.dumps(out, indent=1))
    print("\n=================  NEUROMAPS SUBSTRATE REVERSE TRANSLATION  =================")
    if da:
        print(f"PARKINSON (dopamine system): {da_hit}/{len(da)} maps route to mouse nigrostriatal "
              f"(CP/ACB/SNc/SNr/VTA) top-3;  {da_sig}/{len(da)} spin-significant.")
        print("  -> anchored clinical hit" if da_hit >= max(1, len(da)//2) else
              "  -> still not landing on the dopamine substrate")
    else:
        print("no dopamine MNI maps matched; inspect the printed available-annotation keywords.")
    print("wrote outputs/logs/reverse_translation_neuromaps.json")


if __name__ == "__main__":
    main()
