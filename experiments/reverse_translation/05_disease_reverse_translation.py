#!/usr/bin/env python3
"""Disease reverse-translation: whether a human disease map routes to the mouse structure
where the field builds that disease's models.

Same pipeline as 01_validate.py, but the ground truth is the MOUSE-MODEL substrate (see
disease_ground_truth.md). Prediction, set a priori:
  * neurological diseases (Parkinson, Alzheimer, Huntington, epilepsy, addiction) -> HIT their
    canonical subcortical mouse substrate;
  * psychiatric conditions (schizophrenia, OCD, autism) -> miss / distributed, with autism the
    built-in expected-negative (no adequate mouse home).
The neuro-vs-psych contrast is the translational statement, scored against ground truth.

Needs disease_maps/<disease>.nii.gz (run 00b_fetch_disease_maps.py first).
Run: cd otter && PYTHONPATH=src python experiments/reverse_translation/05_disease_reverse_translation.py
Read-only; writes outputs/logs/reverse_translation_disease.json
"""
from __future__ import annotations
import json, sys, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached, load_pi, pi_provenance      # noqa: E402

_s = importlib.util.spec_from_file_location("rt01", ROOT / "experiments/reverse_translation/01_validate.py")
rt01 = importlib.util.module_from_spec(_s); _s.loader.exec_module(rt01)

MAPDIR = ROOT / "experiments/reverse_translation/disease_maps"
N_SPINS = 1000
TOPK = 3

# disease -> (map stem, acceptable mouse-model structures, group).  See disease_ground_truth.md
DISEASE_GT = {
    "parkinson":     ("parkinson",     ["SNc", "SNr", "CP", "STN", "VTA"],          "neurological"),
    "alzheimer":     ("alzheimer",     ["ENTl", "ENTm", "CA1", "SUB", "DG", "CA3"], "neurological"),
    "huntington":    ("huntington",    ["CP", "ACB"],                                "neurological"),
    "epilepsy":      ("epilepsy",      ["CA1", "CA3", "DG", "ENTl", "ENTm"],        "neurological"),
    "addiction":     ("addiction",     ["ACB", "VTA"],                               "neurological"),
    "schizophrenia": ("schizophrenia", ["CA1", "SUB", "PL", "ILA", "CP"],           "psychiatric"),
    "ocd":           ("obsessive",     ["CP", "ACB"],                                "psychiatric"),
    "autism":        ("autism",        [],                                           "psychiatric"),  # expected-negative
}


def main():
    prov = pi_provenance(); print(f"pi: {prov['pi_file']} sha256={prov['pi_sha256'][:12]}")
    if not MAPDIR.exists() or not any(MAPDIR.glob("*.nii*")):
        print(f"\nNO DISEASE MAPS in {MAPDIR}. Run 00b_fetch_disease_maps.py first."); return
    pi = load_pi()
    acr = rt01.mouse_acr()
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    hxyz = H.var[["x", "y", "z"]].to_numpy(float)
    parc, numids = rt01.build_node_voxel_map(H.var)
    rown = pi / pi.sum(1, keepdims=True).clip(1e-12)
    spins = rt01.spin_indices(hxyz, N_SPINS)
    structs = [s for s in set(acr.tolist()) if s != "NA" and (acr == s).sum() >= 5]

    rows, hits1, hits3, sig, n_anchored = [], 0, 0, 0, 0
    for dis, (term, gt, grp) in DISEASE_GT.items():
        cand = list(MAPDIR.glob(f"{term}.nii*")) + list(MAPDIR.glob(f"{dis}.nii*"))
        if not cand:
            rows.append({"disease": dis, "group": grp, "status": "map_missing"}); continue
        v_h = rt01.sample_to_nodes(cand[0], parc, numids)
        ok = np.isfinite(v_h); v_h = np.where(ok, v_h, np.nanmean(v_h[ok]))
        v_m = rown @ v_h
        sc = {s: float(np.nanmean(v_m[acr == s])) for s in structs}
        ranked = sorted(sc, key=sc.get, reverse=True)
        gt_present = [g for g in gt if g in sc]
        best = min((ranked.index(g) for g in gt_present), default=None)
        # spin null on the best ground-truth structure (only if GT defined)
        p = np.nan
        if gt_present:
            obs = max(sc[g] for g in gt_present)
            null = np.array([max(float(np.nanmean((rown @ v_h[perm])[acr == g])) for g in gt_present)
                             for perm in spins])
            p = float((np.sum(null >= obs) + 1) / (N_SPINS + 1))
        h1 = any(g in ranked[:1] for g in gt_present)
        h3 = any(g in ranked[:TOPK] for g in gt_present)
        if gt:                       # anchored disease (autism has empty GT -> not counted)
            n_anchored += 1; hits1 += h1; hits3 += h3; sig += (p < 0.05)
        rows.append({"disease": dis, "group": grp, "term": term, "ground_truth": gt_present,
                     "best_gt_rank": None if best is None else best + 1,
                     "hit_top1": bool(h1), "hit_top3": bool(h3), "spin_p": p,
                     "top3": [(s, round(sc[s], 4)) for s in ranked[:TOPK]]})

    out = {**prov, "n_spins": N_SPINS, "n_anchored": n_anchored,
           "hit_top1": hits1, "hit_top3": hits3, "n_spin_sig": sig, "results": rows}
    (ROOT / "outputs/logs/reverse_translation_disease.json").write_text(json.dumps(out, indent=1))

    print("\n=============  DISEASE REVERSE TRANSLATION  =============")
    for grp in ("neurological", "psychiatric"):
        print(f"\n[{grp}]")
        for r in rows:
            if r["group"] != grp:
                continue
            if r.get("status") == "map_missing":
                print(f"  {r['disease']:14s} MAP MISSING"); continue
            gt = "/".join(r["ground_truth"]) if r["ground_truth"] else "(expected none)"
            mark = "OK " if r["hit_top3"] else ("--" if not r["ground_truth"] else "MISS")
            sp = r["spin_p"]
            sps = f"{sp:.3f}" if sp == sp else "NA"
            print(f"  {r['disease']:14s} {mark} rank {r['best_gt_rank']}  spin p={sps}")
            print(f"       expect {gt};  top: " + ", ".join(s for s, _ in r["top3"]))
    if n_anchored:
        print(f"\nanchored diseases: {n_anchored}   hit@1 {hits1}   hit@3 {hits3}   spin-sig {sig}")
        neuro = [r for r in rows if r["group"] == "neurological" and "hit_top3" in r]
        nh = sum(r["hit_top3"] for r in neuro)
        print(f"neurological hit@3: {nh}/{len(neuro)}   "
              f"(the anchored-clinical result: conserved disease substrates recover their mouse model)")
    print("\nwrote outputs/logs/reverse_translation_disease.json")


if __name__ == "__main__":
    main()
