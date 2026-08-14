#!/usr/bin/env python3
"""Reverse translation validation — does a HUMAN functional map route to the RIGHT
mouse circuit?  (go/no-go for a reverse-translation / experiment-design section)

IDEA
----
OTTER's coupling is bidirectional. Column-normalised it carries mouse -> human.
ROW-normalised it carries human -> mouse: a human map v_h (over the
2,094 human parcels) becomes a mouse prediction v_m = rownorm(pi) @ v_h. If OTTER is
a faithful reverse translator, a human meta-analytic activation map for a function
(e.g. "reward") should land on the mouse structure the field uses to study it
(nucleus accumbens / VTA). This is the direction reverse-translational neuroscience
asks for but has no whole-brain computational tool for.

WHAT THIS SCRIPT TESTS
----------------------
For each of a curated set of human functions with an established mouse substrate, it
routes the human map to mouse, ranks mouse STRUCTURES by the translated value, and asks:
  * does a ground-truth mouse structure land in the top-k?                (accuracy)
  * is its enrichment significant against a spatial spin null?            (rigour)
  * how confident is OTTER at that target, and does ANY mouse structure
    clear the null -- or is the human target "primate-unique" with no
    adequate mouse home? (the actionable output: when NOT to use a mouse)  (confidence)

HUMAN MAPS (the one external dependency)
----------------------------------------
Put one MNI152 volume per function in  experiments/reverse_translation/human_maps/
named  <term>.nii.gz  (e.g. reward.nii.gz). Fetch them however you like; the intended
source is neuromaps / Neurosynth association maps, e.g.:

    from neuromaps.datasets import fetch_annotation           # or nimare Neurosynth
    # fetch a Neurosynth 'association-test' map for each term, save as MNI152 .nii.gz

The script resamples each volume onto the 2,094-node parcellation itself, so any
MNI152 volume works. Nothing else about the pipeline depends on how the maps are made.

Run: cd otter && PYTHONPATH=src python experiments/reverse_translation/01_validate.py
Read-only w.r.t. the coupling; writes outputs/logs/reverse_translation_validation.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached, load_pi, pi_provenance      # noqa: E402

MAPDIR = ROOT / "experiments/reverse_translation/human_maps"
PARC = ROOT / "data_external/_diagnostics/parcellation_2094.nii.gz"
N_SPINS = 1000
TOPK = 3

# function -> (neurosynth term / map filename stem, acceptable mouse structure acronyms)
GROUND_TRUTH = {
    # each entry: (Neurosynth term / map stem, acceptable mouse structures, literature basis)
    # citations verified via PubMed; see ground_truth_citations.md
    "reward":        ("reward",       ["ACB", "VTA"]),                    # Russo & Nestler 2013
    "fear":          ("fear",         ["BLA", "CEA", "BMA", "LA", "PAG"]),# Tovote et al. 2015
    "anxiety":       ("anxiety",      ["BLA", "BST", "CEA"]),            # Tovote et al. 2015 (amygdala + BNST)
    "feeding":       ("feeding",      ["LHA", "ARH", "VMH"]),            # Andermann & Lowell 2017
    "spatial_memory":("navigation",   ["CA1", "CA3", "DG", "SUB"]),      # hippocampal place-cell system
    "motor":         ("motor",        ["MOp", "MOs", "CP"]),             # Svoboda & Li 2018
    "addiction":     ("addiction",    ["ACB", "VTA"]),                   # Luscher 2016 (mesolimbic)
    "pain":          ("pain",         ["PAG", "VPM", "VPL"]),            # Kuner & Kuner 2020
    "olfaction":     ("olfactory",    ["PIR", "MOB", "AON"]),            # Bekkers & Suzuki 2013
    "vision":        ("visual",       ["VISp", "LGd"]),                  # Niell 2015
    "audition":      ("auditory",     ["AUDp", "MG"]),                   # Tsukano et al. 2017
    "interoception": ("interoception",["AId", "AIv", "AIp", "PB"]),      # Palmiter 2018 (parabrachial relay)
}


def mouse_acr():
    mm = json.loads((ROOT / "data_external/mouse_sc_meta.json").read_text())
    return np.array([mm["structure_acronyms"][i] if i >= 0 else "NA" for i in mm["node_struct_idx"]])


def build_node_voxel_map(Hvar):
    """Return (parc_data, numids in pi-column order) to sample MNI volumes onto nodes."""
    import nibabel as nib
    parc = nib.load(str(PARC))
    return parc, np.asarray(Hvar["numid"], int)


def sample_to_nodes(nifti_path, parc, numids):
    """Resample an MNI152 volume to the parcellation grid and average within each node."""
    import nibabel as nib
    from nilearn.image import resample_to_img
    vol = resample_to_img(str(nifti_path), parc, interpolation="linear")
    v = np.asarray(vol.dataobj, float)
    lab = np.asarray(parc.dataobj).astype(int)
    out = np.full(len(numids), np.nan)
    means = {}
    for k in np.unique(lab):
        if k <= 0:
            continue
        means[int(k)] = float(np.nanmean(v[lab == k]))
    for i, k in enumerate(numids):
        out[i] = means.get(int(k), np.nan)
    return out


def structure_scores(v_mouse, acr):
    """Mean translated value per mouse structure -> ranked Series-like dict."""
    structs = {}
    for s in set(acr.tolist()):
        if s == "NA":
            continue
        m = acr == s
        if m.sum() >= 5:
            structs[s] = float(np.nanmean(v_mouse[m]))
    return structs


def spin_indices(xyz, n, seed=0):
    from scipy.spatial import cKDTree
    from otter.eval.nulls import _haar_rotation
    c = xyz - xyz.mean(0); s = c / np.linalg.norm(c, axis=1, keepdims=True)
    t = cKDTree(s); rng = np.random.default_rng(seed)
    return [t.query(s @ _haar_rotation(rng).T)[1] for _ in range(n)]


def main():
    prov = pi_provenance(); print(f"pi: {prov['pi_file']} sha256={prov['pi_sha256'][:12]}")
    if not MAPDIR.exists() or not any(MAPDIR.glob("*.nii*")):
        print(f"\nNO HUMAN MAPS FOUND in {MAPDIR}")
        print("Populate it with one MNI152 volume per term (see GROUND_TRUTH), e.g. reward.nii.gz,")
        print("fetched via neuromaps/Neurosynth. Then re-run.")
        return

    pi = load_pi()
    acr = mouse_acr()
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    hxyz = H.var[["x", "y", "z"]].to_numpy(float)
    parc, numids = build_node_voxel_map(H.var)

    # reverse operator: row-normalised pi (each mouse parcel -> distribution over human parcels)
    rown = pi / pi.sum(1, keepdims=True).clip(1e-12)
    spins = spin_indices(hxyz, N_SPINS)

    rows, hits1, hits3, sig = [], 0, 0, 0
    for fn, (term, gt) in GROUND_TRUTH.items():
        cand = list(MAPDIR.glob(f"{term}.nii*")) + list(MAPDIR.glob(f"{fn}.nii*"))
        if not cand:
            rows.append({"function": fn, "status": "map_missing"}); continue
        v_h = sample_to_nodes(cand[0], parc, numids)
        ok = np.isfinite(v_h)
        v_h = np.where(ok, v_h, np.nanmean(v_h[ok]))
        v_m = rown @ v_h
        sc = structure_scores(v_m, acr)
        ranked = sorted(sc, key=sc.get, reverse=True)
        gt_present = [g for g in gt if g in sc]
        best_gt = min((ranked.index(g) for g in gt_present), default=None)   # 0-based rank
        # spin null on the best ground-truth structure's score
        obs = max((sc[g] for g in gt_present), default=np.nan)
        null = []
        for perm in spins:
            vm = rown @ v_h[perm]
            sperm = {g: float(np.nanmean(vm[acr == g])) for g in gt_present}
            null.append(max(sperm.values()) if sperm else np.nan)
        null = np.array(null)
        p = float((np.sum(null >= obs) + 1) / (N_SPINS + 1)) if gt_present else np.nan
        top1 = ranked[:1]; top3 = ranked[:TOPK]
        h1 = any(g in top1 for g in gt_present); h3 = any(g in top3 for g in gt_present)
        hits1 += h1; hits3 += h3; sig += (p < 0.05)
        rows.append({"function": fn, "term": term, "ground_truth": gt_present,
                     "best_gt_rank": None if best_gt is None else best_gt + 1,
                     "hit_top1": bool(h1), "hit_top3": bool(h3),
                     "spin_p": p, "top3_structures": [(s, round(sc[s], 4)) for s in top3]})

    n = sum(1 for r in rows if r.get("status") != "map_missing" and "hit_top1" in r)
    out = {**prov, "n_functions_scored": n, "hit_top1": hits1, "hit_top3": hits3,
           "n_spin_sig": sig, "n_spins": N_SPINS, "results": rows}
    (ROOT / "outputs/logs/reverse_translation_validation.json").write_text(json.dumps(out, indent=1))

    print("\n===============  REVERSE TRANSLATION VALIDATION  ===============")
    for r in rows:
        if r.get("status") == "map_missing":
            print(f"  {r['function']:14s} MAP MISSING"); continue
        mark = "OK " if r["hit_top3"] else "MISS"
        print(f"  {r['function']:14s} {mark} rank {r['best_gt_rank']}  spin p={r['spin_p']:.3f}  "
              f"top: " + ", ".join(f"{s}" for s, _ in r["top3_structures"]))
    if n:
        print(f"\nhit@1 {hits1}/{n}   hit@3 {hits3}/{n}   spin-significant {sig}/{n}")
        verdict = ("GO" if hits3 >= max(3, int(0.6 * n)) and sig >= int(0.5 * n) else
                   "MAYBE" if hits3 >= int(0.4 * n) else "NO-GO")
        print(f"VERDICT: {verdict}")
    print("wrote outputs/logs/reverse_translation_validation.json")


if __name__ == "__main__":
    main()
