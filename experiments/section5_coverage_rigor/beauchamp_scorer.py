#!/usr/bin/env python3
"""Reusable external Beauchamp scorer: precompute region masks once, score any in-memory pi.

Shared yardstick for (i) eps selection, (ii) the region x hyperparameter map, (iii) the warped
spatial-term evaluation. Wraps pipeline/05f_beauchamp_validation.py's mask logic and evaluate_pair.

    from beauchamp_scorer import BeauchampScorer
    sc = BeauchampScorer()                 # builds/loads cached masks
    res = sc.score(pi)                     # {pair: metrics, '__aggregate__': {...}}

Per-pair metrics include top1/top5/top10, mean_mass_in_region, mean_rank_in_region. The aggregate
adds parcel-count-weighted top-k and enrichment over a uniform-argmax chance baseline.
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
MASKCACHE = Path("/var/tmp/beauchamp_masks.npz")

# import the heavy mask/eval functions from the pipeline script
_spec = importlib.util.spec_from_file_location("b05f", ROOT / "pipeline/05f_beauchamp_validation.py")
b05f = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b05f)


class BeauchampScorer:
    def __init__(self):
        from otter.data import load_cached
        self.M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
        self.H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
        self.h_xyz = self.H.var[["x", "y", "z"]].to_numpy()
        self._build_or_load_masks()

    def _build_or_load_masks(self):
        if MASKCACHE.exists():
            z = np.load(MASKCACHE, allow_pickle=True)
            self.pairs = list(z["pairs"])
            self.m_masks = {p: z[f"m_{i}"] for i, p in enumerate(self.pairs)}
            self.h_masks = {p: z[f"h_{i}"] for i, p in enumerate(self.pairs)}
            self.is_anchor = {p: bool(z[f"a_{i}"]) for i, p in enumerate(self.pairs)}
            return
        EXT = ROOT / "data_external/MouseHumanTranscriptomicSimilarity"
        name_to_dsurqe = b05f.parse_dsurqe_tree(EXT / "AMBA/data/DSURQE_tree.json")
        parcel_dsurqe = b05f.assign_dsurqe_labels(self.M, EXT / "AMBA/data/imaging/DSURQE_CCFv3_labels_200um.mnc")
        human_membership = b05f.assign_human_region_membership(self.H, b05f.HUMAN_REGION_MNI)
        HIPPO = {"Subiculum", "Field CA1", "Field CA2", "Field CA3", "Dentate gyrus", "Medulla"}
        self.pairs, self.m_masks, self.h_masks, self.is_anchor = [], {}, {}, {}
        for mname, hname in b05f.BEAUCHAMP_PAIRS:
            if mname not in name_to_dsurqe:
                continue
            m_mask = np.isin(parcel_dsurqe, list(name_to_dsurqe[mname]))
            h_mask = human_membership[hname]
            if h_mask is None or m_mask.sum() == 0 or h_mask.sum() == 0:
                continue
            key = f"{mname} -> {hname}"
            self.pairs.append(key)
            self.m_masks[key] = m_mask
            self.h_masks[key] = h_mask
            self.is_anchor[key] = (mname not in HIPPO)
        save = {"pairs": np.array(self.pairs, dtype=object)}
        for i, p in enumerate(self.pairs):
            save[f"m_{i}"] = self.m_masks[p]; save[f"h_{i}"] = self.h_masks[p]
            save[f"a_{i}"] = self.is_anchor[p]
        np.savez_compressed(MASKCACHE, **save)

    def score(self, pi):
        pi = np.asarray(pi, float)
        N_h = pi.shape[1]
        res = {}
        succ = []
        for p in self.pairs:
            m = self.evaluate(pi, self.m_masks[p], self.h_masks[p])
            if m is None:
                continue
            m["is_anchor_overlapping"] = self.is_anchor[p]
            res[p] = m
            succ.append(m)
        if succ:
            wts = np.array([r["n_mouse_parcels"] for r in succ], float); ws = wts.sum()
            agg = {k: float(sum(wts[i] * succ[i][k] for i in range(len(succ))) / ws)
                   for k in ("top1", "top5", "top10", "mean_mass_in_region", "mean_rank_in_region")}
            ch1 = float(sum((wts[i] / ws) * (1 - (1 - succ[i]["n_human_parcels"] / N_h)) for i in range(len(succ))))
            agg["chance_top1"] = ch1
            agg["enrichment_top1"] = agg["top1"] / max(ch1, 1e-9)
            agg["n_pairs"] = len(succ)
            res["__aggregate__"] = agg
        return res

    def evaluate(self, pi, m_mask, h_mask):
        return b05f.evaluate_pair(pi, m_mask, h_mask, self.h_xyz, k_top=10)


if __name__ == "__main__":
    import json
    sc = BeauchampScorer()
    print(f"masks built: {len(sc.pairs)} scorable pairs")
    # Verify the reference coupling and score the epsilon family.
    from otter.data import load_pi
    prod = load_pi()
    for name, pi in [("production", prod)]:
        a = sc.score(pi)["__aggregate__"]
        print(f"{name:14s} top1={a['top1']:.3f} top5={a['top5']:.3f} "
              f"mass={a['mean_mass_in_region']:.3f} enrich_top1={a['enrichment_top1']:.1f}x "
              f"({a['n_pairs']} pairs)")
