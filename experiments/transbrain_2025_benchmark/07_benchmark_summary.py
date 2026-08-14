#!/usr/bin/env python3

"""Benchmark metrics for both methods, derived from the cached Brainnetome distributions.

AUROC, top-k, mass-in-region, sharpness and the per-region win counts are computed from
`transbrain_bn_distributions.json` and written to a log.

Requires `outputs/logs/transbrain_bn_distributions.json`, which is committed, so this script
runs on a fresh clone without the Zenodo download.

Run: cd otter && PYTHONPATH=src python experiments/transbrain_2025_benchmark/07_benchmark_summary.py
Writes outputs/logs/transbrain_benchmark_summary.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
BN = json.loads((ROOT / "outputs/logs/transbrain_bn_distributions.json").read_text())


def auroc(w, true):
    w = np.asarray(w, float)
    pos, neg = w[list(true)], np.delete(w, list(true))
    return float(sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg)))


def topk(w, true, k):
    return float(len(set(np.argsort(-np.asarray(w, float))[:k]) & set(true)) > 0)


def eff_n(w):
    w = np.asarray(w, float)
    s = w.sum()
    return float(1 / np.sum((w / s) ** 2)) if s > 0 else np.nan


def main():
    O = {k: [] for k in ("top1", "top3", "top5", "auroc", "mass", "eff_n")}
    T = {k: [] for k in O}
    for _, o in BN["regions"].items():
        for D, w in ((O, o["otter_w"]), (T, o["tb_w"])):
            tr = o["true"]
            D["top1"].append(topk(w, tr, 1))
            D["top3"].append(topk(w, tr, 3))
            D["top5"].append(topk(w, tr, 5))
            D["auroc"].append(auroc(w, tr))
            D["mass"].append(float(np.asarray(w, float)[list(tr)].sum()))
            D["eff_n"].append(eff_n(w))

    oa, ta = np.array(O["auroc"]), np.array(T["auroc"])
    out = {
        "n_benchmark_regions": len(oa),
        "n_bn_regions": len(BN["bn_names"]),
        "otter": {k: float(np.nanmean(v)) for k, v in O.items()},
        "transbrain": {k: float(np.nanmean(v)) for k, v in T.items()},
        "auroc_wins": {"otter": int(np.sum(oa > ta)), "transbrain": int(np.sum(ta >= oa))},
        "auroc_paired_wilcoxon_p": float(wilcoxon(oa, ta).pvalue),
        "_note": ("AUROC/top-k/mass are means over the 24 benchmark regions, scored as "
                  "distributions over the common Brainnetome set. The win split is by AUROC; "
                  "the mass-in-region split differs and must not be quoted as the AUROC split."),
    }
    p = ROOT / "outputs/logs/transbrain_benchmark_summary.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"OTTER      AUROC {out['otter']['auroc']:.4f}  top1 {out['otter']['top1']:.3f}  "
          f"top5 {out['otter']['top5']:.3f}  mass {out['otter']['mass']:.3f}  effN {out['otter']['eff_n']:.1f}")
    print(f"TransBrain AUROC {out['transbrain']['auroc']:.4f}  top1 {out['transbrain']['top1']:.3f}  "
          f"top5 {out['transbrain']['top5']:.3f}  mass {out['transbrain']['mass']:.3f}  effN {out['transbrain']['eff_n']:.1f}")
    print(f"AUROC wins: TransBrain {out['auroc_wins']['transbrain']} / OTTER {out['auroc_wins']['otter']}   "
          f"paired Wilcoxon p = {out['auroc_paired_wilcoxon_p']:.3f}")
    print("wrote", p)


if __name__ == "__main__":
    main()
