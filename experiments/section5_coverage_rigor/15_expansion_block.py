#!/usr/bin/env python3
"""Does coverage track expansion when tested as a CONCENTRATED effect rather than a global gradient?

Positive controls proved the spin null detects r>=0.25 (script logs). Coverage vs Xu2020 expansion
is a global-correlation null (-0.05) because coverage is noisy per parcel (reliability 0.22) and its
deficit is concentrated in the most-expanded frontal cortex, not smoothly graded. A global Pearson is
underpowered for that shape. The shape that matches the biology is tested instead:

  - bin human regions by Xu2020 mouse->human expansion (tertiles / quintiles),
  - compare SOUND coverage (log10 mass-normalised region mean) across bins,
  - top-vs-bottom expansion contrast with a spin null on coverage (asymmetric signal vs the
    symmetric expansion map = the calibrated config),
  - monotone trend across bins (Spearman of bin-mean coverage vs bin index).

Writes: outputs/logs/section5_expansion_block.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached, load_pi                 # noqa: E402
from otter.eval.nulls import _haar_rotation                 # noqa: E402
np.seterr(divide="ignore", invalid="ignore")
N_SPIN = 2000


def main():
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    xyz = H.var[["x", "y", "z"]].to_numpy(float)
    nr = np.asarray(json.loads((ROOT / "data_external/human_sc_meta.json").read_text())["node_region"], int)
    pi = load_pi(); col = pi.sum(0)

    b = json.loads((ROOT / "outputs/logs/section5_evolution_battery.json").read_text())
    xu = b["Xu2020 mouse→human expansion"]
    exp = dict(zip(np.asarray(xu["schaefer_ids"], int), np.asarray(xu["map_values"], float)))

    ids = [k for k in range(1, 401) if (nr == k).any() and k in exp]
    cov = np.array([np.log10(col[nr == k].mean() + 1e-300) for k in ids])   # SOUND coverage
    ev = np.array([exp[k] for k in ids])
    cen = np.array([xyz[nr == k].mean(0) for k in ids])
    covz = (cov - cov.mean()) / cov.std()

    # spin perms on region centroids
    C = cen - cen.mean(0); sph = C / np.linalg.norm(C, axis=1, keepdims=True)
    tree = cKDTree(sph); rng = np.random.default_rng(0)
    perms = [tree.query(sph @ _haar_rotation(rng).T)[1] for _ in range(N_SPIN)]

    out = {"_q": "Coverage vs Xu2020 expansion as a concentrated (block) effect, not a global gradient.",
           "n_regions": len(ids), "coverage": "log10 mass-normalised region mean (sound)"}

    def contrast(hi_mask, lo_mask, name):
        obs = covz[hi_mask].mean() - covz[lo_mask].mean()
        null = np.array([covz[p][hi_mask].mean() - covz[p][lo_mask].mean() for p in perms])
        p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (N_SPIN + 1)
        r = {"gap_sd": float(obs), "spin_p": float(p),
             "hi_mean_cov_sd": float(covz[hi_mask].mean()), "lo_mean_cov_sd": float(covz[lo_mask].mean())}
        print(f"  {name:<32} coverage(hi-exp) - coverage(lo-exp) = {obs:+.2f} SD  spin p={p:.4f}")
        return r

    order = np.argsort(ev)
    n = len(ids)
    # tertiles
    t = n // 3
    lo_t = np.zeros(n, bool); lo_t[order[:t]] = True
    hi_t = np.zeros(n, bool); hi_t[order[-t:]] = True
    out["expansion_tertile_contrast"] = contrast(hi_t, lo_t, "expansion tertiles (top vs bottom)")
    # quintiles
    q = n // 5
    lo_q = np.zeros(n, bool); lo_q[order[:q]] = True
    hi_q = np.zeros(n, bool); hi_q[order[-q:]] = True
    out["expansion_quintile_contrast"] = contrast(hi_q, lo_q, "expansion quintiles (top vs bottom)")
    # top-decile "most expanded"
    d = max(5, n // 10)
    hi_d = np.zeros(n, bool); hi_d[order[-d:]] = True
    rest = ~hi_d
    out["most_expanded_decile_vs_rest"] = contrast(hi_d, rest, "most-expanded decile vs rest")

    # monotone trend across 5 bins
    bins = np.array_split(order, 5)
    binmeans = [float(covz[bi].mean()) for bi in bins]
    out["quintile_bin_mean_coverage_sd"] = binmeans
    out["monotone_trend_spearman"] = float(spearmanr(range(5), binmeans).statistic)
    print(f"  quintile coverage means (low->high expansion): "
          f"{['%+.2f' % x for x in binmeans]}  trend rho={out['monotone_trend_spearman']:+.2f}")

    dst = ROOT / "outputs/logs/section5_expansion_block.json"
    dst.write_text(json.dumps(out, indent=2)); print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
