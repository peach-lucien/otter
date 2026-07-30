#!/usr/bin/env python3
"""Why does the coverage<->expansion link break at visual cortex and default-mode? Is it xyz?

At network level, sound coverage vs Xu2020 expansion is weakly negative (expected direction) but
diluted by two breakers:
  - VISUAL: low coverage but NOT expanded.
  - DEFAULT-MODE (DefaultA/C): expanded but WELL covered.

Test whether the spatial (xyz) cost term drives them by comparing coverage on the production
coupling vs the xyz-zeroed coupling, per network, alongside each network's spatial isolation
(mean distance from its human parcels to the nearest mouse parcel, from M_xyz) and Xu expansion.

Writes: outputs/logs/section5_why_visual_dmn.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from homer.data import load_cached, load_pi                 # noqa: E402
np.seterr(divide="ignore", invalid="ignore")


def main():
    nr = np.asarray(json.loads((ROOT / "data_external/human_sc_meta.json").read_text())["node_region"], int)
    rows = [l.split("\t") for l in (ROOT / "outputs/anndata/_schaefer_order.txt").read_text().splitlines() if l.strip()]
    net = {int(p[0]): p[1].split("_", 2)[2].split("_")[0] for p in rows}
    costs = np.load(ROOT / "outputs/anndata/full_costs.npz")
    Mxyz = costs["M_xyz"]                                    # (mouse, human) cross spatial cost
    spatial_isolation = Mxyz.min(0)                          # per human parcel: cost of nearest mouse

    b = json.loads((ROOT / "outputs/logs/section5_evolution_battery.json").read_text())
    xu = dict(zip(np.asarray(b["Xu2020 mouse→human expansion"]["schaefer_ids"], int),
                  np.asarray(b["Xu2020 mouse→human expansion"]["map_values"], float)))

    prod = load_pi().sum(0)
    xyz0 = np.load(ROOT / "outputs/coupling/pi_fc_plus_SC_xyz_zero.npy").sum(0)

    ids = [k for k in range(1, 401) if (nr == k).any() and k in xu]

    def rnet(vec, agg="logmean"):
        d = defaultdict(list)
        for k in ids:
            v = np.log10(vec[nr == k].mean() + 1e-300) if agg == "logmean" else vec[nr == k].mean()
            d[net[k]].append(v)
        return {n: float(np.mean(v)) for n, v in d.items()}

    cov_p = rnet(prod); cov_0 = rnet(xyz0)
    iso = rnet(spatial_isolation, agg="lin")
    exp = defaultdict(list)
    for k in ids:
        exp[net[k]].append(xu[k])
    exp = {n: float(np.mean(v)) for n, v in exp.items()}

    order = sorted(cov_p, key=lambda n: cov_p[n])
    print(f"{'network':<14}{'cov_prod':>9}{'cov_xyz0':>9}{'d_cov':>7}{'spatial_iso':>12}{'Xu_exp':>8}")
    out = {"per_network": {}}
    for n in order:
        d = cov_0[n] - cov_p[n]
        out["per_network"][n] = {"cov_prod": cov_p[n], "cov_xyz0": cov_0[n], "delta": d,
                                 "spatial_isolation": iso[n], "xu_expansion": exp[n]}
        print(f"{n:<14}{cov_p[n]:>9.2f}{cov_0[n]:>9.2f}{d:>+7.2f}{iso[n]:>12.3f}{exp[n]:>8.1f}")

    nets = order
    cp = np.array([cov_p[n] for n in nets]); c0 = np.array([cov_0[n] for n in nets])
    ex = np.array([exp[n] for n in nets]); isov = np.array([iso[n] for n in nets])
    out["corr"] = {
        "cov_prod_vs_expansion": float(spearmanr(cp, ex).statistic),
        "cov_xyz0_vs_expansion": float(spearmanr(c0, ex).statistic),
        "cov_prod_vs_spatial_isolation": float(spearmanr(cp, isov).statistic),
        "expansion_vs_spatial_isolation": float(spearmanr(ex, isov).statistic),
    }
    print("\nnetwork-level Spearman:")
    for k, v in out["corr"].items():
        print(f"  {k:<38} {v:+.3f}")

    dst = ROOT / "outputs/logs/section5_why_visual_dmn.json"
    dst.write_text(json.dumps(out, indent=2)); print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
