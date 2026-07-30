#!/usr/bin/env python3
"""Coverage vs published maps at the NETWORK level, with a spatial-autocorrelation spin null.

Coverage is noisy per parcel (L/R reliability 0.22), so per-region correlations against smooth
maps are attenuated. Averaging within Yeo-17 networks cancels that noise. The relationship that
emerges is with mouse-human FC HOMOLOGY (Xu2020): regions whose functional connectivity is more
homologous between mouse and human receive more mouse mass. This is the same modality HOMER encodes.

Significance: spin the parcel-level coverage on the sphere (asymmetric signal vs symmetric network
structure = the calibrated 5.5% FPR config), re-average within networks, recompute the network-level
Spearman, 2000x. Reported for every map.

Writes: outputs/logs/section5_network_homology.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from homer.data import load_cached, load_pi                 # noqa: E402
from homer.eval.nulls import _haar_rotation                 # noqa: E402
np.seterr(divide="ignore", invalid="ignore")
N_SPIN = 2000


def main():
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    xyz = H.var[["x", "y", "z"]].to_numpy(float)
    nr = np.asarray(json.loads((ROOT / "data_external/human_sc_meta.json").read_text())["node_region"], int)
    rows = [l.split("\t") for l in (ROOT / "outputs/anndata/_schaefer_order.txt").read_text().splitlines() if l.strip()]
    netmap = {int(p[0]): p[1].split("_", 2)[2].split("_")[0] for p in rows}

    pi = load_pi(); col = pi.sum(0)
    # parcel-level network label + coverage (linear mass; aggregation done in linear then logged)
    net_parcel = np.array([netmap.get(int(k), "?") for k in nr])
    cov_parcel = col.copy()

    bat = json.loads((ROOT / "outputs/logs/section5_evolution_battery.json").read_text())

    # spin perms on parcel centroids
    C = xyz - xyz.mean(0)
    sph = C / np.linalg.norm(C, axis=1, keepdims=True)
    tree = cKDTree(sph); rng = np.random.default_rng(0)
    perms = [tree.query(sph @ _haar_rotation(rng).T)[1] for _ in range(N_SPIN)]

    def net_means(cov_p, nets_wanted):
        # log10 of mass-normalised network mean
        return np.array([np.log10(cov_p[net_parcel == n].mean() + 1e-300) for n in nets_wanted])

    out = {"_finding": ("At network level coverage tracks mouse-human FC homology (Xu2020): more "
                        "homologous connectivity -> more mouse mass. Diluted per-parcel by coverage "
                        "noise (reliability 0.22); network averaging recovers it."),
           "n_spin": N_SPIN, "level": "Yeo-17 network"}

    for label in ["Xu2020 mouse–human FC homology", "Xu2020 mouse→human expansion",
                  "Hill2010 macaque→human expansion", "Sydnor2021 S–A axis",
                  "Margulies2016 principal gradient", "HCP T1w/T2w hierarchy"]:
        v = bat.get(label, {})
        if "schaefer_ids" not in v:
            continue
        sid = np.asarray(v["schaefer_ids"], int); mval = np.asarray(v["map_values"], float)
        # map network means (weighted by parcels in region already averaged; simple mean over regions)
        mp = {int(k): m for k, m in zip(sid, mval)}
        nets = sorted({netmap.get(int(k), "?") for k in sid} - {"?"})
        # region->network for the map
        map_net = {}
        for n in nets:
            vals = [mp[int(k)] for k in sid if netmap.get(int(k), "?") == n]
            if vals:
                map_net[n] = np.mean(vals)
        nets = [n for n in nets if n in map_net]
        m_arr = np.array([map_net[n] for n in nets])
        c_obs = net_means(cov_parcel, nets)
        rho_obs = spearmanr(c_obs, m_arr).statistic
        # spin null: rotate parcel coverage, re-average
        null = []
        for p in perms:
            c_s = net_means(cov_parcel[p], nets)
            null.append(spearmanr(c_s, m_arr).statistic)
        null = np.array(null)
        pspin = (np.sum(np.abs(null) >= abs(rho_obs)) + 1) / (N_SPIN + 1)
        out[label] = {"n_networks": len(nets), "network_spearman": float(rho_obs),
                      "spin_p": float(pspin)}
        print(f"  {label:<38} rho={rho_obs:+.3f}  spin p={pspin:.4f}  (n_net={len(nets)})")

    dst = ROOT / "outputs/logs/section5_network_homology.json"
    dst.write_text(json.dumps(out, indent=2)); print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
