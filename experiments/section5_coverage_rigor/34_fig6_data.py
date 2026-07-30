"""Generate all data Figure 6 needs: full Yeo-17 network z-profiles for the AI-opto circuit and the
5 autism-mutation circuits (canonical pi), the AI-opto salience-enrichment permuted-pi null
distribution, and the HOMER-vs-TransBrain head-to-head numbers.

Writes outputs/logs/fig6_data.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from homer.data import load_cached, load_pi, pi_provenance  # noqa: E402
DATA = ROOT / "data_external/transbrain_2025"


def route(vb, parcel_acr, pi):
    mvec = np.array([vb.get(a, np.nan) for a in parcel_acr]); mask = np.isfinite(mvec)
    num = mvec[mask] @ pi[mask, :]; den = pi[mask, :].sum(0)
    out = np.full(pi.shape[1], np.nan); ok = den > 1e-12; out[ok] = num[ok] / den[ok]
    return out


def main():
    mm = json.loads((ROOT / "data_external/mouse_sc_meta.json").read_text())
    parcel_acr = np.array([mm["structure_acronyms"][i] for i in mm["node_struct_idx"]])
    nr = np.asarray(json.loads((ROOT / "data_external/human_sc_meta.json").read_text())["node_region"], int)
    rows = [l.split("\t") for l in (ROOT / "outputs/anndata/_schaefer_order.txt").read_text().splitlines() if l.strip()]
    net = np.array([{int(p[0]): p[1].split("_", 2)[2].split("_")[0] for p in rows}.get(int(k), "?") for k in nr])
    pi = load_pi()
    nets = sorted({u for u in set(net) if u != "?" and (net == u).sum() >= 10})

    def profile(hum):
        m = np.isfinite(hum); z = (hum - np.nanmean(hum[m])) / np.nanstd(hum[m])
        return {u: float(np.nanmean(z[m & (net == u)])) for u in nets}, z, m

    out = {"networks": nets}
    ai = pd.read_csv(DATA / "ai_opto.csv", index_col=0).iloc[:, 0].to_dict()
    prof, z, m = profile(route(ai, parcel_acr, pi))
    out["aiopto_network_profile"] = prof
    # permuted-pi null distribution of salience enrichment
    sal = m & np.char.startswith(net.astype(str), "SalVentAttn")
    obs = float(np.nanmean(z[sal]) - np.nanmean(z[m & ~sal]))
    rng = np.random.default_rng(0); null = []
    for _ in range(1000):
        hh = route(ai, parcel_acr, pi[rng.permutation(pi.shape[0])]); mm2 = np.isfinite(hh)
        zz = (hh - np.nanmean(hh[mm2])) / np.nanstd(hh[mm2])
        null.append(float(np.nanmean(zz[sal & mm2]) - np.nanmean(zz[m & ~sal & mm2])))
    out["aiopto_salience_obs"] = obs; out["aiopto_salience_null"] = null

    # autism heterogeneity: 5 genes x networks
    mut = pd.read_csv(DATA / "mouse_mutation_pattern.csv", index_col=0)
    out["autism_profiles"] = {g: profile(route(mut[g].to_dict(), parcel_acr, pi))[0] for g in mut.columns}

    # transbrain head-to-head numbers
    tb = json.loads((ROOT / "outputs/logs/section6_transbrain_aiopto.json").read_text())
    out["headtohead"] = tb

    out.update(pi_provenance())   # which coupling produced these numbers
    (ROOT / "outputs/logs/fig6_data.json").write_text(json.dumps(out, indent=2))
    print("networks:", len(nets))
    print("AI-opto salience obs:", round(obs, 2), "null p:", (np.sum(np.array(null) >= obs) + 1) / 1001)
    print("autism genes:", list(mut.columns))
    print("wrote fig6_data.json")


if __name__ == "__main__":
    main()
