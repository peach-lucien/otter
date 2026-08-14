"""OTTER as a translator: route a MOUSE anterior-insula optogenetic circuit map into human space and
check it lands on the human salience network (anterior insula / cingulate), a causal-circuit
translation that requires a cross-species coupling.

Mouse input: data_external/transbrain_2025/ai_opto.csv (Allen-acronym -> AI-opto effect; TransBrain's
own case study). Route through pi_canonical via transport-weighted averaging. Validate the translated
human map peaks in salience/insula/cingulate cortex (Yeo-17 SalVentAttn + anatomical insula/ACC), and
beats a permuted-pi null.

Writes outputs/logs/section6_aiopto_translation.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached, load_pi, pi_provenance  # noqa: E402


def route_norm(value_by_acr, parcel_acr, pi):
    mvec = np.array([value_by_acr.get(a, np.nan) for a in parcel_acr])
    mask = np.isfinite(mvec)
    num = mvec[mask] @ pi[mask, :]; den = pi[mask, :].sum(0)
    out = np.full(pi.shape[1], np.nan); ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out, int(mask.sum())


def main():
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    mm = json.loads((ROOT / "data_external/mouse_sc_meta.json").read_text())
    parcel_acr = [mm["structure_acronyms"][i] for i in mm["node_struct_idx"]]
    ai = pd.read_csv(ROOT / "data_external/transbrain_2025/ai_opto.csv", index_col=0)
    value_by_acr = ai.iloc[:, 0].to_dict()
    n_mouse_regions = len(value_by_acr)
    overlap = len(set(value_by_acr) & set(parcel_acr))
    print(f"AI-opto: {n_mouse_regions} mouse regions; {overlap} overlap OTTER parcel acronyms")
    print("strongest mouse input regions:", ", ".join(
        f"{k}={v:.2f}" for k, v in sorted(value_by_acr.items(), key=lambda kv: -kv[1])[:6]))

    pi = load_pi()
    hum, nvis = route_norm(value_by_acr, parcel_acr, pi)

    # human anatomical region + Yeo-17 network per parcel
    nr = np.asarray(json.loads((ROOT / "data_external/human_sc_meta.json").read_text())["node_region"], int)
    rows = [l.split("\t") for l in (ROOT / "outputs/anndata/_schaefer_order.txt").read_text().splitlines() if l.strip()]
    net = np.array([{int(p[0]): p[1].split("_", 2)[2].split("_")[0] for p in rows}.get(int(k), "?") for k in nr])
    region = H.var["region"].astype(str).to_numpy()

    m = np.isfinite(hum)
    z = (hum - np.nanmean(hum[m])) / np.nanstd(hum[m])

    # rank Yeo-17 networks by mean translated value
    out = {"n_visible_mouse_parcels": nvis, "overlap_acr": overlap}
    netmeans = {}
    for u in sorted(set(net[m])):
        sel = m & (net == u)
        if sel.sum() >= 10: netmeans[u] = float(np.nanmean(z[sel]))
    out["network_means_z"] = netmeans
    print("\nTranslated AI-opto by Yeo-17 network (z, sorted):")
    for u, v in sorted(netmeans.items(), key=lambda kv: -kv[1]):
        tag = "  <- salience" if u.startswith("SalVentAttn") else ("  <- limbic/insula" if u == "Limbic" else "")
        print(f"  {u:14s} {v:+.2f}{tag}")

    # top human anatomical regions
    reg_means = {}
    for rg in set(region[m]):
        sel = m & (region == rg)
        if sel.sum() >= 3: reg_means[rg] = float(np.nanmean(z[sel]))
    top = sorted(reg_means.items(), key=lambda kv: -kv[1])[:8]
    out["top_human_regions"] = top
    print("\nTop human anatomical regions (translated AI-opto):")
    for rg, v in top:
        print(f"  {v:+.2f}  {rg}")

    # permuted-pi null: does the real coupling land on salience more than a shuffled one?
    rng = np.random.default_rng(0)
    sal = m & (np.char.startswith(net.astype(str), "SalVentAttn"))
    obs = float(np.nanmean(z[sal]) - np.nanmean(z[m & ~sal]))
    null = []
    for _ in range(1000):
        pp = pi[rng.permutation(pi.shape[0])]
        hh, _ = route_norm(value_by_acr, parcel_acr, pp)
        mm2 = np.isfinite(hh); zz = (hh - np.nanmean(hh[mm2])) / np.nanstd(hh[mm2])
        null.append(np.nanmean(zz[sal & mm2]) - np.nanmean(zz[m & ~sal & mm2]))
    null = np.array(null); p = float((np.sum(null >= obs) + 1) / (len(null) + 1))
    out["salience_enrichment_z"] = obs; out["salience_perm_p"] = p
    print(f"\nSalience-network enrichment of translated AI-opto: {obs:+.2f} SD, permuted-pi p={p:.3f}")
    out.update(pi_provenance())   # which coupling produced these numbers
    (ROOT / "outputs/logs/section6_aiopto_translation.json").write_text(json.dumps(out, indent=2))
    print("wrote section6_aiopto_translation.json")


if __name__ == "__main__":
    main()
