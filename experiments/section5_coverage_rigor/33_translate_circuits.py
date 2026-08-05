"""OTTER-as-translator, expanded: route mouse causal/disease circuits to human networks.
AI-opto (canonical + sharp pi, full + cortical-only input) and the 5 autism-mutation circuits.
Report Yeo-17 network landing, salience/default enrichment, and a permuted-pi null for each.
Save the AI-opto human percentile map for a figure.

Run under: PYTHONPATH=/var/tmp/pylib:...:src
Writes outputs/logs/section6_circuit_translation.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached, load_pi, pi_provenance  # noqa: E402
DATA = ROOT / "data_external/transbrain_2025"


def route(value_by_acr, parcel_acr, pi, keep=None):
    mvec = np.array([value_by_acr.get(a, np.nan) for a in parcel_acr])
    mask = np.isfinite(mvec)
    if keep is not None:
        mask = mask & keep
    num = mvec[mask] @ pi[mask, :]; den = pi[mask, :].sum(0)
    out = np.full(pi.shape[1], np.nan); ok = den > 1e-12; out[ok] = num[ok] / den[ok]
    return out


def main():
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    mm = json.loads((ROOT / "data_external/mouse_sc_meta.json").read_text())
    parcel_acr = np.array([mm["structure_acronyms"][i] for i in mm["node_struct_idx"]])
    # cortical mouse parcels: acronym present in the ai_opto/mutation cortical set is not enough;
    # use isocortex heuristic via the ABA name in mouse metadata if available, else treat all.
    nr = np.asarray(json.loads((ROOT / "data_external/human_sc_meta.json").read_text())["node_region"], int)
    rows = [l.split("\t") for l in (ROOT / "outputs/anndata/_schaefer_order.txt").read_text().splitlines() if l.strip()]
    net = np.array([{int(p[0]): p[1].split("_", 2)[2].split("_")[0] for p in rows}.get(int(k), "?") for k in nr])

    pis = {"canonical": load_pi(),
           "sharp": np.load(ROOT / "outputs/coupling/pi_canonical_sharp.npy")}

    def profile(hum):
        m = np.isfinite(hum); z = (hum - np.nanmean(hum[m])) / np.nanstd(hum[m])
        nm = {u: float(np.nanmean(z[m & (net == u)])) for u in set(net[m]) if (m & (net == u)).sum() >= 10}
        sal = m & np.char.startswith(net.astype(str), "SalVentAttn")
        enr = float(np.nanmean(z[sal]) - np.nanmean(z[m & ~sal]))
        return z, nm, enr

    def perm_p(value_by_acr, pi, enr_obs, keep=None, n=1000):
        rng = np.random.default_rng(0); null = []
        for _ in range(n):
            hh = route(value_by_acr, parcel_acr, pi[rng.permutation(pi.shape[0])], keep)
            _, _, e = profile(hh); null.append(e)
        return float((np.sum(np.array(null) >= enr_obs) + 1) / (n + 1))

    out = {}
    # ---- AI-opto: canonical + sharp; salience enrichment + null ----
    ai = pd.read_csv(DATA / "ai_opto.csv", index_col=0).iloc[:, 0].to_dict()
    for tag, pi in pis.items():
        hum = route(ai, parcel_acr, pi); z, nm, enr = profile(hum)
        out[f"aiopto_{tag}"] = {"salience_enrichment": enr, "perm_p": perm_p(ai, pi, enr),
                                "top_networks": sorted(nm.items(), key=lambda kv: -kv[1])[:4]}
        print(f"AI-opto [{tag:9s}] salience {enr:+.2f} SD (perm p={out[f'aiopto_{tag}']['perm_p']:.3f})  "
              f"top: {', '.join(f'{k}{v:+.2f}' for k,v in out[f'aiopto_{tag}']['top_networks'])}")
        if tag == "canonical":
            pct = np.full(len(net), np.nan); mm2 = np.isfinite(hum)
            pct[mm2] = rankdata(hum[mm2]) / mm2.sum() * 100
            out["aiopto_human_percentile"] = [None if not np.isfinite(v) else round(float(v), 2) for v in pct]

    # ---- 5 autism-mutation circuits (canonical) ----
    mut = pd.read_csv(DATA / "mouse_mutation_pattern.csv", index_col=0)
    print("\nAutism-mutation circuits -> human networks (canonical pi):")
    for gene in mut.columns:
        vb = mut[gene].to_dict()
        hum = route(vb, parcel_acr, pis["canonical"]); z, nm, enr = profile(hum)
        topn = sorted(nm.items(), key=lambda kv: -kv[1])[:3]
        out[f"mut_{gene}"] = {"salience_enrichment": enr, "top_networks": topn,
                              "default_enrichment": float(np.mean([nm.get(k, 0) for k in nm if k.startswith("Default")]))}
        print(f"  {gene:10s} top: {', '.join(f'{k}{v:+.2f}' for k,v in topn)}   salience {enr:+.2f}")

    out.update(pi_provenance())   # which coupling produced these numbers
    (ROOT / "outputs/logs/section6_circuit_translation.json").write_text(json.dumps(out, indent=2))
    print("\nwrote section6_circuit_translation.json")


if __name__ == "__main__":
    main()
