"""The salience result under a spatial null, and under a change in coupling softness.

Run: cd otter && PYTHONPATH=src python experiments/section5_coverage_rigor/aiopto_spin_null.py

33_translate_circuits.py tests the salience enrichment against a permuted-pi null, which shuffles
which mouse parcel each row of pi belongs to. That destroys the whole cross-species assignment and
does not preserve mouse spatial autocorrelation, so it is close to a null of zero. Section 5 of the
manuscript argues at length that such a null is too liberal for spatially structured data.

This adds the translation null the paper uses elsewhere: rotate the mouse input map on the mouse
brain and route the rotated map through the real coupling. Mouse spatial autocorrelation is
preserved, the coupling is untouched, and only the anatomy of the input is destroyed.

It also reports what softening the coupling does, so the claim that the result is independent of
softness can be stated with the sharpness numbers attached.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# experiments/section5_coverage_rigor/ -> parents[2] is the package dir, as elsewhere in the repo.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

PI = {"canonical (eps 0.05)": "outputs/coupling/pi_canonical.npy",
      "sharp (eps 0.005)": "outputs/coupling/pi_canonical_sharp.npy"}

from otter.data import load_cached                     # noqa: E402
from otter.eval.nulls import _haar_rotation            # noqa: E402

N_ROT = 1000


def route(value_by_acr, parcel_acr, pi):
    mvec = np.array([value_by_acr.get(a, np.nan) for a in parcel_acr])
    return route_vec(mvec, pi)


def route_vec(mvec, pi):
    mask = np.isfinite(mvec)
    num = mvec[mask] @ pi[mask, :]
    den = pi[mask, :].sum(0)
    out = np.full(pi.shape[1], np.nan)
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out


def main() -> int:
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    mxyz = M.var[["x", "y", "z"]].to_numpy(float)
    mm = json.loads((ROOT / "data_external/mouse_sc_meta.json").read_text())
    parcel_acr = np.array([mm["structure_acronyms"][i] for i in mm["node_struct_idx"]])
    ai = pd.read_csv(ROOT / "data_external/transbrain_2025/ai_opto.csv",
                     index_col=0).iloc[:, 0].to_dict()

    nr = np.asarray(json.loads((ROOT / "data_external/human_sc_meta.json").read_text())["node_region"], int)
    rows = [l.split("\t") for l in (ROOT / "outputs/anndata/_schaefer_order.txt").read_text().splitlines() if l.strip()]
    lut = {int(p[0]): p[1].split("_", 2)[2].split("_")[0] for p in rows}
    net = np.array([lut.get(int(k), "?") for k in nr])
    sal = np.char.startswith(net.astype(str), "SalVentAttn")
    cortex = net != "?"

    def enrich(hum, baseline):
        m = np.isfinite(hum)
        z = (hum - np.nanmean(hum[m])) / np.nanstd(hum[m])
        a = m & sal
        b = m & baseline & ~sal
        return float(np.nanmean(z[a]) - np.nanmean(z[b]))

    couplings = {k: np.load(ROOT / v) for k, v in PI.items()}

    out = {"n_rotations": N_ROT, "arms": {}, "pi_files": PI}
    for name, pi in couplings.items():
        sha = hashlib.sha256((ROOT / PI[name]).read_bytes()).hexdigest()
        r = pi / pi.sum(1, keepdims=True)
        sharp_top1 = float(np.median(r.max(1)))
        eff = float(np.mean(1 / np.sum(r ** 2, 1)))
        hum = route(ai, parcel_acr, pi)
        obs_all = enrich(hum, np.ones(len(net), bool))
        obs_ctx = enrich(hum, cortex)
        print(f"\n{name}   median top-1 row probability {sharp_top1:.3f}, "
              f"mean effective targets {eff:.1f}")
        print(f"  salience enrichment against the whole parcellation {obs_all:+.3f} SD")
        print(f"  salience enrichment against cortex only            {obs_ctx:+.3f} SD")

        # translation null: rotate the mouse input, route through the real coupling
        mvec = np.array([ai.get(a, np.nan) for a in parcel_acr])
        c = mxyz - mxyz.mean(0)
        sph = c / np.linalg.norm(c, axis=1, keepdims=True)
        tree = cKDTree(sph)
        rng = np.random.default_rng(0)
        null = []
        for _ in range(N_ROT):
            idx = tree.query(sph @ _haar_rotation(rng).T)[1]
            null.append(enrich(route_vec(mvec[idx], pi), np.ones(len(net), bool)))
        null = np.asarray(null)
        p = float((np.sum(null >= obs_all) + 1) / (N_ROT + 1))
        print(f"  rotated-input null mean {null.mean():+.3f} SD, 95th pct {np.percentile(null, 95):+.3f}"
              f"   p = {p:.4f}")
        out["arms"][name] = {"pi_sha256": sha,
                             "median_top1_row_probability": round(sharp_top1, 4),
                             "mean_effective_targets": round(eff, 2),
                             "salience_enrichment_vs_all_parcels": round(obs_all, 4),
                             "salience_enrichment_vs_cortex_only": round(obs_ctx, 4),
                             "rotated_input_null_mean": round(float(null.mean()), 4),
                             "rotated_input_null_p95": round(float(np.percentile(null, 95)), 4),
                             "rotated_input_p": round(p, 4)}

    a = couplings["canonical (eps 0.05)"].argmax(1)
    b = couplings["sharp (eps 0.005)"].argmax(1)
    out["argmax_agreement_between_couplings"] = round(float((a == b).mean()), 4)
    print(f"\nthe two couplings choose the same top human parcel for "
          f"{out['argmax_agreement_between_couplings'] * 100:.1f} per cent of mouse parcels")

    (ROOT / "outputs/logs/aiopto_spin_null.json").write_text(json.dumps(out, indent=1))
    print("wrote aiopto_spin_null.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
