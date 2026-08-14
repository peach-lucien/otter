"""Spin-test every Yeo-17 network on reconstruction accuracy.

Run: cd otter && PYTHONPATH=src python experiments/section5_coverage_rigor/network_sweep_recon.py

11_dlpfc_deficit.py runs this sweep on log column mass, the retired coverage metric. In that log
the only spin test of Control B on reconstruction accuracy is a single entry in the
molecular_control block, computed on the 1,040 parcels that also carry gene expression.

This runs the same sweep with the same machinery, block_gap and spin_perms taken from
11_dlpfc_deficit.py, on reconstruction accuracy over the 1,824 cortical parcels, and again on the
1,040-parcel subset so the two denominators can be compared.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

# experiments/section5_coverage_rigor/ -> parents[2] is the package dir, as elsewhere in the repo.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

PI = "outputs/coupling/pi_canonical.npy"

from otter.data import load_cached                     # noqa: E402
from otter.eval.nulls import _haar_rotation            # noqa: E402

N_SPIN, SEED, MIN_N = 2000, 0, 30


def spin_perms(coords, n=N_SPIN, seed=SEED):
    c = coords - coords.mean(0)
    sph = c / np.linalg.norm(c, axis=1, keepdims=True)
    tree = cKDTree(sph)
    rng = np.random.default_rng(seed)
    return [tree.query(sph @ _haar_rotation(rng).T)[1] for _ in range(n)]


def block_gap(sig, sel, perms):
    f = lambda s: s[sel].mean() - s[~sel].mean()          # noqa: E731
    obs = f(sig)
    null = np.abs([f(sig[p]) for p in perms])
    return {"gap_sd": float(obs),
            "spin_p": float((np.sum(null >= abs(obs)) + 1) / (len(perms) + 1)),
            "null_abs_p95": float(np.percentile(null, 95))}


def main() -> int:
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    Mfc = np.asarray(M.uns["fc_mean"], float)
    Hfc = np.asarray(H.uns["fc_mean"], float)
    xyz = H.var[["x", "y", "z"]].to_numpy(float)
    pi = np.load(ROOT / PI)
    sha = hashlib.sha256((ROOT / PI).read_bytes()).hexdigest()
    pit = pi / np.maximum(pi.sum(0), 1e-300)
    pred = pit.T @ Mfc @ pit
    rc = np.full(pred.shape[0], np.nan)
    for j in range(pred.shape[0]):
        a, b = pred[j].copy(), Hfc[j].copy()
        a[j] = np.nan; b[j] = np.nan
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 10 and a[ok].std() > 1e-9:
            rc[j] = np.corrcoef(a[ok], b[ok])[0, 1]

    mye = np.asarray(json.loads(
        (ROOT / "outputs/logs/buckner_krienen_2013_tethering.json").read_text())["myelin_per_parcel"], float)
    nr = np.asarray(json.loads((ROOT / "data_external/human_sc_meta.json").read_text())["node_region"], int)
    rows = [l.split("\t") for l in (ROOT / "outputs/anndata/_schaefer_order.txt").read_text().splitlines() if l.strip()]
    lut = {int(p[0]): p[1].split("_", 2)[2].split("_")[0] for p in rows}
    net = np.array([lut.get(int(k), "?") for k in nr])

    # the gene panel that defines the 1,040-parcel subset
    hg = np.load(ROOT / "data_external/human_genes_aligned.npy").astype(float)
    has_genes = np.isfinite(hg).all(1)

    out = {"pi_file": PI, "pi_sha256": sha,
           "n_spin": N_SPIN, "min_parcels_per_network": MIN_N, "arms": {}}

    for arm, mask in (("all_cortical", np.isfinite(rc) & np.isfinite(mye)),
                      ("gene_supported", np.isfinite(rc) & np.isfinite(mye) & has_genes)):
        z = (rc[mask] - rc[mask].mean()) / rc[mask].std()
        nt = net[mask]
        perms = spin_perms(xyz[mask])
        nets = sorted({u for u in set(nt) if (nt == u).sum() >= MIN_N})
        res = {}
        for u in nets:
            r = block_gap(z, nt == u, perms)
            r["mean_sd"] = float(z[nt == u].mean())
            r["n"] = int((nt == u).sum())
            r["bonferroni_p"] = min(1.0, r["spin_p"] * len(nets))
            res[u] = r
        out["arms"][arm] = {"n_parcels": int(mask.sum()), "n_networks": len(nets), "networks": res}
        print(f"\n=== {arm}, {mask.sum()} parcels, {len(nets)} networks, "
              f"Bonferroni over {len(nets)} ===")
        for u, r in sorted(res.items(), key=lambda kv: kv[1]["gap_sd"]):
            star = "  *" if r["bonferroni_p"] < 0.05 else ("  (uncorrected)" if r["spin_p"] < 0.05 else "")
            print(f"  {u:14s} n={r['n']:4d}  gap {r['gap_sd']:+.2f} SD  spin p={r['spin_p']:.4f}  "
                  f"Bonf p={r['bonferroni_p']:.4f}{star}")

    (ROOT / "outputs/logs/network_sweep_recon.json").write_text(json.dumps(out, indent=1))
    print("\nwrote network_sweep_recon.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
