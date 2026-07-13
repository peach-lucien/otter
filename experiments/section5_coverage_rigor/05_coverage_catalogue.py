"""§5: comprehensive, MASS-NORMALISED catalogue of unmapped human territory.

The `uncovered_by_region.json` table ranks regions by the FRACTION of their parcels
below a coverage threshold. That metric is confounded by parcel count / relative size:
finely-parcellated, expanded-but-homologous regions (e.g. human V1) score as highly
'uncovered' even though the mouse maps to them correctly, just diluted across many
parcels. Here we rank the SAME 21 macro-regions (each human parcel assigned to its
nearest Garin homology-class anchor by normalised position — the atlas behind
uncovered_by_region) by MASS-NORMALISED coverage = mean mouse π-mass per parcel,
which is not parcel-count-biased. Both metrics are reported side by side so the
V1-type artefact is visible.

Run: cd homer && PYTHONPATH=src python experiments/section5_coverage_rigor/05_coverage_catalogue.py
Writes outputs/logs/section5_coverage_catalogue.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from homer.data import load_cached, load_pi
from homer.data.anchors import get_anchor_index


def nearest_garin_pid(var):
    """Assign each parcel to its nearest Garin anchor pair_id by normalised xyz
    (identical logic to experiments/autism_subtypes/07 assign_*_networks)."""
    idx = get_anchor_index(var)
    coords = var[["x", "y", "z"]].to_numpy(float)
    lo = coords.min(0, keepdims=True); hi = coords.max(0, keepdims=True)
    cn = (coords - lo) / np.maximum(hi - lo, 1e-9)
    axyz = cn[idx.pos]
    d2 = (cn ** 2).sum(1, keepdims=True) + (axyz ** 2).sum(1) - 2 * cn @ axyz.T
    pid = np.array([int(idx.pair_ids[k]) for k in d2.argmin(1)])
    pid[idx.pos] = idx.pair_ids.astype(int)              # anchors keep own pid
    return pid


def main():
    pi = load_pi()
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    col = pi.sum(0); cmax = col.max()
    pid = nearest_garin_pid(H.var)

    # pid -> human-readable name from the existing uncovered_by_region table
    unc = {r["pid"]: r["region"] for r in json.loads((ROOT / "outputs/logs/uncovered_by_region.json").read_text())}

    rows = []
    for p in sorted(set(pid.tolist())):
        idx = np.where(pid == p)[0]; n = len(idx)
        mean_cov = float(col[idx].mean())
        rows.append({"pid": int(p), "region": unc.get(p, f"pid{p}"), "n_parcels": int(n),
                     "mean_coverage": mean_cov, "log10_mean_coverage": float(np.log10(mean_cov + 1e-300)),
                     "total_mass": float(col[idx].sum()),
                     "frac_uncovered": float((col[idx] < 1e-4 * cmax).mean())})

    mc = np.array([r["log10_mean_coverage"] for r in rows]); lo, hi = mc.min(), mc.max()
    for r in rows:
        r["coverage_score"] = float((r["log10_mean_coverage"] - lo) / (hi - lo))     # 0=least,1=most covered
    rows.sort(key=lambda r: r["log10_mean_coverage"])                                  # least covered first

    print(f"{'region':40s} {'n':>4} {'log10 meanCov':>13} {'covScore':>8} {'fracUnc(old)':>12}")
    for r in rows:
        print(f"{r['region'][:40]:40s} {r['n_parcels']:4d} {r['log10_mean_coverage']:13.2f} "
              f"{r['coverage_score']:8.2f} {r['frac_uncovered']:12.2f}")

    (ROOT / "outputs/logs/section5_coverage_catalogue.json").write_text(
        json.dumps({"metric": "mass-normalised mean coverage per Garin macro-region",
                    "n_regions": len(rows), "regions": rows}, indent=2))
    print("\nwrote outputs/logs/section5_coverage_catalogue.json")


if __name__ == "__main__":
    main()
