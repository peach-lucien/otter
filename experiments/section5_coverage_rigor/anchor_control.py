"""Is the expansion result explained by where the cross-species supervision sits?

Both the Garin point anchors and the correspondence packs pin particular human parcels to particular
mouse parcels. If those pins are concentrated in sensory cortex, reconstruction would be better there
for a reason that has nothing to do with evolution. The comparison is made at the same regional
resolution as the cortical-map battery.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, rankdata

# experiments/section5_coverage_rigor/ -> parents[2] is the package dir, as elsewhere in the repo.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from otter.data import load_cached                                # noqa: E402
from otter.data.anchors import get_anchor_index                   # noqa: E402
from otter.data.anchor_packs import build_default_pack_entries    # noqa: E402
from battery_assumptions import reconstruction_accuracy, MAPS, pi_stamp   # noqa: E402


def partial_spearman(a, b, c):
    ra, rb, rc_ = rankdata(a), rankdata(b), rankdata(c)
    X = np.c_[np.ones_like(rc_), rc_]
    ea = ra - X @ np.linalg.lstsq(X, ra, rcond=None)[0]
    eb = rb - X @ np.linalg.lstsq(X, rb, rcond=None)[0]
    return float(spearmanr(ea, eb).statistic)


def main() -> int:
    rc, xyz, Hfc, ph, M, H = reconstruction_accuracy()
    ih = get_anchor_index(H.var)
    garin = np.unique(np.asarray(ih.pos, int))
    entries = build_default_pack_entries(M.var, H.var, atlas_root=ROOT)
    pack = set()
    for e in entries:
        for j in np.asarray(e.human_indices, int).ravel():
            pack.add(int(j))
    pack = np.array(sorted(pack), int)
    print(f"{garin.size} human parcels carry a Garin point anchor, "
          f"{pack.size} sit in a correspondence pack")

    d_garin = np.sqrt(((xyz[:, None, :] - xyz[None, garin, :]) ** 2).sum(-1)).min(1)
    supervised = np.zeros(len(xyz), bool)
    supervised[garin] = True
    if pack.size:
        supervised[pack] = True
    d_any = np.sqrt(((xyz[:, None, :] - xyz[None, supervised, :]) ** 2).sum(-1)).min(1)

    ok = np.isfinite(rc)
    for nm, d in (("distance to nearest Garin anchor", d_garin),
                  ("distance to nearest supervised parcel", d_any)):
        r, p = spearmanr(rc[ok], d[ok])
        print(f"parcel level, accuracy vs {nm}: rho = {r:+.3f}  p = {p:.4g}  n = {ok.sum()}")

    nr = np.asarray(json.loads((ROOT / "data_external/human_sc_meta.json").read_text())["node_region"], int)
    batt = json.loads((ROOT / "data_external/published_cortical_maps.json").read_text())["maps"]
    out = {}
    for key, short, _grp in MAPS:
        v = batt.get(key, {})
        mp = dict(zip(np.asarray(v["schaefer_ids"], int), np.asarray(v["map_values"], float)))
        ids = [k for k in range(1, 401) if (nr == k).any() and k in mp]
        cc = np.array([np.nanmean(rc[nr == k]) for k in ids])
        mv = np.array([mp[k] for k in ids])
        dg = np.array([np.nanmean(d_garin[nr == k]) for k in ids])
        da = np.array([np.nanmean(d_any[nr == k]) for k in ids])
        raw = spearmanr(cc, mv).statistic
        out[short] = {"raw": round(float(raw), 3),
                      "partial_garin": round(partial_spearman(cc, mv, dg), 3),
                      "partial_any_supervision": round(partial_spearman(cc, mv, da), 3),
                      "n_regions": len(ids)}
        print(f"  {short:22s} raw {raw:+.3f}  |Garin partialled {out[short]['partial_garin']:+.3f}"
              f"  |all supervision partialled {out[short]['partial_any_supervision']:+.3f}")

    out["_supervision_vs_accuracy"] = {
        "garin_rho": round(float(spearmanr(rc[ok], d_garin[ok]).statistic), 3),
        "any_rho": round(float(spearmanr(rc[ok], d_any[ok]).statistic), 3),
        "n_garin_parcels": int(garin.size), "n_pack_parcels": int(pack.size)}
    out.update(pi_stamp())
    (ROOT / "outputs/logs/anchor_control.json").write_text(json.dumps(out, indent=1))
    print("wrote anchor_control.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
