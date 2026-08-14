"""OTTER x TransBrain, head-to-head on the mouse anterior-insula optogenetic circuit.

Both methods translate the SAME mouse phenotype (data_external/transbrain_2025/ai_opto.csv,
TransBrain's own AI-opto case study; Allen acronym -> effect size) into human space, and
each translated human map is scored with the identical salience-vs-rest enrichment metric
used by experiments/section5_coverage_rigor/32_translate_aiopto.py.

Apples-to-apples: both methods produce a value per OTTER human parcel.
  * OTTER: transport-weighted average of the mouse map through pi_canonical.
  * TransBrain: SpeciesTrans(bn).mouse_to_human(...) -> human Brainnetome region values,
    broadcast onto each OTTER human parcel via that parcel's BN label.
Then, on the shared support (parcels with a BN label, a Yeo-17 network and a finite
value from each method), the map is z-scored across parcels and used to compute
    salience_enrichment = mean(z[SalVentAttn parcels]) - mean(z[rest]).
Salience = Yeo-17 SalVentAttnA/B, the same mask OTTER uses.

Specificity null: for each method, which mouse region carries which value is permuted
(shuffling the mouse->value assignment) and the enrichment recomputed. This is the
analogue of OTTER's permuted-coupling null for a fixed-mapping method like TransBrain.

Writes outputs/logs/section6_transbrain_aiopto.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached, load_pi, pi_provenance  # noqa: E402
import nibabel as nib                                              # noqa: E402
from transbrain.config import Config                               # noqa: E402
from transbrain.trans import SpeciesTrans                          # noqa: E402

N_PERM = 1000
SEED = 0


# ---------------------------------------------------------------------------
# BN label per OTTER human parcel (same routine as 01_transbrain_benchmark.py)
# ---------------------------------------------------------------------------
def load_bn_atlas(H_var):
    nii = nib.load(Config.bnatlas_path)
    vol = np.asarray(nii.dataobj).astype(int)
    aff = nii.affine
    inv = np.linalg.inv(aff)
    lab = pd.read_csv(Config.bnatlas_label_path, index_col=0)
    id2name = dict(zip(lab["Atlas Index"].astype(int), lab["Anatomical Name"]))
    xyz = H_var[["x", "y", "z"]].to_numpy(float)
    vox = np.round((inv[:3, :3] @ xyz.T).T + inv[:3, 3]).astype(int)
    shp = np.array(vol.shape)
    bn_id = np.zeros(len(xyz), int)
    for k in range(len(xyz)):
        v = vox[k]
        if not ((v >= 0).all() and (v < shp).all()):
            continue
        lbl = vol[v[0], v[1], v[2]]
        if lbl == 0:
            sl = tuple(slice(max(0, v[d] - 1), min(shp[d], v[d] + 2)) for d in range(3))
            nb = vol[sl][vol[sl] > 0]
            if nb.size:
                lbl = int(np.bincount(nb).argmax())
        bn_id[k] = lbl
    return bn_id, id2name


def route_norm(value_by_acr, parcel_acr, pi):
    """OTTER: transport-weighted average of a mouse acronym->value map through pi."""
    mvec = np.array([value_by_acr.get(a, np.nan) for a in parcel_acr])
    mask = np.isfinite(mvec)
    num = mvec[mask] @ pi[mask, :]
    den = pi[mask, :].sum(0)
    out = np.full(pi.shape[1], np.nan)
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out


def transbrain_map(value_by_acr, st_all):
    """TransBrain: mouse map -> {BN region name: value} using region_type='all'."""
    df = pd.DataFrame({"v": value_by_acr}).dropna()
    return st_all.mouse_to_human(df, region_type="all").iloc[:, 0].to_dict()


def tb_to_parcels(tb_dict, bn_id, id2name):
    """Broadcast a {BN name: value} dict onto OTTER human parcels via their BN label."""
    out = np.full(len(bn_id), np.nan)
    for i, rid in enumerate(bn_id):
        if rid and rid in id2name:
            v = tb_dict.get(id2name[rid], np.nan)
            out[i] = v
    return out


def enrichment(vals, sal, support):
    """z-score `vals` over `support` parcels; salience mean minus rest mean (SD units)."""
    m = support & np.isfinite(vals)
    z = np.full(len(vals), np.nan)
    z[m] = (vals[m] - vals[m].mean()) / vals[m].std()
    s = m & sal
    r = m & ~sal
    return float(z[s].mean() - z[r].mean()), int(s.sum()), int(r.sum())


def main():
    # ---- inputs ----------------------------------------------------------
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    mm = json.loads((ROOT / "data_external/mouse_sc_meta.json").read_text())
    parcel_acr = [mm["structure_acronyms"][i] for i in mm["node_struct_idx"]]
    ai = pd.read_csv(ROOT / "data_external/transbrain_2025/ai_opto.csv", index_col=0)
    value_by_acr = ai.iloc[:, 0].to_dict()
    pi = load_pi()

    bn_id, id2name = load_bn_atlas(H.var)
    st_all = SpeciesTrans(atlas_type="bn")

    # ---- Yeo-17 network per OTTER human parcel (same source as exp 32) ----
    nr = np.asarray(json.loads((ROOT / "data_external/human_sc_meta.json").read_text())["node_region"], int)
    rows = [l.split("\t") for l in (ROOT / "outputs/anndata/_schaefer_order.txt").read_text().splitlines() if l.strip()]
    sch2net = {int(p[0]): p[1].split("_", 2)[2].split("_")[0] for p in rows}
    net = np.array([sch2net.get(int(k), "?") for k in nr])
    sal = np.char.startswith(net.astype(str), "SalVentAttn")
    has_net = net != "?"
    has_bn = bn_id > 0

    # ---- translate with both methods ------------------------------------
    otter_vals = route_norm(value_by_acr, parcel_acr, pi)
    tb_dict = transbrain_map(value_by_acr, st_all)
    tb_vals = tb_to_parcels(tb_dict, bn_id, id2name)

    # SHARED support: parcel needs a Yeo net, a BN label, and a finite value from BOTH
    support = has_net & has_bn & np.isfinite(otter_vals) & np.isfinite(tb_vals)
    print(f"shared support parcels: {int(support.sum())} "
          f"(salience {int((support & sal).sum())}, rest {int((support & ~sal).sum())})")

    h_enr, ns, nr_ = enrichment(otter_vals, sal, support)
    t_enr, _, _ = enrichment(tb_vals, sal, support)
    print(f"\nSalience-vs-rest enrichment on shared support:")
    print(f"  OTTER      {h_enr:+.3f} SD")
    print(f"  TransBrain {t_enr:+.3f} SD")

    # ---- per-method Yeo-17 network ranking (z over shared support) -------
    def net_rank(vals):
        m = support & np.isfinite(vals)
        z = (vals[m] - vals[m].mean()) / vals[m].std()
        zf = np.full(len(vals), np.nan); zf[m] = z
        d = {}
        for u in sorted(set(net[m])):
            sel = m & (net == u)
            if sel.sum() >= 10:
                d[u] = float(np.nanmean(zf[sel]))
        return dict(sorted(d.items(), key=lambda kv: -kv[1]))

    otter_nets = net_rank(otter_vals)
    tb_nets = net_rank(tb_vals)
    print("\nYeo-17 network ranking (z, shared support):")
    print(f"{'network':16s} {'OTTER':>8s} {'TransBrain':>11s}")
    for u in sorted(set(otter_nets) | set(tb_nets),
                    key=lambda k: -(tb_nets.get(k, -9))):
        tag = " <-SAL" if u.startswith("SalVentAttn") else ""
        print(f"  {u:14s} {otter_nets.get(u, float('nan')):+8.2f} {tb_nets.get(u, float('nan')):+11.2f}{tag}")

    # rank of the salience networks within each method (1 = highest-z network)
    def sal_ranks(nets):
        order = list(nets.keys())
        return {u: order.index(u) + 1 for u in order if u.startswith("SalVentAttn")}, len(order)
    h_sr, h_nn = sal_ranks(otter_nets)
    t_sr, t_nn = sal_ranks(tb_nets)
    print(f"\nSalience-network rank (of {t_nn} networks): "
          f"OTTER {h_sr}  |  TransBrain {t_sr}")

    # ---- TransBrain: where do the human insula BN regions rank? ----------
    tb_sorted = sorted(tb_dict.items(), key=lambda kv: -kv[1])
    ins_kw = ("Ia", "Ig", "Id")  # BN insula subregion suffixes: agranular/granular/dysgranular
    tb_ins = [(i + 1, name, val) for i, (name, val) in enumerate(tb_sorted)
              if any(k in name for k in ins_kw)]
    print(f"\nTransBrain human BN insula subregions (rank of {len(tb_sorted)} BN regions):")
    for rank, name, val in tb_ins:
        print(f"  #{rank:3d}  {name:22s} {val:+.3f}")

    # ---- specificity null: permute mouse->value assignment, re-translate --
    rng = np.random.default_rng(SEED)
    acrs = list(value_by_acr.keys())
    vals_arr = np.array([value_by_acr[a] for a in acrs])

    # OTTER null (permute mouse assignment; cheap)
    h_null = []
    for _ in range(N_PERM):
        perm = {a: v for a, v in zip(acrs, rng.permutation(vals_arr))}
        hv = route_norm(perm, parcel_acr, pi)
        e, _, _ = enrichment(hv, sal, support)
        h_null.append(e)
    h_null = np.array(h_null)
    h_p = float((np.sum(h_null >= h_enr) + 1) / (len(h_null) + 1))

    # TransBrain null (permute mouse assignment; re-translate).
    # Matched to OTTER's N_PERM. An unequal number of permutations would make the two
    # p-values non-comparable, and the head-to-head comparison requires both nulls to be
    # estimated at the same resolution.
    N_TB = N_PERM
    rng2 = np.random.default_rng(SEED)
    t_null = []

    # Translating the identity matrix once yields TransBrain's mapping operator, turning each
    # permutation into a matrix-vector product instead of a full re-translation. Without it a
    # matched 1000-permutation null takes ~30 min.
    #
    # mouse_to_human is NOT strictly linear: OP @ v reproduces the direct call with
    # correlation 1.000000 but a different scale and offset (direct = a*(OP@v) + b, a > 0).
    # That affine difference is irrelevant here because enrichment() z-scores its input
    # before differencing group means, so any positive affine transform cancels exactly.
    # The assertion below therefore checks the enrichment statistic, the quantity used,
    # rather than the raw translated values.
    acr_used = list(pd.DataFrame({"v": value_by_acr}).dropna().index)
    assert len(acr_used) == len(acrs), "NaN mouse values would change the index set per permutation"
    eye = pd.DataFrame(np.eye(len(acr_used)), index=acr_used, columns=acr_used)
    OP = st_all.mouse_to_human(eye, region_type="all")
    op_names = list(OP.index)
    OPv = OP[acr_used].to_numpy(float)                      # (n_human_regions, n_mouse_acrs)
    v0 = np.array([value_by_acr[a] for a in acr_used], float)
    _e_op, _, _ = enrichment(tb_to_parcels(dict(zip(op_names, OPv @ v0)), bn_id, id2name),
                             sal, support)
    _de = abs(_e_op - t_enr)
    assert _de < 1e-9, (f"operator route gives a different enrichment than the direct call "
                        f"({_e_op:.6f} vs {t_enr:.6f})")
    print(f"  [tb-null] operator verified: enrichment identical to direct call (Δ {_de:.1e})")

    for _ in range(N_TB):
        pmap = dict(zip(acrs, rng2.permutation(vals_arr)))
        pv = np.array([pmap[a] for a in acr_used], float)
        td = dict(zip(op_names, OPv @ pv))
        tv = tb_to_parcels(td, bn_id, id2name)
        e, _, _ = enrichment(tv, sal, support)
        t_null.append(e)
    t_null = np.array(t_null)
    t_p = float((np.sum(t_null >= t_enr) + 1) / (len(t_null) + 1))
    print(f"\nSpecificity (shuffled mouse->value null):")
    print(f"  OTTER      obs {h_enr:+.3f} vs null {h_null.mean():+.3f}+/-{h_null.std():.3f}  p={h_p:.3f}  (n={N_PERM})")
    print(f"  TransBrain obs {t_enr:+.3f} vs null {t_null.mean():+.3f}+/-{t_null.std():.3f}  p={t_p:.3f}  (n={N_TB})")

    # ---- verdict ---------------------------------------------------------
    diff = h_enr - t_enr
    if abs(diff) < 0.15:
        verdict = "comparable"
    elif diff > 0:
        verdict = "OTTER stronger"
    else:
        verdict = "TransBrain stronger"
    print(f"\nVERDICT: {verdict} (OTTER {h_enr:+.3f} vs TransBrain {t_enr:+.3f} SD, diff {diff:+.3f})")

    out = {
        "input": "data_external/transbrain_2025/ai_opto.csv",
        "pi_file": "outputs/coupling/pi_canonical.npy",
        "n_mouse_regions": len(value_by_acr),
        "shared_support_parcels": int(support.sum()),
        "n_salience_parcels": ns,
        "n_rest_parcels": nr_,
        "salience_enrichment_z": {"otter": h_enr, "transbrain": t_enr, "diff_otter_minus_tb": diff},
        "salience_network_rank": {"otter": h_sr, "transbrain": t_sr, "n_networks": t_nn},
        "yeo17_network_means_z": {"otter": otter_nets, "transbrain": tb_nets},
        "transbrain_insula_bn_ranks": [
            {"rank": r, "bn_region": n, "value": v} for r, n, v in tb_ins],
        "transbrain_top_bn_regions": [
            {"rank": i + 1, "bn_region": n, "value": v}
            for i, (n, v) in enumerate(tb_sorted[:15])],
        "specificity": {
            "otter": {"obs": h_enr, "null_mean": float(h_null.mean()),
                      "null_std": float(h_null.std()), "p": h_p, "n_perm": N_PERM},
            "transbrain": {"obs": t_enr, "null_mean": float(t_null.mean()),
                           "null_std": float(t_null.std()), "p": t_p, "n_perm": N_TB},
        },
        "verdict": verdict,
    }
    outp = ROOT / "outputs/logs/section6_transbrain_aiopto.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    out.update(pi_provenance())   # which coupling produced these numbers
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
