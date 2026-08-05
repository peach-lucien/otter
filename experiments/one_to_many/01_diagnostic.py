#!/usr/bin/env python3
"""One-to-many correspondence diagnostic (v2) — meaningful, artefact-guarded.

QUESTION
--------
Which mouse structures correspond to >=2 GENUINELY DISTINCT human structures
(one-to-many)? The textbook case is mouse caudoputamen (CP) -> human caudate AND
putamen. If OTTER recovers that and surfaces further, non-obvious, connectionally
coherent splits that a single-assignment method cannot represent, it is the
demonstrated payoff of the probabilistic parcel-level coupling.

WHY v1 WAS NOT ENOUGH
---------------------
v1 aggregated to 127 named Brainnetome structures, which fixed the parcel-count
confound but NOT the "the atlas subdivides one continuous system" confound: mouse
primary somatosensory cortex (SSp) mapped across several Brainnetome subfields of the
human S1 body-map strip (A1/2/3ulhf, A1/2/3tru, ...) and was wrongly flagged
one-to-many. Those subfields are one system, not distinct structures.

THE FIX — define "distinct" by CONNECTIVITY, not by atlas labels
----------------------------------------------------------------
Two human structures are genuinely distinct only if they are wired differently. We
therefore weight the count of human targets by how connectionally SIMILAR they are:

    D(structure) = 1 / (q^T S q)

where q is the mouse structure's mass distribution over the 127 human regions
(sum q = 1) and S_ij is the similarity (clipped Pearson r in [0,1]) between the human
FC fingerprints of regions i and j (S_ii = 1). This is a Hill/Rao similarity-sensitive
diversity number. Behaviour:
    * mass on ONE region                          -> D = 1   (one-to-one)
    * mass split 50/50 on two DISTINCT regions    -> D = 2   (S_ij = 0)
    * mass split 50/50 on two SIMILAR regions     -> D = 1   (S_ij = 1)  <- collapses S1
So D is the effective number of connectionally-DISTINCT human targets. One-to-many is
D >= 1.8. This uses human connectivity only to interpret the targets; it does not use
the coupling, so it is not circular.

VALIDATION (hard asserts)
    * CP must have D >= 1.8 with caudate AND putamen among its top targets.
    * VISp (primary visual) must be ~one-to-one (D < 1.6).
    * At least one SSp (somatosensory) case must have its plain effective number
      (1/sum q^2) COLLAPSE under S-weighting (D << effN) — i.e. the metric removes the
      atlas-subdivision artefact rather than us removing it by hand.

CONVERGENCE (reverse view, reported not gated)
    A genuine evolutionary split predicts the two human targets both point BACK to the
    same mouse structure. We report, for each split, whether the top-2 human targets
    name this mouse structure as their dominant mouse origin (column-normalised pi).

Run (needs transbrain for the Brainnetome atlas, as in the Fig. 4 dumps):
    cd otter && PYTHONPATH=src python experiments/one_to_many/01_diagnostic.py
Read-only; writes outputs/logs/one_to_many_diagnostic.json (pi-provenance stamped).
"""
from __future__ import annotations
import json, sys, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached, load_pi, pi_provenance  # noqa: E402

MIN_PARCELS = 10
D_ONE2MANY = 1.8       # effective number of connectionally-distinct targets
SPLIT_MASS = 0.45      # top-2 distinct targets must jointly hold at least this


def _bn_atlas(Hvar):
    p = ROOT / "experiments/transbrain_2025_benchmark/01_transbrain_benchmark.py"
    spec = importlib.util.spec_from_file_location("tb01", p)
    tb01 = importlib.util.module_from_spec(spec); spec.loader.exec_module(tb01)
    return tb01.load_bn_atlas(Hvar)


def _mouse_acr():
    mm = json.loads((ROOT / "data_external/mouse_sc_meta.json").read_text())
    return np.array([mm["structure_acronyms"][i] if i >= 0 else "NA"
                     for i in mm["node_struct_idx"]])


def _anchored():
    try:
        M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
        from otter.data.anchors import get_anchor_index
        acr = _mouse_acr(); out = {acr[int(p)] for p in get_anchor_index(M.var).pos}
        try:
            from otter.data.region_anchors import build_default_pack_entries
            for e in build_default_pack_entries(M.var, None, atlas_root=ROOT):
                for i in getattr(e, "mouse_parcels", []) or []:
                    out.add(acr[int(i)])
        except Exception as exc:
            print(f"[warn] packs unavailable ({exc}); anchored = Garin only")
        return out
    except Exception as exc:
        print(f"[warn] anchor flags unavailable ({exc})"); return None


def region_fc_similarity(Hfc, bn_id, region_ids, ridx):
    """127x127 similarity S = clip(corr of region FC fingerprints, 0, 1), diag 1."""
    n_h = Hfc.shape[0]
    Hn = np.zeros((n_h, len(region_ids)))
    for j, b in enumerate(bn_id):
        b = int(b)
        if b > 0:
            Hn[j, ridx[b]] = 1.0
    sz = Hn.sum(0).clip(1)
    Hmean = Hn / sz                                   # column = mean over region's parcels
    R = Hmean.T @ Hfc @ Hmean                         # region-by-region FC (127x127)
    # fingerprint similarity = correlation of rows of R
    Rc = R - R.mean(1, keepdims=True)
    denom = np.sqrt((Rc ** 2).sum(1))
    S = (Rc @ Rc.T) / np.outer(denom, denom).clip(1e-12)
    S = np.clip(S, 0.0, 1.0)
    np.fill_diagonal(S, 1.0)
    return S


def main():
    prov = pi_provenance(); print(f"pi: {prov['pi_file']} sha256={prov['pi_sha256'][:12]}")
    pi = load_pi()
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    Hfc = np.asarray(H.uns["fc_mean"], float)
    acr = _mouse_acr()
    bn_id, id2name, name2cent = _bn_atlas(H.var)
    n_h = pi.shape[1]

    region_ids = sorted({int(r) for r in np.unique(bn_id) if int(r) > 0})
    ridx = {r: i for i, r in enumerate(region_ids)}
    names = [id2name[r] for r in region_ids]
    Hagg = np.zeros((n_h, len(region_ids)))
    for j, b in enumerate(bn_id):
        if int(b) > 0:
            Hagg[j, ridx[int(b)]] = 1.0
    S = region_fc_similarity(Hfc, bn_id, region_ids, ridx)
    print(f"human structures: {len(region_ids)}; FC-similarity S built")

    # column-normalised pi aggregated to mouse structures, for the convergence check
    anchored = _anchored()
    structs = [s for s in sorted(set(acr.tolist()))
               if s != "NA" and (acr == s).sum() >= MIN_PARCELS]
    pihat = pi / pi.sum(0).clip(1e-12)                # human parcel -> mouse distribution
    # mouse-structure aggregator (rows)
    Sm = {s: np.where(acr == s)[0] for s in structs}

    def mouse_origin(region_pos):
        """dominant mouse structure for a human region (by column-normalised mass)."""
        cols = np.where(Hagg[:, region_pos] > 0)[0]
        v = pihat[:, cols].sum(1)
        best = {s: v[idx].sum() for s, idx in Sm.items()}
        return max(best, key=best.get)

    rows = []
    for s in structs:
        idx = Sm[s]
        qr = Hagg.T @ pi[idx, :].sum(0)
        tot = float(qr.sum())
        if tot <= 0:
            continue
        q = qr / tot
        order = np.argsort(-q)
        effN = float(1.0 / np.sum(q ** 2))            # plain effective number
        D = float(1.0 / (q @ S @ q))                  # connectionally-distinct number
        o1, o2 = order[0], order[1]
        sim12 = float(S[o1, o2])
        split2 = float(q[o1] + q[o2])
        one2many = bool(D >= D_ONE2MANY and split2 >= SPLIT_MASS)
        conv = None
        if one2many:
            conv = [mouse_origin(o1) == s, mouse_origin(o2) == s]
        rows.append(dict(structure=s, n_parcels=int(len(idx)),
                         top=[(names[o], float(q[o])) for o in order[:4]],
                         effN=effN, D=D, top2_similarity=sim12, split2=split2,
                         one_to_many=one2many,
                         top2_converge_back=conv,
                         anchored=(None if anchored is None else s in anchored)))

    by = {r["structure"]: r for r in rows}
    problems = []
    if "CP" in by:
        cp = by["CP"]; tn = " ".join(n for n, _ in cp["top"]).lower()
        if cp["D"] < D_ONE2MANY: problems.append(f"CP D={cp['D']:.2f} < {D_ONE2MANY}")
        if not ("audate" in tn and "utamen" in tn): problems.append(f"CP lacks caudate+putamen: {cp['top']}")
    else:
        problems.append("CP missing")
    if "VISp" in by and by["VISp"]["D"] >= 1.6:
        problems.append(f"VISp D={by['VISp']['D']:.2f} not one-to-one")
    # metric must collapse at least one S1 artefact (effN high, D low)
    s1 = [r for r in rows if r["structure"].startswith("SSp")]
    collapsed = [r for r in s1 if r["effN"] >= 2.5 and r["D"] < D_ONE2MANY]
    if s1 and not collapsed:
        problems.append("no SSp case collapsed under S-weighting — metric may not be removing the atlas artefact")

    o2m = sorted([r for r in rows if r["one_to_many"]], key=lambda r: -r["D"])
    novel = [r for r in o2m if r["anchored"] is False]

    out = {**prov, "params": dict(MIN_PARCELS=MIN_PARCELS, D_ONE2MANY=D_ONE2MANY,
                                  SPLIT_MASS=SPLIT_MASS),
           "validation_problems": problems, "n_structures": len(rows),
           "n_one_to_many": len(o2m), "n_one_to_many_nonanchored": len(novel),
           "structures": rows}
    (ROOT / "outputs/logs/one_to_many_diagnostic.json").write_text(json.dumps(out, indent=1))

    print("\n===============  ONE-TO-MANY (connectionally-distinct)  ===============")
    print(f"scored {len(rows)} structures; one-to-many {len(o2m)} (novel {len(novel)})")
    if "CP" in by:
        cp = by["CP"]
        print(f"VALIDATION CP: D={cp['D']:.2f} (plain effN {cp['effN']:.1f}) "
              f"top-2 sim {cp['top2_similarity']:.2f} -> "
              + ", ".join(f"{n} {v:.2f}" for n, v in cp["top"][:3]))
    if collapsed:
        c = collapsed[0]
        print(f"ARTEFACT CONTROL {c['structure']}: plain effN {c['effN']:.1f} -> "
              f"D {c['D']:.2f} (collapsed; S1 subfields correctly count as ~one system)")
    print("problems:", problems or "none")
    print("\nGENUINE ONE-TO-MANY (by D = distinct targets):")
    for r in o2m[:15]:
        tag = "anchored" if r["anchored"] else ("NOVEL" if r["anchored"] is False else "?")
        cb = "" if r["top2_converge_back"] is None else f" conv{r['top2_converge_back']}"
        print(f"  {r['structure']:9s}[{tag:8s}] D={r['D']:.2f} sim12={r['top2_similarity']:.2f}{cb} -> "
              + ", ".join(f"{n} {v:.2f}" for n, v in r["top"][:3]))
    verdict = ("GO" if (not problems and len(novel) >= 3) else
               "MAYBE" if not problems else "NO-GO (validation failed)")
    print(f"\nVERDICT: {verdict}")
    print("wrote outputs/logs/one_to_many_diagnostic.json")


if __name__ == "__main__":
    main()
