#!/usr/bin/env python3
"""Anchor-drop control for the caudoputamen -> caudate + putamen split.

The striatum pack (Voorn 2004) directly supervises the split: it maps
dorsolateral caudoputamen -> putamen and ventromedial caudoputamen -> caudate.
So on the canonical coupling the CP split could be handed to the model, not
discovered. This control refits the canonical coupling with EVERY default pack
EXCEPT striatum (the anchor-warped spatial term, the Garin point anchors, and all
other packs are kept), then asks whether CP still splits into caudate + putamen.

  * split SURVIVES  -> the caudate/putamen correspondence is recovered from
                       connectivity + space, not merely from the striatum anchor.
  * split COLLAPSES -> it was supervised in; the CP result cannot be presented as
                       a discovered one-to-many.

The Garin point anchor for CP is retained (it only says CP <-> striatum, not the
sub-split), so this isolates the effect of the striatum SUBDIVISION supervision.

Run: cd otter && PYTHONPATH=src python experiments/one_to_many/02_anchor_drop_control.py
Writes /var/tmp/pi_canonical_nostriatum.npy (scratch) and prints PASS/FAIL. Does not
touch outputs/coupling.
"""
from __future__ import annotations
import sys, importlib.util
from pathlib import Path
import numpy as np
from scipy.interpolate import RBFInterpolator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached                                    # noqa: E402
from otter.data.anchors import get_anchor_index                      # noqa: E402
from otter.models.multimodal import MultimodalFGW                    # noqa: E402
from otter.data.anchor_packs.registry import PACKS                   # noqa: E402

# reuse the exact metric helpers from the diagnostic
_spec = importlib.util.spec_from_file_location("o2m", ROOT / "experiments/one_to_many/01_diagnostic.py")
o2m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(o2m)

EPS, XYZW = 0.05, 0.25       # deployed canonical hyperparameters


def warped_xyz_M(M, H):
    im = get_anchor_index(M.var); ih = get_anchor_index(H.var)
    hlut = {(int(p), str(h)): int(k) for k, p, h in zip(ih.pos, ih.pair_ids, ih.hemispheres)}
    trip = [(int(mp), hlut[(int(pid), str(hemi))])
            for mp, pid, hemi in zip(im.pos, im.pair_ids, im.hemispheres)
            if (int(pid), str(hemi)) in hlut]
    mxyz = M.var[["x", "y", "z"]].to_numpy(float); hxyz = H.var[["x", "y", "z"]].to_numpy(float)
    warp = RBFInterpolator(mxyz[[a for a, b in trip]], hxyz[[b for a, b in trip]],
                           kernel="thin_plate_spline", smoothing=1e-3)
    d = np.sqrt(((warp(mxyz)[:, None, :] - hxyz[None, :, :]) ** 2).sum(-1))
    return (d / max(d.max(), 1e-9)).astype(np.float64)


def cp_split(pi, M, H):
    """Return CP's (D, top-3 targets) on a given coupling, using the v1 metric."""
    acr = o2m._mouse_acr()
    bn_id, id2name, _ = o2m._bn_atlas(H.var)
    region_ids = sorted({int(r) for r in np.unique(bn_id) if int(r) > 0})
    ridx = {r: i for i, r in enumerate(region_ids)}
    names = [id2name[r] for r in region_ids]
    Hagg = np.zeros((pi.shape[1], len(region_ids)))
    for j, b in enumerate(bn_id):
        if int(b) > 0:
            Hagg[j, ridx[int(b)]] = 1.0
    S = o2m.region_fc_similarity(np.asarray(H.uns["fc_mean"], float), bn_id, region_ids, ridx)
    idx = np.where(acr == "CP")[0]
    q = (Hagg.T @ pi[idx, :].sum(0)); q = q / q.sum()
    D = float(1.0 / (q @ S @ q))
    order = np.argsort(-q)
    return D, [(names[o], float(q[o])) for o in order[:3]]


def main():
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    M_xyz = warped_xyz_M(M, H)

    entries, kept, dropped = [], [], []
    for name, spec in PACKS.items():
        if not getattr(spec, "default", False):
            continue
        if name == "striatum":
            dropped.append(name); continue
        entries += spec.builder(M.var, H.var, atlas_root=ROOT); kept.append(name)
    print(f"packs kept ({len(kept)}): {kept}")
    print(f"packs dropped: {dropped}")

    model = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7,
                          epsilon=EPS, xyz_weight=XYZW, lam_anchor=1.0)
    costs = np.load(ROOT / "outputs/anndata/full_costs.npz")
    model.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"],
              M_xyz=M_xyz, region_anchors=entries)
    pi = model.pi.astype(np.float64)
    np.save("/var/tmp/pi_canonical_nostriatum.npy", pi)

    D, top = cp_split(pi, M, H)
    tn = " ".join(n for n, _ in top).lower()
    survives = bool(D >= o2m.D_ONE2MANY and "audate" in tn and "utamen" in tn)
    print("\nCP without the striatum pack:")
    print(f"  D = {D:.2f}   top -> " + ", ".join(f"{n} {v:.2f}" for n, v in top))
    print(f"\nRESULT: split {'SURVIVES: recovered from connectivity and space' if survives else 'COLLAPSES: supervised by the striatum pack'}")


if __name__ == "__main__":
    main()
