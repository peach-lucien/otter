"""The production recipe for fitting a OTTER coupling.

Every analysis in the paper starts either from the released coupling or from a coupling refitted
under some ablation of the production recipe. That recipe was previously written out separately in
each experiment script and again in the harness used for the review round, so a change had to be
applied in several places and a notebook could disagree with the script that produced the
manuscript number. Scripts and notebooks now import the recipe from here instead.

Typical use::

    from otter.repro import load_inputs, anchor_warped_xyz, fit_coupling, CANONICAL

    M, H, costs, packs = load_inputs()
    M_xyz = anchor_warped_xyz(M, H)
    pi = fit_coupling(M, H, costs, packs, M_xyz, **CANONICAL)

To score a coupling against the held-out transcriptomic benchmark::

    BB = beauchamp_scorer()
    pairs, reg_cents, reg_masks, h_xyz, brain_c, _ = BB.build(M, H)
    agg = BB.score_all(pi, pairs, reg_cents, reg_masks, h_xyz, brain_c)["aggregate"]

Reproducing a released number does not require refitting: `load_canonical()` returns the released
coupling together with its sha256, and every log in ``outputs/logs`` records the sha of the coupling
that produced it. Compare those two rather than trusting that a re-run used the same input.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

__all__ = [
    "CANONICAL", "EPSILON", "XYZ_WEIGHT", "FC_WEIGHT", "SC_WEIGHT", "ALPHA",
    "repo_root", "load_inputs", "anchor_warped_xyz", "fit_coupling",
    "beauchamp_scorer", "load_canonical",
    "provenance", "refit_provenance", "coupling_sha_index", "stamp",
]

# ---------------------------------------------------------------------------------------------
# The production recipe. Selected by five-fold cross-validation on the 19 held-out Beauchamp
# homology pairs, not tuned by hand; see Methods and 02_methodology.ipynb.
# ---------------------------------------------------------------------------------------------
ALPHA = 0.5             # weight on the relational (Gromov-Wasserstein) term
EPSILON = 0.05          # entropic regularisation; 0.005 gives the near-deterministic showcase
XYZ_WEIGHT = 0.25       # weight on the anchor-warped spatial cost
FC_WEIGHT = 0.7         # within-species functional connectivity
SC_WEIGHT = 0.3         # within-species structural connectivity

#: Keyword arguments to :func:`fit_coupling` that reproduce ``pi_canonical.npy``.
CANONICAL = dict(alpha=ALPHA, xyz_weight=XYZ_WEIGHT, garin=True, packs=True, epsilon=EPSILON)


def repo_root() -> Path:
    """The `otter/` directory, however the caller happens to be invoked."""
    from otter.data.fetch import find_root
    return Path(find_root())


def load_inputs(root: Path | None = None):
    """Load both AnnDatas, the cost bundle and the anchor packs.

    Returns ``(M, H, costs, pack_entries)``. ``costs`` is the ``full_costs.npz`` archive, which
    carries the within-species FC and SC cost matrices.
    """
    from otter.data import load_cached
    from otter.data.anchor_packs import build_default_pack_entries

    root = Path(root) if root is not None else repo_root()
    M, _ = load_cached("mouse", cache_dir=root / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=root / "outputs/anndata")
    costs = np.load(root / "outputs/anndata/full_costs.npz")
    entries = build_default_pack_entries(M.var, H.var, atlas_root=str(root))
    return M, H, costs, entries


def anchor_warped_xyz(M, H) -> np.ndarray:
    """Cross-species spatial cost, after warping mouse coordinates onto the human brain.

    The two brains share no coordinate frame, so a raw coordinate distance is meaningless. The
    bilateral Garin anchors present in both species define a thin-plate-spline warp; mouse parcel
    centroids are pushed through it and the resulting mouse-to-human distance matrix, scaled to
    [0, 1], becomes the spatial term of the cross-species cost.

    The warp is fitted to the Garin landmark pairs, so the spatial scaffold is not supervision-free.
    The Figure 2 caption states this.
    """
    from scipy.interpolate import RBFInterpolator

    from otter.data.anchors import get_anchor_index

    im, ih = get_anchor_index(M.var), get_anchor_index(H.var)
    human_lookup = {(int(p), str(h)): int(k)
                    for k, p, h in zip(ih.pos, ih.pair_ids, ih.hemispheres)}
    matched = [(int(mp), human_lookup[(int(pid), str(hm))])
               for mp, pid, hm in zip(im.pos, im.pair_ids, im.hemispheres)
               if (int(pid), str(hm)) in human_lookup]
    if not matched:
        raise RuntimeError("no bilateral anchor matched between species; check the anchor index")

    mouse_xyz = M.var[["x", "y", "z"]].to_numpy(float)
    human_xyz = H.var[["x", "y", "z"]].to_numpy(float)
    warp = RBFInterpolator(mouse_xyz[[a for a, _ in matched]],
                           human_xyz[[b for _, b in matched]],
                           kernel="thin_plate_spline", smoothing=1e-3)
    d = np.sqrt(((warp(mouse_xyz)[:, None, :] - human_xyz[None, :, :]) ** 2).sum(-1))
    return (d / max(d.max(), 1e-9)).astype(np.float64)


def fit_coupling(M, H, costs, pack_entries, M_xyz, *,
                 alpha: float = ALPHA,
                 xyz_weight: float = XYZ_WEIGHT,
                 garin: bool = True,
                 packs: bool = True,
                 epsilon: float = EPSILON,
                 Cm_FC=None, Ch_FC=None, Cm_SC=None, Ch_SC=None) -> np.ndarray:
    """Fit a coupling. Defaults reproduce ``pi_canonical.npy``.

    The four switches are the rungs of the Figure 2a ablation ladder:

    ``alpha=0``        drop the connectivity term entirely (space and curation only)
    ``xyz_weight=0``   drop the spatial scaffold (connectivity and curation only)
    ``garin=False``    withhold the 21 Garin point anchors
    ``packs=False``    withhold the 26 curated region packs

    Passing ``Cm_FC``/``Ch_FC`` substitutes a different functional-connectivity cost, which is how
    the split-half refits and the rotated-connectome surrogate are built.

    The Garin flag is applied by masking ``garin_anchor`` in ``.var``; the original column is
    restored before returning, so callers may reuse the same AnnData across arms.
    """
    from otter.models import MultimodalFGW

    garin_M = M.var["garin_anchor"].fillna(False).to_numpy().astype(bool)
    garin_H = H.var["garin_anchor"].fillna(False).to_numpy().astype(bool)
    try:
        M.var["garin_anchor"] = garin_M if garin else np.zeros_like(garin_M)
        H.var["garin_anchor"] = garin_H if garin else np.zeros_like(garin_H)

        model = MultimodalFGW(use_sc=True, sc_weight=SC_WEIGHT, fc_weight=FC_WEIGHT,
                              epsilon=epsilon, xyz_weight=xyz_weight,
                              lam_anchor=1.0, alpha=alpha)
        kwargs = {} if xyz_weight == 0 else {"M_xyz": M_xyz}
        if Cm_FC is not None:
            kwargs["Cm_FC"] = Cm_FC
        if Ch_FC is not None:
            kwargs["Ch_FC"] = Ch_FC
        model.fit(M, H,
                  Cm_SC=costs["Cm_SC"] if Cm_SC is None else Cm_SC,
                  Ch_SC=costs["Ch_SC"] if Ch_SC is None else Ch_SC,
                  region_anchors=(pack_entries if packs else []),
                  **kwargs)
        return model.pi.astype(np.float64)
    finally:
        M.var["garin_anchor"] = garin_M
        H.var["garin_anchor"] = garin_H


def beauchamp_scorer(root: Path | None = None):
    """The held-out benchmark scorer, as a module with ``build()`` and ``score_all()``.

    Scores a coupling against the 19 mouse-human region correspondences of Beauchamp et al.,
    derived from whole-brain transcriptomic similarity and used by no part of the fit.

    Loaded by path because it still lives under ``experiments/``. It belongs in ``otter.eval``;
    moving it is deferred until the notebooks run green, so that the move is protected by something
    that would catch a mistake.
    """
    root = Path(root) if root is not None else repo_root()
    path = root / "experiments/section2_supervision/beauchamp_battery.py"
    if not path.exists():
        raise FileNotFoundError(f"Beauchamp scorer not found at {path}")
    spec = importlib.util.spec_from_file_location("otter_beauchamp_battery", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_canonical(name: str = "pi_canonical.npy") -> tuple[np.ndarray, dict]:
    """The released coupling and its provenance stamp, as ``(pi, {"pi_file", "pi_sha256"})``.

    Compare the returned sha against the ``pi_sha256`` recorded in whichever log you are trying to
    reproduce. If they differ, the number came from a different coupling and no amount of re-running
    will reconcile them.
    """
    from otter.data import load_pi, pi_provenance
    return load_pi(name), pi_provenance(name)


# ---------------------------------------------------------------------------------------------
# Provenance. Every log records which coupling produced it, because a re-run does not establish
# which input was used. The form of the record depends on whether the producer loads or refits.
# ---------------------------------------------------------------------------------------------
def provenance(name: str = "pi_canonical.npy") -> dict:
    """Stamp for a producer that LOADS a coupling: ``{"pi_file", "pi_sha256"}``.

    Use when the script reads a coupling off disk. The sha is of the file this run actually opened,
    so it cannot claim an input it did not use.
    """
    from otter.data import pi_provenance
    return pi_provenance(name)


def refit_provenance(pi: np.ndarray, *, recipe: dict | None = None,
                     reference: str = "pi_canonical.npy") -> dict:
    """Stamp for a producer that FITS its own coupling.

    There is no loaded file to hash, so borrowing the released sha would assert something this run
    never checked. Instead record the recipe, and measure how far the coupling fitted here agrees
    with the released one::

        {"fitted_here": True,
         "recipe": {...},
         "reproduces": {"reference", "reference_sha256", "argmax_match", "entrywise_r"}}

    For a production-arm refit ``argmax_match`` should be close to 1.0. For an ablation arm it will
    be lower, and the recorded value quantifies how far that arm departs from the release.
    """
    ref, ref_prov = load_canonical(reference)
    out = {"fitted_here": True, "recipe": dict(recipe or CANONICAL)}
    if ref.shape == pi.shape:
        out["reproduces"] = {
            "reference": ref_prov["pi_file"],
            "reference_sha256": ref_prov["pi_sha256"],
            "argmax_match": float((pi.argmax(1) == ref.argmax(1)).mean()),
            "entrywise_r": float(np.corrcoef(pi.ravel(), ref.ravel())[0, 1]),
        }
    else:
        out["reproduces"] = {"reference": ref_prov["pi_file"],
                             "reference_sha256": ref_prov["pi_sha256"],
                             "note": f"shape {pi.shape} != reference {ref.shape}; not comparable"}
    return out


def stamp(payload: dict, **prov) -> dict:
    """Merge a provenance stamp into an output dict, for use at the write site::

        json.dumps(stamp(out, **provenance()), indent=2)
        json.dumps(stamp(out, **refit_provenance(pi)), indent=2)

    The call belongs at the write site. A stamp added elsewhere in the file can come to describe
    something other than what the script wrote.
    """
    clash = set(payload) & set(prov)
    if clash:
        raise ValueError(f"provenance keys collide with result keys: {sorted(clash)}")
    return {**payload, **prov}


def coupling_sha_index(coupling_dir: Path | None = None) -> dict[str, str]:
    """``{sha256: filename}`` over every coupling in ``outputs/coupling``.

    Allows an audit to resolve a stamped sha to the coupling it names, rather than only testing
    whether it equals the released one. A log that compares couplings, such as the Control B
    sensitivity analysis, carries several shas. Each should resolve here.
    """
    import hashlib

    coupling_dir = Path(coupling_dir) if coupling_dir is not None else repo_root() / "outputs/coupling"
    index: dict[str, str] = {}
    for path in sorted(coupling_dir.glob("pi*.npy")):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        index[h.hexdigest()] = path.name
    return index
