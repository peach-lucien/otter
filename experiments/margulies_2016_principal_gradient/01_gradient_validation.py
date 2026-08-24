"""OTTER × Margulies 2016 + Huntenburg 2021 principal-gradient validation.

[Margulies et al. 2016, PNAS](https://www.pnas.org/doi/10.1073/pnas.1608282113)
introduced the principal connectivity gradient, derived by diffusion-map
embedding of the resting-state FC matrix, the first non-trivial component
spans from primary sensorimotor cortex (unimodal end) to default-mode network
(transmodal end) and is the dominant organisational axis of human cortex.

[Huntenburg et al. 2021, Nat Comm](https://www.nature.com/articles/s41467-021-26703-z)
extended the same procedure to mouse rsfMRI and showed that a broadly
analogous principal gradient exists in mouse.

This experiment tests whether OTTER's π preserves the cross-species principal
gradient. Procedure:

  1. Compute the principal gradient on each species' FC matrix (Fisher-z →
     top-10% row threshold → cosine-similarity affinity → symmetric-normalised
     Laplacian → second eigenvector).
  2. Translate the mouse gradient through π as a **transport-weighted average**
       predicted_h[j] = Σ_i mouse_grad[i]·π[i,j] / Σ_i π[i,j]
     (the bare un-normalised `mouse_grad @ π` conflates the gradient with π's
     per-column mass; normalising by the column mass removes that confound).
  3. Compare predicted vs observed human gradient via Pearson r, both at parcel
     level and aggregated to Schaefer-400 regions. Eigenvectors are
     sign-ambiguous, so |r| is the headline.
  4. Permuted-π null (200 trials).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from otter.data import load_cached, load_pi, pi_provenance


def diffusion_components(fc: np.ndarray, *, top_pct: float = 10.0,
                         n_comp: int = 3) -> np.ndarray:
    """First ``n_comp`` non-trivial diffusion-map components of an (N, N) FC matrix.

    Returns an (n_comp, N) array. The embedding coordinate is D^(-1/2) u_k, not the
    raw symmetric-Laplacian eigenvector u_k (the two differ by a degree weighting).
    """
    n = fc.shape[0]
    fc = np.clip(fc, -0.9999, 0.9999).astype(np.float64)
    fcz = np.arctanh(fc)
    np.fill_diagonal(fcz, 0.0)
    thresh = np.percentile(fcz, 100.0 - top_pct, axis=1, keepdims=True)
    fcz_thr = np.where(fcz >= thresh, fcz, 0.0)
    row_norms = np.maximum(np.linalg.norm(fcz_thr, axis=1, keepdims=True), 1e-9)
    normed = fcz_thr / row_norms
    affinity = np.maximum(normed @ normed.T, 0.0)
    affinity = 0.5 * (affinity + affinity.T)
    degree = np.maximum(affinity.sum(axis=1), 1e-9)
    d_inv_sqrt = 1.0 / np.sqrt(degree)
    L_sym = np.eye(n) - (d_inv_sqrt[:, None] * affinity * d_inv_sqrt[None, :])
    L_sym = 0.5 * (L_sym + L_sym.T)
    _, eigvecs = eigh(L_sym, subset_by_index=[0, n_comp])
    return np.array([d_inv_sqrt * eigvecs[:, i] for i in range(1, n_comp + 1)])


def principal_gradient(fc: np.ndarray, hierarchy_ref: np.ndarray, *,
                       top_pct: float = 10.0, n_comp: int = 3):
    """The unimodal->transmodal (Margulies) gradient, selected against an external
    hierarchy reference (that species' own T1w/T2w myelin map).

    The component index is not hard-coded. In this FC data the leading non-trivial
    component is an anterior-posterior spatial axis and the unimodal->transmodal
    hierarchy is the second, so the component is selected by its correlation with
    the external reference rather than assumed.

    Returns (gradient, diagnostics).
    """
    comps = diffusion_components(fc, top_pct=top_pct, n_comp=n_comp)
    m = np.isfinite(hierarchy_ref)
    rhos = [float(spearmanr(c[m], hierarchy_ref[m]).statistic) for c in comps]
    k = int(np.argmax(np.abs(rhos)))
    grad = comps[k] * np.sign(rhos[k])        # orient so high = transmodal (low myelin)
    diag = {"selected_component": k + 1,
            "spearman_with_t1w_t2w_per_component": rhos,
            "n_ref_parcels": int(m.sum())}
    if abs(rhos[k]) < 0.3:
        print(f"  WARNING: best component correlates with the hierarchy reference at "
              f"only rho = {rhos[k]:+.2f}; the gradient may not be identifiable here.")
    return grad, diag


def route_normalized(mouse_vec: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Transport-weighted average, translate a mouse map to human space.

    predicted[j] = Σ_i mouse_vec[i]·π[i,j] / Σ_i π[i,j].  Human parcels that
    receive negligible π mass (OTTER's coupling is concentrated) are NaN.
    """
    num = mouse_vec @ pi
    den = pi.sum(axis=0)
    out = np.full(pi.shape[1], np.nan)
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    """Pearson r, Spearman ρ, n, over entries finite in both."""
    m = np.isfinite(a) & np.isfinite(b)
    return (float(pearsonr(a[m], b[m])[0]), float(spearmanr(a[m], b[m])[0]),
            int(m.sum()))


def aggregate_to_regions(node_vals: np.ndarray, node_region: np.ndarray) -> np.ndarray:
    """Mean a per-node vector into Schaefer-400 regions (index 1..400)."""
    finite = np.isfinite(node_vals)
    sums = np.bincount(node_region[finite], weights=node_vals[finite], minlength=401)
    cnt = np.bincount(node_region[finite], minlength=401)
    out = np.full(401, np.nan)
    nz = cnt > 0
    out[nz] = sums[nz] / cnt[nz]
    out[0] = np.nan
    return out


def main():
    print("=" * 80)
    print("OTTER × Margulies/Huntenburg principal-gradient validation")
    print("=" * 80)

    M, _ = load_cached("mouse", cache_dir=str(ROOT / "outputs/anndata"))
    H, _ = load_cached("human", cache_dir=str(ROOT / "outputs/anndata"))
    fc_mouse = M.uns["fc_mean"]
    fc_human = H.uns["fc_mean"]
    pi = load_pi()
    prov = pi_provenance()
    node_region = np.asarray(json.loads(
        (ROOT / "data_external/human_sc_meta.json").read_text())["node_region"], int)
    print(f"  π: {prov['pi_file']}  sha256 {prov['pi_sha256']}")
    print(f"  π: {pi.shape}, total mass {pi.sum():.4f}")

    # ---- external hierarchy references, used to select the right component ----
    human_ref = np.asarray(json.loads(
        (ROOT / "outputs/logs/buckner_krienen_2013_tethering.json").read_text()
    )["myelin_per_parcel"], float)                                   # human T1w/T2w
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fu", ROOT / "experiments/fulcher_2019_multimodal_gradient/01_gradient_validation.py")
    fu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fu)
    t1t2 = fu.load_mouse_t1t2()
    mouse_ref = np.array([t1t2.get(a, np.nan) for a in fu.load_mouse_parcel_acronyms()],
                         float)                                      # mouse T1w/T2w

    print("\nComputing principal gradients (diffusion-map embedding)...")
    mouse_grad, mouse_diag = principal_gradient(fc_mouse, mouse_ref, top_pct=10.0)
    human_grad, human_diag = principal_gradient(fc_human, human_ref, top_pct=10.0)
    print(f"  mouse: selected component {mouse_diag['selected_component']} "
          f"(rho with mouse T1w/T2w per component: "
          f"{[round(r, 2) for r in mouse_diag['spearman_with_t1w_t2w_per_component']]})")
    print(f"  human: selected component {human_diag['selected_component']} "
          f"(rho with human T1w/T2w per component: "
          f"{[round(r, 2) for r in human_diag['spearman_with_t1w_t2w_per_component']]})")

    # ---- translate mouse gradient through π (transport-weighted average) ----
    pred = route_normalized(mouse_grad, pi)
    if _corr(pred, human_grad)[0] < 0:        # resolve eigenvector sign
        pred = -pred
    r_p, rho_p, n_p = _corr(pred, human_grad)

    pred_reg = aggregate_to_regions(pred, node_region)
    hg_reg = aggregate_to_regions(human_grad, node_region)
    r_r, rho_r, n_r = _corr(pred_reg, hg_reg)

    print(f"\nCross-species gradient agreement (transport-weighted routing):")
    print(f"  parcel level   |r| = {abs(r_p):.3f}   |ρ| = {abs(rho_p):.3f}   (n = {n_p})")
    print(f"  region level   |r| = {abs(r_r):.3f}   |ρ| = {abs(rho_r):.3f}   (n = {n_r})")

    # ---- permuted-π null ---------------------------------------------------
    print(f"\nPermuted-π null (200 trials, row shuffle):")
    rng = np.random.default_rng(42)
    null_abs = []
    for _ in range(200):
        pn = route_normalized(mouse_grad, pi[rng.permutation(pi.shape[0])])
        null_abs.append(abs(_corr(pn, human_grad)[0]))
    null_abs = np.array(null_abs)
    emp_p = float((null_abs >= abs(r_p)).mean())
    print(f"  null |r| mean = {null_abs.mean():.3f}, "
          f"95% CI ({np.percentile(null_abs, 2.5):.3f}, "
          f"{np.percentile(null_abs, 97.5):.3f})")
    print(f"  empirical p = {emp_p:.3f}   (observed |r| = {abs(r_p):.3f}, "
          f"{abs(r_p) / max(null_abs.mean(), 1e-6):.0f}× null mean)")

    # ---- spatial-autocorrelation-preserving null (the permuted-π null above is
    #      too lenient; see otter.eval.nulls docstring) --------------------------
    from otter.eval.nulls import spin_null
    hx = H.var[["x", "y", "z"]].to_numpy(float)
    fin = np.isfinite(pred) & np.isfinite(human_grad)
    sp = spin_null(pred[fin], human_grad[fin], hx[fin], n_trials=1000, seed=0)
    print(f"\nSpin null (spatial-autocorrelation-preserving):")
    print(f"  |r| = {abs(r_p):.3f}   spin p = {sp['p_spin']:.4f}")

    out = {
        **prov,
        "routing": "transport-weighted average (normalised)",
        "top_pct_threshold": 10.0,
        "component_selection": {"mouse": mouse_diag, "human": human_diag,
                                "note": "component chosen by |rho| against that species' "
                                        "own T1w/T2w map"},
        "spin": {"p_spin": sp["p_spin"], "r_observed": sp["r_observed"],
                 "n_trials": 1000},
        "parcel_level": {"abs_pearson_r": abs(r_p), "abs_spearman_r": abs(rho_p),
                         "n": n_p},
        "region_level": {"abs_pearson_r": abs(r_r), "abs_spearman_r": abs(rho_r),
                         "n": n_r},
        "pearson_r": float(r_p),
        "abs_pearson_r": abs(r_p),
        "spearman_r": float(rho_p),
        "null": {"n_trials": 200, "abs_r_mean": float(null_abs.mean()),
                 "abs_r_ci95": [float(np.percentile(null_abs, 2.5)),
                                float(np.percentile(null_abs, 97.5))],
                 "empirical_p": emp_p},
        "mouse_gradient": mouse_grad.tolist(),
        "human_gradient": human_grad.tolist(),
        "predicted_human_gradient": pred.tolist(),
    }
    out_path = ROOT / "outputs" / "logs" / "margulies_2016_gradient.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
