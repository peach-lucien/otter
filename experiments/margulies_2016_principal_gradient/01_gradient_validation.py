"""HOMER × Margulies 2016 + Huntenburg 2021 principal-gradient validation.

[Margulies et al. 2016, PNAS](https://www.pnas.org/doi/10.1073/pnas.1608282113)
introduced the principal connectivity gradient — derived by diffusion-map
embedding of the resting-state FC matrix, the first non-trivial component
spans from primary sensorimotor cortex (unimodal end) to default-mode network
(transmodal end) and is the dominant organisational axis of human cortex.

[Huntenburg et al. 2021, Nat Comm](https://www.nature.com/articles/s41467-021-26703-z)
extended the same procedure to mouse rsfMRI and showed that a broadly
analogous principal gradient exists in mouse, spanning from sensorimotor to
mouse-DMN-like regions.

This experiment tests whether HOMER's π preserves the cross-species principal
gradient. Procedure:

  1. Compute the principal gradient on the mouse FC matrix:
       a. Fisher-z transform the FC correlations
       b. Threshold (keep top 10 % per row)
       c. Cosine similarity of row-profiles → affinity W
       d. Symmetric-normalised graph Laplacian L = I − D^{-½} W D^{-½}
       e. Take the second-smallest-eigenvalue eigenvector → principal gradient
  2. Same procedure on human FC → observed human gradient.
  3. Translate mouse gradient through π:  predicted_h = mouse_grad @ π.
  4. Compare predicted vs observed human gradient via Pearson r (over 2,094
     human parcels).  Handle the ±sign ambiguity of eigenvectors by reporting
     |r| as the headline metric.
  5. Permuted-π null (200 trials, within-row permutation).

This is a Beauchamp-independent, anchor-orthogonal test — the gradient is a
brain-wide ordering, not a per-region label.  If HOMER's π preserves the
gross organisational axis of cortex across species, |r| should be well above
the permuted-π null.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from homer.data import load_cached


def principal_gradient(fc: np.ndarray, *, top_pct: float = 10.0,
                        n_components: int = 1) -> np.ndarray:
    """Margulies-style principal gradient from a (N, N) FC correlation matrix.

    Returns the second-smallest-eigenvalue eigenvector of the symmetric
    normalised graph Laplacian (first non-trivial diffusion-map component).
    """
    n = fc.shape[0]
    # Fisher-z, blank diagonal
    fc = np.clip(fc, -0.9999, 0.9999).astype(np.float64)
    fcz = np.arctanh(fc)
    np.fill_diagonal(fcz, 0.0)

    # Row-wise threshold: keep top `top_pct` per row, zero the rest
    thresh = np.percentile(fcz, 100.0 - top_pct, axis=1, keepdims=True)
    fcz_thr = np.where(fcz >= thresh, fcz, 0.0)

    # Cosine similarity of row profiles → affinity (non-negative)
    row_norms = np.linalg.norm(fcz_thr, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-9)
    normed = fcz_thr / row_norms
    affinity = normed @ normed.T
    affinity = np.maximum(affinity, 0.0)
    affinity = 0.5 * (affinity + affinity.T)  # ensure symmetric

    # Symmetric-normalised graph Laplacian
    degree = affinity.sum(axis=1)
    degree[degree < 1e-9] = 1e-9
    d_inv_sqrt = 1.0 / np.sqrt(degree)
    L_sym = np.eye(n) - (d_inv_sqrt[:, None] * affinity * d_inv_sqrt[None, :])
    L_sym = 0.5 * (L_sym + L_sym.T)

    # Smallest eigenvalues (we need 2nd smallest = first non-trivial)
    # eigh on dense is O(n^3) but fine for n ≤ ~2500
    eigvals, eigvecs = eigh(L_sym, subset_by_index=[0, n_components])
    # Skip the zero-eigenvalue constant vector at index 0
    if n_components == 1:
        return eigvecs[:, 1]
    return eigvecs[:, 1:n_components + 1]


def _flip_sign_to_match(g: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Eigenvectors are sign-ambiguous; flip g so its correlation with
    reference is positive."""
    r = np.corrcoef(g, reference)[0, 1]
    return -g if r < 0 else g


def main():
    print("=" * 80)
    print("HOMER × Margulies/Huntenburg principal-gradient validation")
    print("=" * 80)

    # ---- Load FC + π ----
    M, _ = load_cached("mouse", cache_dir=str(ROOT / "outputs/anndata"))
    H, _ = load_cached("human", cache_dir=str(ROOT / "outputs/anndata"))
    fc_mouse = M.uns["fc_mean"]
    fc_human = H.uns["fc_mean"]
    pi = np.load(ROOT / "outputs/coupling/pi_fc_plus_SC_with_all_packs.npy")
    print(f"  FC mouse: {fc_mouse.shape}  range {fc_mouse.min():.2f} to {fc_mouse.max():.2f}")
    print(f"  FC human: {fc_human.shape}  range {fc_human.min():.2f} to {fc_human.max():.2f}")
    print(f"  π:        {pi.shape}, total mass {pi.sum():.4f}")

    # ---- Compute principal gradients per species ----
    print("\nComputing principal gradients (diffusion-map embedding)...")
    mouse_grad = principal_gradient(fc_mouse, top_pct=10.0)
    human_grad = principal_gradient(fc_human, top_pct=10.0)
    print(f"  mouse_grad: shape={mouse_grad.shape}, range "
          f"{mouse_grad.min():+.3f} to {mouse_grad.max():+.3f}")
    print(f"  human_grad: shape={human_grad.shape}, range "
          f"{human_grad.min():+.3f} to {human_grad.max():+.3f}")

    # ---- Translate mouse gradient via π → predicted human ----
    pred_human = mouse_grad @ pi
    # Fix sign ambiguity: flip predicted to maximise alignment with observed
    pred_aligned = _flip_sign_to_match(pred_human, human_grad)

    r_p, p_p = pearsonr(pred_aligned, human_grad)
    r_s, _ = spearmanr(pred_aligned, human_grad)
    abs_r = abs(r_p)

    print(f"\nCross-species gradient agreement (Pearson r over 2,094 human parcels):")
    print(f"  Pearson r           = {r_p:+.3f}  (analytical p = {p_p:.4e})")
    print(f"  |r|  (sign-resolved) = {abs_r:.3f}")
    print(f"  Spearman ρ          = {r_s:+.3f}")

    # ---- Permuted-π null ----
    print(f"\nPermuted-π null (200 trials, within-row shuffle):")
    rng = np.random.default_rng(seed=42)
    n_trials = 200
    null_abs = []
    null_r = []
    for _ in range(n_trials):
        perm = rng.permutation(pi.shape[0])
        pi_n = pi[perm]
        pred_n = mouse_grad @ pi_n
        pred_n = _flip_sign_to_match(pred_n, human_grad)
        r_n, _ = pearsonr(pred_n, human_grad)
        null_r.append(r_n)
        null_abs.append(abs(r_n))
    null_r = np.array(null_r); null_abs = np.array(null_abs)
    emp_p = float((null_abs >= abs_r).mean())
    print(f"  null |r| mean = {null_abs.mean():+.3f}, "
          f"95% CI ({np.percentile(null_abs, 2.5):+.3f}, "
          f"{np.percentile(null_abs, 97.5):+.3f})")
    print(f"  empirical p (observed |r| ≥ null |r|) = {emp_p:.3f}")

    # ---- Save ----
    out = {
        "pi_file":             "outputs/coupling/pi_fc_plus_SC_with_all_packs.npy",
        "top_pct_threshold":   10.0,
        "n_human_parcels":     int(len(human_grad)),
        "pearson_r":           float(r_p),
        "abs_pearson_r":       float(abs_r),
        "pearson_p_analytical": float(p_p),
        "spearman_r":          float(r_s),
        "null": {
            "n_trials":   n_trials,
            "abs_r_mean": float(null_abs.mean()),
            "abs_r_ci95": [float(np.percentile(null_abs, 2.5)),
                            float(np.percentile(null_abs, 97.5))],
            "empirical_p": emp_p,
        },
        "mouse_gradient":      mouse_grad.tolist(),
        "human_gradient":      human_grad.tolist(),
        "predicted_human_gradient": pred_aligned.tolist(),
    }
    out_path = ROOT / "outputs" / "logs" / "margulies_2016_gradient.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
