"""OTTER × Buckner & Krienen 2013, the tethering hypothesis (negative control).

[Buckner & Krienen 2013, Trends Cogn Sci](https://doi.org/10.1016/j.tics.2013.09.017),
"The evolution of distributed association networks in the human brain", argue
that human association cortex expanded so much it became evolutionarily
**"untethered"** from the sensory hierarchies and molecular gradients that
organise primary cortex. Implication for a mouse↔human mapping: there is no
well-defined mouse homologue of the expanded human association cortex, so a
faithful coupling should be **confident over sensorimotor cortex and sparse /
unconfident over association cortex**.

This is a negative-control / falsification test: if OTTER's π were uniformly
confident everywhere, including over association cortex that the field says
has no clear mouse homologue, that would signal over-fitting.

Test: for every human cortical parcel, measure OTTER's **coverage**, the total
π mass it receives from the mouse brain (the per-column mass of the coupling)
and ask whether it collapses toward association cortex along the sensorimotor →
association axis (the HCP T1w/T2w myelin map; high myelin = sensorimotor).

Note: π's per-parcel *entropy* (the diffuseness of a parcel's mouse origin) is
reported too but is flat, it is the *amount* of coverage, not its diffuseness,
that carries the tethering signal.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import entropy, mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from otter.data import load_pi, pi_provenance                    # noqa: E402

DATA = ROOT / "data_external"
PI_NAME = "pi_canonical.npy"                       # canonical coupling
PI_FILE = f"outputs/coupling/{PI_NAME}"
N_NULL = 1000
SEED = 42


def main():
    print("=" * 80)
    print("OTTER × Buckner & Krienen 2013, tethering-hypothesis negative control")
    print("=" * 80)

    pi = load_pi(PI_NAME)
    prov = pi_provenance(PI_NAME)
    print(f"π file: {prov['pi_file']}  sha256 {prov['pi_sha256']}")
    node_region = np.asarray(json.loads(
        (DATA / "human_sc_meta.json").read_text())["node_region"], int)

    # sensorimotor → association axis: HCP T1w/T2w myelin per Schaefer region
    myelin_reg = {}
    with open(DATA / "fulcher_2019_gradients/human_myelinmap_schaefer400_OTTERorder.csv") as f:
        for row in csv.DictReader(f):
            myelin_reg[int(row["otter_region_id"])] = float(row["t1t2_myelin"])
    myelin = np.array([myelin_reg.get(r, np.nan) for r in node_region])

    # ---- per-human-parcel OTTER coverage + entropy ------------------------
    col_mass = pi.sum(axis=0)
    coverage = np.log10(np.maximum(col_mass, 1e-300))      # log π mass per parcel
    col_norm = pi / np.maximum(col_mass, 1e-300)
    col_entropy = np.array([entropy(col_norm[:, j]) for j in range(pi.shape[1])])

    ctx = (node_region >= 1) & np.isfinite(myelin)
    cov, mye, ent = coverage[ctx], myelin[ctx], col_entropy[ctx]
    print(f"\n  human cortical parcels: {ctx.sum()}")
    print(f"  π coverage (log10 column mass): {cov.min():.0f} … {cov.max():.0f}")

    # ---- correlation with the sensorimotor→association axis ---------------
    rho_cov, p_cov = spearmanr(cov, mye)
    rho_ent, _ = spearmanr(ent, mye)
    print(f"\n  Spearman ρ, coverage vs myelin axis = {rho_cov:+.3f}  (p = {p_cov:.1e})")
    print(f"  Spearman ρ, entropy  vs myelin axis = {rho_ent:+.3f}  (flat, see docstring)")

    # ---- decile curve along the sensorimotor→association axis -------------
    order = np.argsort(mye)               # ascending myelin: association → sensorimotor
    deciles = []
    print(f"\n  coverage by myelin decile (D1 = association … D10 = sensorimotor):")
    for d in range(10):
        b = order[d * len(order) // 10:(d + 1) * len(order) // 10]
        deciles.append(float(cov[b].mean()))
        print(f"    D{d + 1:<2d}  mean log10 coverage = {cov[b].mean():+7.2f}")

    # ---- association vs sensorimotor tertile contrast --------------------
    t = len(order) // 3
    assoc, senso = order[:t], order[-t:]
    mw = mannwhitneyu(cov[senso], cov[assoc])
    print(f"\n  association tertile  mean log10 coverage = {cov[assoc].mean():.1f}")
    print(f"  sensorimotor tertile mean log10 coverage = {cov[senso].mean():.1f}")
    print(f"  Mann-Whitney U: sensorimotor > association  p = {mw.pvalue:.2e}")

    # ---- permuted null: shuffle the myelin axis --------------------------
    rng = np.random.default_rng(SEED)
    obs_diff = cov[senso].mean() - cov[assoc].mean()
    null = []
    for _ in range(N_NULL):
        o = rng.permutation(len(cov))
        null.append(cov[o[-t:]].mean() - cov[o[:t]].mean())
    null = np.array(null)
    emp_p = float((np.abs(null) >= abs(obs_diff)).mean())
    print(f"\n  observed sensorimotor−association coverage gap = {obs_diff:.1f} log units")
    print(f"  permuted-axis null gap: mean {null.mean():+.2f}, |gap| 95th pct "
          f"{np.percentile(np.abs(null), 95):.2f}  →  empirical p = {emp_p:.3f}")

    passed = mw.pvalue < 1e-3 and obs_diff > 1.0
    verdict = ("PASS. OTTER is sparsest over association cortex, "
               "consistent with tethering") if passed else "INCONCLUSIVE"
    print(f"\nVERDICT: {verdict}")

    out = {
        **prov,
        "n_cortical_parcels": int(ctx.sum()),
        "spearman_coverage_vs_myelin": float(rho_cov),
        "spearman_entropy_vs_myelin": float(rho_ent),
        "decile_coverage": deciles,
        "association_tertile_coverage": float(cov[assoc].mean()),
        "sensorimotor_tertile_coverage": float(cov[senso].mean()),
        "coverage_gap_log_units": float(obs_diff),
        "mannwhitney_p": float(mw.pvalue),
        "null": {"n_trials": N_NULL, "gap_95pct": float(np.percentile(np.abs(null), 95)),
                 "empirical_p": emp_p},
        "verdict": verdict,
        "coverage_per_parcel": coverage.tolist(),
        "entropy_per_parcel": col_entropy.tolist(),
        "myelin_per_parcel": myelin.tolist(),
    }
    out_path = ROOT / "outputs" / "logs" / "buckner_krienen_2013_tethering.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
