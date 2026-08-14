"""OTTER × Fulcher 2019 multimodal-gradient validation.

[Fulcher, Murray, Zerbi & Wang 2019, PNAS](https://doi.org/10.1073/pnas.1814144116)
"Multimodal gradients across mouse cortex" showed that cytoarchitecture, gene
expression, interneuron density and long-range connectivity all vary together
along a single sensorimotor → prefrontal hierarchical axis of mouse cortex.
Two of their modality maps are used here:

  * **T1w:T2w ratio**, an intracortical-myelin / cytoarchitecture proxy, for
    40 mouse isocortical areas (`structInfoT1T2_ABAcortex40.csv`).
  * **Cytoarchitectural type**. Goulas et al.'s ordinal eulamination scale
    1 (agranular) → 4 (eulaminate), for 38 areas (`CytoarchitectureTypes.txt`).

The test is orthogonal to OTTER's inputs: both mouse modalities are
structural (π is built from FC + SC), and the human reference is the
independent HCP S1200 T1w/T2w myelin map.

Three panels:

  1. **T1w:T2w → human myelin.** Translate the mouse T1w:T2w map through π and
     compare to the human myelin map. Apples-to-apples, same modality.
  2. **Routed-territory characterisation.** We quantify how the human territory
     π actually reaches sits on the human principal connectivity gradient
     (Margulies/Huntenburg). Under the retired coupling that territory was a
     compact 205 of 400 Schaefer regions; the canonical coupling reaches 388,
     so the "concentration" framing no longer describes it, and the gradient-SD
     compression ratio is now ~1.00.
  3. **Cytoarchitecture → human myelin.** A second, independent mouse modality
     routed through π, multimodal convergence in the spirit of Fulcher's paper.

Routing is a transport-weighted average:
    predicted_h[j] = Σ_i m[i]·π[i,j] / Σ_i π[i,j]   over the assigned parcels.
Comparisons are at Schaefer-400 region granularity, with a permuted-π null.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data_external" / "fulcher_2019_gradients"
# The canonical coupling is whatever otter.data.load_pi() defaults to
# (pi_canonical.npy). Do NOT hard-code a path here: this script previously
# pinned pi_fc_plus_SC_with_all_packs.npy and so kept using the RETIRED
# coupling after the 2026-07-17 switch, silently, through several re-runs.
RETIRED_PI_FILE = "pi_fc_plus_SC_with_all_packs.npy"   # for the coverage control only
N_NULL = 200
SEED = 42


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------
def load_mouse_t1t2() -> dict[str, float]:
    """Allen acronym → T1w:T2w ratio (Fulcher's 40-area table)."""
    out: dict[str, float] = {}
    with open(DATA / "structInfoT1T2_ABAcortex40.csv") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if row and row[2]:
                out[row[2]] = float(row[10])  # col 10 = "NEW Ratio T1/T2"
    return out


def load_mouse_cytoarch() -> dict[str, float]:
    """Allen acronym → Goulas cytoarchitectural type (1–4 eulamination scale)."""
    out: dict[str, float] = {}
    for line in (DATA / "CytoarchitectureTypes.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            out[parts[0]] = float(parts[1])
    return out


def load_mouse_parcel_acronyms() -> list[str]:
    """Allen structure acronym for each of OTTER's 1,864 mouse parcels."""
    meta = json.loads((ROOT / "data_external" / "mouse_sc_meta.json").read_text())
    acr = meta["structure_acronyms"]
    return [acr[i] for i in meta["node_struct_idx"]]


def load_human_node_region() -> np.ndarray:
    """Schaefer-400 region id (1..400; 0 = subcortical) for each human node."""
    meta = json.loads((ROOT / "data_external" / "human_sc_meta.json").read_text())
    return np.asarray(meta["node_region"], dtype=int)


def load_human_myelin() -> np.ndarray:
    """HCP T1w/T2w myelin per Schaefer region, indexed [1..400]."""
    vals = np.full(401, np.nan)
    with open(DATA / "human_myelinmap_schaefer400_HOMERorder.csv") as f:
        for row in csv.DictReader(f):
            vals[int(row["otter_region_id"])] = float(row["t1t2_myelin"])
    return vals


def load_margulies_gradient() -> np.ndarray:
    """Per-node (2,094) human principal FC gradient from the Margulies experiment."""
    p = ROOT / "outputs" / "logs" / "margulies_2016_gradient.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found, run experiments/margulies_2016_principal_gradient/"
            "01_gradient_validation.py first (this experiment reuses its human gradient)."
        )
    return np.asarray(json.loads(p.read_text())["human_gradient"], dtype=float)


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------
def route_through_pi(mouse_vec: np.ndarray, pi: np.ndarray,
                     mask: np.ndarray) -> np.ndarray:
    """Transport-weighted average: translate a partial mouse map to human space.

    predicted[j] = Σ_{i∈mask} mouse_vec[i]·π[i,j] / Σ_{i∈mask} π[i,j].
    Human nodes receiving negligible mass from the masked parcels are NaN.
    """
    num = mouse_vec[mask] @ pi[mask, :]
    den = pi[mask, :].sum(axis=0)
    out = np.full(pi.shape[1], np.nan)
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out


def aggregate_to_regions(node_vals: np.ndarray,
                         node_region: np.ndarray) -> np.ndarray:
    """Mean a per-node vector into Schaefer regions, indexed [1..400]."""
    finite = np.isfinite(node_vals)
    sums = np.bincount(node_region[finite], weights=node_vals[finite], minlength=401)
    counts = np.bincount(node_region[finite], minlength=401)
    out = np.full(401, np.nan)
    nz = counts > 0
    out[nz] = sums[nz] / counts[nz]
    out[0] = np.nan  # index 0 = subcortical
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, int]:
    """Pearson r + analytical p, Spearman ρ, n, over entries finite in both."""
    m = np.isfinite(a) & np.isfinite(b)
    rp, pp = pearsonr(a[m], b[m])
    rs, _ = spearmanr(a[m], b[m])
    return float(rp), float(pp), float(rs), int(m.sum())


def main():
    print("=" * 80)
    print("OTTER × Fulcher 2019, multimodal cortical-gradient validation")
    print("=" * 80)

    from otter.data import load_pi, pi_provenance
    pi = load_pi()
    prov = pi_provenance()
    print(f"  π: {prov['pi_file']}  sha256 {prov['pi_sha256']}")
    parcel_acr = load_mouse_parcel_acronyms()
    node_region = load_human_node_region()
    if len(parcel_acr) != pi.shape[0]:
        raise ValueError(f"mouse parcels {len(parcel_acr)} != π rows {pi.shape[0]}")
    if len(node_region) != pi.shape[1]:
        raise ValueError(f"human nodes {len(node_region)} != π cols {pi.shape[1]}")
    print(f"  π: {pi.shape}, total mass {pi.sum():.4f}")

    myelin_reg = load_human_myelin()

    def route_modality(value_by_acr: dict[str, float]):
        vec = np.array([value_by_acr.get(a, np.nan) for a in parcel_acr])
        mask = np.isfinite(vec)
        pred_reg = aggregate_to_regions(route_through_pi(vec, pi, mask), node_region)
        n_areas = len({a for a in parcel_acr if a in value_by_acr})
        return vec, mask, pred_reg, n_areas

    # ===== Panel 1. T1w:T2w → human myelin =================================
    t1t2_vec, t1t2_mask, pred_t1t2, n_t1t2_areas = route_modality(load_mouse_t1t2())
    r1, p1, rs1, n1 = _corr(pred_t1t2, myelin_reg)
    territory = np.isfinite(pred_t1t2)  # the routed human territory (regions)
    print(f"\n[Panel 1] T1w:T2w → human myelin")
    print(f"  {t1t2_mask.sum()} mouse parcels ({n_t1t2_areas} areas) → "
          f"{int(territory.sum())} Schaefer regions")
    print(f"  Pearson r = {r1:+.3f}  Spearman ρ = {rs1:+.3f}  "
          f"p = {p1:.2e}  (n = {n1})")

    # ===== Panel 3, cytoarchitecture → human myelin ========================
    cyto_vec, cyto_mask, pred_cyto, n_cyto_areas = route_modality(load_mouse_cytoarch())
    r3, p3, rs3, n3 = _corr(pred_cyto, myelin_reg)
    print(f"\n[Panel 3] Cytoarchitectural type → human myelin")
    print(f"  {cyto_mask.sum()} mouse parcels ({n_cyto_areas} areas)")
    print(f"  Pearson r = {r3:+.3f}  Spearman ρ = {rs3:+.3f}  "
          f"p = {p3:.2e}  (n = {n3})")

    # ===== Panel 2, routed-territory characterisation ======================
    grad_reg = aggregate_to_regions(load_margulies_gradient(), node_region)
    # orient principal gradient so high end = heavily myelinated (sensorimotor)
    if _corr(grad_reg, myelin_reg)[0] < 0:
        grad_reg = -grad_reg
    all_cortex = np.isfinite(grad_reg)
    g_all = grad_reg[all_cortex]
    g_terr = grad_reg[all_cortex & territory]
    std_all, std_terr = float(np.std(g_all)), float(np.std(g_terr))
    r_pg, _, _, n_pg = _corr(pred_t1t2, grad_reg)
    print(f"\n[Panel 2] Routed-territory position on the principal gradient")
    print(f"  gradient SD, all cortex: {std_all:.4f}   routed territory: "
          f"{std_terr:.4f}   (compression ×{std_terr/std_all:.2f})")
    print(f"  predicted-vs-gradient r = {r_pg:+.3f}, territory is "
          f"gradient-degenerate, not a hierarchy ruler here")

    # ===== coverage control =================================================
    # The canonical coupling reaches a LARGER human territory than the retired
    # one. Any gain in panels 1/3 could therefore be territory size rather than
    # a better coupling. Re-score the canonical prediction on exactly the region
    # set the retired coupling reached, so the two are compared on equal ground.
    from otter.data import load_pi as _load_pi, pi_provenance as _prov
    pi_retired = _load_pi(RETIRED_PI_FILE)
    prov_retired = _prov(RETIRED_PI_FILE)
    ret_t1t2 = aggregate_to_regions(
        route_through_pi(t1t2_vec, pi_retired, t1t2_mask), node_region)
    ret_cyto = aggregate_to_regions(
        route_through_pi(cyto_vec, pi_retired, cyto_mask), node_region)
    ret_territory = np.isfinite(ret_t1t2)

    def _restrict(vec, mask):
        out = np.full_like(vec, np.nan)
        out[mask] = vec[mask]
        return out

    r1_cov, p1_cov, rs1_cov, n1_cov = _corr(_restrict(pred_t1t2, ret_territory),
                                            myelin_reg)
    r3_cov, p3_cov, rs3_cov, n3_cov = _corr(_restrict(pred_cyto, ret_territory),
                                            myelin_reg)
    # and the retired coupling's own numbers, on its own territory, for reference
    r1_ret, _, _, n1_ret = _corr(ret_t1t2, myelin_reg)
    r3_ret, _, _, n3_ret = _corr(ret_cyto, myelin_reg)

    print(f"\n[Coverage control] canonical π scored on the RETIRED π's territory")
    print(f"  retired territory: {int(ret_territory.sum())} regions   "
          f"canonical territory: {int(territory.sum())} regions")
    print(f"  panel 1  full {r1:+.3f} (n={n1})  →  restricted {r1_cov:+.3f} "
          f"(n={n1_cov})   [retired π on its own territory {r1_ret:+.3f}]")
    print(f"  panel 3  full {r3:+.3f} (n={n3})  →  restricted {r3_cov:+.3f} "
          f"(n={n3_cov})   [retired π on its own territory {r3_ret:+.3f}]")

    # ===== permuted-π null for panels 1 & 3 =================================
    print(f"\nPermuted-π null ({N_NULL} trials)...")
    rng = np.random.default_rng(SEED)
    null1, null3 = [], []
    for _ in range(N_NULL):
        perm = rng.permutation(pi.shape[0])
        pn1 = aggregate_to_regions(route_through_pi(t1t2_vec, pi[perm], t1t2_mask),
                                   node_region)
        pn3 = aggregate_to_regions(route_through_pi(cyto_vec, pi[perm], cyto_mask),
                                   node_region)
        null1.append(_corr(pn1, myelin_reg)[0])
        null3.append(_corr(pn3, myelin_reg)[0])
    null1, null3 = np.array(null1), np.array(null3)
    emp1 = float((null1 >= r1).mean())
    emp3 = float((null3 >= r3).mean())
    print(f"  panel 1  null r mean = {null1.mean():+.3f}  "
          f"95% CI ({np.percentile(null1, 2.5):+.3f}, "
          f"{np.percentile(null1, 97.5):+.3f})  empirical p = {emp1:.3f}")
    print(f"  panel 3  null r mean = {null3.mean():+.3f}  "
          f"95% CI ({np.percentile(null3, 2.5):+.3f}, "
          f"{np.percentile(null3, 97.5):+.3f})  empirical p = {emp3:.3f}")

    # ===== save =============================================================
    def _nullblock(null):
        return {"n_trials": N_NULL, "r_mean": float(null.mean()),
                "r_ci95": [float(np.percentile(null, 2.5)),
                           float(np.percentile(null, 97.5))]}

    out = {
        **prov,
        "granularity": "Schaefer-400 cortical regions",
        "coverage_control": {
            "note": "canonical π re-scored on exactly the human regions the "
                    "retired coupling reached, isolating territory size from "
                    "the coupling itself",
            "retired_pi_file": prov_retired["pi_file"],
            "retired_pi_sha256": prov_retired["pi_sha256"],
            "n_regions_retired_territory": int(ret_territory.sum()),
            "n_regions_canonical_territory": int(territory.sum()),
            "panel1_t1t2": {
                "canonical_full_r": r1, "canonical_full_n": n1,
                "canonical_restricted_r": r1_cov, "canonical_restricted_n": n1_cov,
                "retired_r": r1_ret, "retired_n": n1_ret,
            },
            "panel3_cytoarch": {
                "canonical_full_r": r3, "canonical_full_n": n3,
                "canonical_restricted_r": r3_cov, "canonical_restricted_n": n3_cov,
                "retired_r": r3_ret, "retired_n": n3_ret,
            },
        },
        "routed_territory": {
            "n_mouse_parcels": int(t1t2_mask.sum()),
            "n_human_regions": int(territory.sum()),
        },
        "panel1_t1t2_vs_myelin": {
            "n_fulcher_areas": n_t1t2_areas, "pearson_r": r1,
            "pearson_p_analytical": p1, "spearman_r": rs1, "n_regions": n1,
            "null": {**_nullblock(null1), "empirical_p": emp1},
        },
        "panel2_gradient_territory": {
            "gradient_sd_all_cortex": std_all,
            "gradient_sd_routed_territory": std_terr,
            "compression_ratio": std_terr / std_all,
            "predicted_vs_gradient_r": r_pg,
            "n_regions": n_pg,
        },
        "panel3_cytoarch_vs_myelin": {
            "n_fulcher_areas": n_cyto_areas, "pearson_r": r3,
            "pearson_p_analytical": p3, "spearman_r": rs3, "n_regions": n3,
            "null": {**_nullblock(null3), "empirical_p": emp3},
        },
        "predicted_t1t2_region": pred_t1t2.tolist(),
        "predicted_cytoarch_region": pred_cyto.tolist(),
        "myelin_region": myelin_reg.tolist(),
        "gradient_region": grad_reg.tolist(),
        "territory_mask": territory.tolist(),
    }
    out_path = ROOT / "outputs" / "logs" / "fulcher_2019_gradient.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
