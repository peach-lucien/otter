"""ROADMAP item D: subject-level cross-validation.

Independent of held-out anchor CV (which tests the model's generalization
across SPATIAL anchor positions). This tests its generalization across
SUBJECTS — does the π trained on 80% of subjects predict the held-out 20%'s
mean FC?

For each of K random splits:
  1. Build mean FC_train (80% of mice, 80% of humans) via stream_mean_fc_subset.
  2. Recompute the FC cost matrices C_m^train, C_h^train.
  3. Solve FGW with full anchor supervision (no anchor holdouts here).
  4. Predict test human FC: F_pred = π.T @ F_test_mouse @ π / norm.
  5. Pearson r vs F_test_human (held-out 20%).

Compares to:
  - Training on ALL subjects, evaluating on the same set
    (zero generalisation gap) — this is the existing E1 result.
  - Random π baseline (sanity).

The 'fc_plus_SC' config is used as the production reference. Other configs
can be added by editing CONFIGS.

Saves outputs/logs/subject_cv.json with per-split metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import ot

ROOT = Path(__file__).resolve().parents[2]  # homer/  (script lives at experiments/D_subject_cv/)
sys.path.insert(0, str(ROOT / "src"))
from homer.data import load_cached, stream_mean_fc_subset            # noqa: E402
from homer.data.anchors  import get_anchor_index                     # noqa: E402
from homer.data.networks import PAIRID_TO_NETWORK, NETWORKS, assign_networks  # noqa: E402
from homer.costs import correlation_distance                         # noqa: E402
from homer.eval.translation import fc_translation_quality            # noqa: E402

ANN = ROOT / "outputs" / "anndata"
LOG = ROOT / "outputs" / "logs"; LOG.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")


CONFIGS = {
    "fc_only": {
        "use_SC": False,
        "M":      {"xyz": 0.5},
    },
    "fc_plus_SC": {
        "use_SC": True,
        "M":      {"xyz": 0.5},
    },
}


def build_M(M_weights, costs, idx_m, idx_h, lam_anchor=1.0):
    """Build M with FULL anchor supervision (no held-out anchors)."""
    M = np.zeros_like(costs["M_xyz"], dtype=np.float64)
    if M_weights.get("xyz", 0):
        M += M_weights["xyz"] * costs["M_xyz"].astype(np.float64)
    # All 42 anchors are visible
    for k, mp in enumerate(idx_m.pos):
        M[mp, :] = lam_anchor
        M[mp, idx_h.pos[k]] = 0.0
    for k, hp in enumerate(idx_h.pos):
        mp_correct = idx_m.pos[k]
        col_mask = M[:, hp] < lam_anchor
        M[col_mask, hp] = lam_anchor
        M[mp_correct, hp] = 0.0
    return M


def build_costs_from_train_fc(fc_train_m, fc_train_h, costs_static, use_SC):
    """Rebuild C_m, C_h from a per-fold train FC plus optional static SC."""
    Cm_FC = correlation_distance(fc_train_m.astype(np.float64))
    Ch_FC = correlation_distance(fc_train_h.astype(np.float64))
    # Normalise to [0, 1] like build_multimodal_costs.py does
    Cm_FC = Cm_FC / max(Cm_FC[~np.eye(Cm_FC.shape[0], dtype=bool)].max(), 1e-9)
    Ch_FC = Ch_FC / max(Ch_FC[~np.eye(Ch_FC.shape[0], dtype=bool)].max(), 1e-9)
    if use_SC:
        Cm = 0.7 * Cm_FC + 0.3 * costs_static["SC_m"].astype(np.float64)
        Ch = 0.7 * Ch_FC + 0.3 * costs_static["SC_h"].astype(np.float64)
    else:
        Cm, Ch = Cm_FC, Ch_FC
    return Cm, Ch


def main(args):
    H, _ = load_cached("human", cache_dir=ANN)
    M_, _ = load_cached("mouse", cache_dir=ANN)
    idx_h = get_anchor_index(H.var); idx_m = get_anchor_index(M_.var)
    n_m, n_h = M_.uns["n_nodes"], H.uns["n_nodes"]
    n_subj_m = M_.uns["n_subjects"]; n_subj_h = H.uns["n_subjects"]
    p = np.full(n_m, 1.0 / n_m)

    d = np.load(ANN / "full_costs.npz")
    costs_static = {
        "SC_m":  d["Cm_SC"], "SC_h":  d["Ch_SC"],
        "M_xyz": d["M_xyz"],
    }
    # assign_networks returns a numpy array (n_h,) of network labels
    net_h_arr = np.asarray(assign_networks(H.var, idx_h))

    rng = np.random.default_rng(args.seed)
    cache_path = LOG / "subject_cv.json"
    state = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    # Per-config K-fold
    cfg_name = args.config
    cfg = CONFIGS[cfg_name]
    # Cache key omits K so we can resume fold-by-fold across runs
    cfg_key = f"{cfg_name}__test{args.test_frac:.2f}_seed{args.seed}"
    if cfg_key not in state:
        state[cfg_key] = {}

    print(f"\n=== {cfg_key} ===")
    print(f"  config: use_SC={cfg['use_SC']}, M={cfg['M']}")
    print(f"  n_subj_m={n_subj_m}, n_subj_h={n_subj_h}, "
          f"K={args.k_folds}, test_frac={args.test_frac}")

    fold_results = state[cfg_key]
    for fold in range(args.k_folds):
        fold_key = f"fold_{fold}"
        if fold_key in fold_results and not args.recompute:
            r = fold_results[fold_key]
            print(f"  fold {fold}: cached  test_r={r['test_r_overall']:.3f}  "
                  f"train_r={r['train_r_overall']:.3f}")
            continue

        # Random subject splits (separate seeds per species so we don't
        # accidentally couple them)
        rng_m = np.random.default_rng(args.seed * 1000 + fold)
        rng_h = np.random.default_rng(args.seed * 1000 + fold + 500)
        test_m = rng_m.choice(n_subj_m,
                              size=max(1, int(round(args.test_frac * n_subj_m))),
                              replace=False)
        test_h = rng_h.choice(n_subj_h,
                              size=max(1, int(round(args.test_frac * n_subj_h))),
                              replace=False)
        train_m = np.setdiff1d(np.arange(n_subj_m), test_m)
        train_h = np.setdiff1d(np.arange(n_subj_h), test_h)
        print(f"  fold {fold}: train m={len(train_m)} h={len(train_h)}, "
              f"test m={len(test_m)} h={len(test_h)}")

        t = time.time()
        # Stream TRAIN means + per-cell counts; derive TEST means by
        # *count-aware* subtraction. The old code assumed every cell sees the
        # same number of subjects (i.e., n_obs ≡ n_subj) and divided by
        # `len(test_*)`. That's exact for mouse (full coverage) but biased on
        # the human side, where ~15% of cells have n_obs < 113 due to
        # subject-specific dropout.
        #   test_mean[i,j] = (total_sum[i,j] - train_sum[i,j]) / test_count[i,j]
        # where total_sum   = total_mean * n_obs_total
        #       train_sum   = train_mean * n_obs_train
        #       test_count  = (n_obs_total - n_obs_train).clip(min=1)
        fc_train_m, n_obs_train_m, _ = stream_mean_fc_subset("mouse", include_subjects=train_m)
        fc_train_h, n_obs_train_h, _ = stream_mean_fc_subset("human", include_subjects=train_h)
        fc_total_m = M_.uns["fc_mean"].astype(np.float64)
        fc_total_h = H.uns["fc_mean"].astype(np.float64)
        n_obs_total_m = np.asarray(M_.uns["fc_n_obs"]).astype(np.float64)
        n_obs_total_h = np.asarray(H.uns["fc_n_obs"]).astype(np.float64)
        n_obs_train_m = n_obs_train_m.astype(np.float64)
        n_obs_train_h = n_obs_train_h.astype(np.float64)
        sum_total_m = fc_total_m * n_obs_total_m
        sum_total_h = fc_total_h * n_obs_total_h
        sum_train_m = fc_train_m * n_obs_train_m
        sum_train_h = fc_train_h * n_obs_train_h
        n_obs_test_m = (n_obs_total_m - n_obs_train_m).clip(min=1)
        n_obs_test_h = (n_obs_total_h - n_obs_train_h).clip(min=1)
        fc_test_m  = ((sum_total_m - sum_train_m) / n_obs_test_m).astype(np.float64)
        fc_test_h  = ((sum_total_h - sum_train_h) / n_obs_test_h).astype(np.float64)
        # Solve FGW on the train side
        Cm, Ch = build_costs_from_train_fc(fc_train_m, fc_train_h, costs_static,
                                            use_SC=cfg["use_SC"])
        M = build_M(cfg["M"], costs_static, idx_m, idx_h)
        pi, _ = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein(
            M=M, C1=Cm, C2=Ch, p=p, alpha=0.5, epsilon=5e-3,
            max_iter=25, tol=1e-5, log=True,
        )
        # Evaluate on TRAIN (sanity — should match the all-subjects baseline)
        train_metrics = fc_translation_quality(
            pi.astype(np.float64),
            fc_train_m.astype(np.float64),
            fc_train_h.astype(np.float64),
            network_labels_h=net_h_arr,
        )
        # Evaluate on TEST (the real generalisation test)
        test_metrics = fc_translation_quality(
            pi.astype(np.float64),
            fc_test_m.astype(np.float64),
            fc_test_h.astype(np.float64),
            network_labels_h=net_h_arr,
        )

        elapsed = time.time() - t
        fold_results[fold_key] = {
            "n_train_m":           int(len(train_m)),
            "n_train_h":           int(len(train_h)),
            "n_test_m":            int(len(test_m)),
            "n_test_h":            int(len(test_h)),
            "train_r_overall":     train_metrics["pearson_r_overall"],
            "train_r_within_net":  train_metrics.get("pearson_r_within_net", float("nan")),
            "train_r_cross_net":   train_metrics.get("pearson_r_cross_net", float("nan")),
            "test_r_overall":      test_metrics["pearson_r_overall"],
            "test_r_within_net":   test_metrics.get("pearson_r_within_net", float("nan")),
            "test_r_cross_net":    test_metrics.get("pearson_r_cross_net", float("nan")),
            "test_minus_train":    test_metrics["pearson_r_overall"]
                                     - train_metrics["pearson_r_overall"],
            "n_human_nodes_kept":  test_metrics.get("n_human_nodes_kept", n_h),
            "elapsed":             round(elapsed, 1),
        }
        state[cfg_key] = fold_results
        cache_path.write_text(json.dumps(state, indent=2, default=float))
        r = fold_results[fold_key]
        print(f"     train r={r['train_r_overall']:.3f} (within={r['train_r_within_net']:.3f}/cross={r['train_r_cross_net']:.3f})")
        print(f"     test  r={r['test_r_overall']:.3f} (within={r['test_r_within_net']:.3f}/cross={r['test_r_cross_net']:.3f})")
        print(f"     gap (test−train) = {r['test_minus_train']:+.3f}    ({elapsed:.0f}s)", flush=True)

    # Aggregate across folds
    fr = list(fold_results.values())
    if len(fr) >= 2:
        train_r = np.array([f["train_r_overall"] for f in fr])
        test_r  = np.array([f["test_r_overall"]  for f in fr])
        gap     = np.array([f["test_minus_train"] for f in fr])
        print(f"\nSUMMARY across {len(fr)} folds:")
        print(f"  train r:  {train_r.mean():.3f} ± {train_r.std():.3f}")
        print(f"  test  r:  {test_r.mean():.3f} ± {test_r.std():.3f}")
        print(f"  gap:      {gap.mean():+.3f} ± {gap.std():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",   default="fc_plus_SC", choices=list(CONFIGS.keys()))
    ap.add_argument("--k-folds",  type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--seed",     type=int, default=0)
    ap.add_argument("--recompute", action="store_true")
    main(ap.parse_args())
