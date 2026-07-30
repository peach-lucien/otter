#!/usr/bin/env python
"""19_region_hyperparam.py

Question (Section-1 framing): is a SINGLE optimal transport coupling optimal
for translating EVERY brain region, or do different regions want different
hyperparameters?  We sweep two knobs of the anchor-free MultimodalFGW coupling:

    - entropic temperature  epsilon  in {0.005, 0.02, 0.05, 0.1, 0.2}
    - spatial weight        xyz_weight in {0.0, 0.25, 0.5, 1.0}   (at eps=5e-3)

and score every coupling against the EXTERNAL Beauchamp mouse->human benchmark
(19 scorable homologous region pairs).  Anchors are deliberately OFF
(region_anchors=[]) because supervised anchors are pinned (mass=1.0) at every
epsilon and cannot discriminate settings.

For each Beauchamp pair we record top1 / mean_mass_in_region at every setting,
find the per-pair-best epsilon and xyz_weight, and quantify how much regions
disagree.  We also compute an ORACLE upper bound (each region uses its own best
epsilon) vs the single global-best epsilon.

Writes: outputs/logs/section5_region_hyperparam.json
"""
import sys, os, json, time
sys.path.insert(0, 'src')
sys.path.insert(0, 'experiments/section5_coverage_rigor')
import numpy as np
from beauchamp_scorer import BeauchampScorer

EPS_GRID = [0.005, 0.02, 0.05, 0.1, 0.2]
XYZW_GRID = [0.0, 0.25, 0.5, 1.0]
BASE_EPS_FOR_XYZ = 0.005

BASE = dict(use_sc=True, sc_weight=0.3, fc_weight=0.7, xyz_weight=0.5,
            lam_anchor=1.0, alpha=0.5)


def refit(epsilon, xyz_weight, out_path):
    """Fit an anchor-free MultimodalFGW coupling and cache it."""
    from homer.data import load_cached
    from homer.models import MultimodalFGW
    M, _ = load_cached('mouse', cache_dir='outputs/anndata')
    H, _ = load_cached('human', cache_dir='outputs/anndata')
    costs = np.load('outputs/anndata/full_costs.npz')
    cfg = dict(BASE); cfg['xyz_weight'] = xyz_weight
    m = MultimodalFGW(epsilon=epsilon, **cfg)
    m.fit(M, H, Cm_SC=costs['Cm_SC'], Ch_SC=costs['Ch_SC'], region_anchors=[])
    np.save(out_path, m.pi.astype(np.float64))
    return m.pi


def get_eps_pi(eps):
    p = f'/var/tmp/pi_eps_{eps:.3f}.npy'
    if not os.path.exists(p):
        print(f'  refit eps={eps}')
        refit(eps, BASE['xyz_weight'], p)
    return np.load(p)


def get_xyzw_pi(w):
    p = f'/var/tmp/pi_xyzw_{w}.npy'
    if not os.path.exists(p):
        print(f'  refit xyz_weight={w}')
        refit(BASE_EPS_FOR_XYZ, w, p)
    return np.load(p)


def best_setting(pairmap, grid):
    """best grid value by top1, tie-break by mean_mass."""
    best_g, best_v, best_mm = None, -1.0, -1.0
    for g in grid:
        v = pairmap[g]['top1']; mm = pairmap[g]['mean_mass']
        if v > best_v + 1e-12 or (abs(v - best_v) <= 1e-12 and mm > best_mm + 1e-12):
            best_g, best_v, best_mm = g, v, mm
    return best_g


def main():
    t0 = time.time()
    sc = BeauchampScorer()

    from homer.data import load_pi
    prod = sc.score(load_pi())
    prod_agg_top1 = prod['__aggregate__']['top1']

    # ---- 1. EPSILON family --------------------------------------------------
    eps_scores = {}
    eps_pair = {}
    for eps in EPS_GRID:
        r = sc.score(get_eps_pi(eps))
        eps_scores[eps] = r['__aggregate__']
        for pair, d in r.items():
            if pair.startswith('__'):
                continue
            eps_pair.setdefault(pair, {})[eps] = {
                'top1': d['top1'], 'mean_mass': d['mean_mass_in_region']}

    pairs = sorted(eps_pair.keys())
    eps_best = {p: best_setting(eps_pair[p], EPS_GRID) for p in pairs}

    # ---- 2. XYZ-WEIGHT family (fixed eps=5e-3) ------------------------------
    xyzw_scores = {}
    xyzw_pair = {}
    for w in XYZW_GRID:
        r = sc.score(get_xyzw_pi(w))
        xyzw_scores[w] = r['__aggregate__']
        for pair, d in r.items():
            if pair.startswith('__'):
                continue
            xyzw_pair.setdefault(pair, {})[w] = {
                'top1': d['top1'], 'mean_mass': d['mean_mass_in_region']}

    xyzw_best = {p: best_setting(xyzw_pair[p], XYZW_GRID) for p in pairs}

    # ---- 3. HETEROGENEITY + ORACLE -----------------------------------------
    global_best_eps = max(EPS_GRID, key=lambda e: eps_scores[e]['top1'])
    global_best_xyzw = max(XYZW_GRID, key=lambda w: xyzw_scores[w]['top1'])

    rref = sc.score(get_eps_pi(EPS_GRID[0]))
    n_parc = {p: rref[p]['n_mouse_parcels'] for p in pairs}
    tot = sum(n_parc.values())

    def agg_from(pairmap_best, pair_data):
        num = 0.0
        for p in pairs:
            g = pairmap_best[p]
            num += pair_data[p][g]['top1'] * n_parc[p]
        return num / tot

    oracle_eps_top1 = agg_from(eps_best, eps_pair)
    global_eps_top1 = eps_scores[global_best_eps]['top1']
    oracle_xyzw_top1 = agg_from(xyzw_best, xyzw_pair)
    global_xyzw_top1 = xyzw_scores[global_best_xyzw]['top1']

    eps_gain = {}
    for p in pairs:
        vals = [eps_pair[p][e]['top1'] for e in EPS_GRID]
        eps_gain[p] = {'best': max(vals), 'worst': min(vals),
                       'gain': max(vals) - min(vals), 'best_eps': eps_best[p]}
    xyzw_gain = {}
    for p in pairs:
        vals = [xyzw_pair[p][w]['top1'] for w in XYZW_GRID]
        xyzw_gain[p] = {'best': max(vals), 'worst': min(vals),
                        'gain': max(vals) - min(vals), 'best_xyzw': xyzw_best[p]}

    eps_best_counts = {}
    for p in pairs:
        k = str(eps_best[p]); eps_best_counts[k] = eps_best_counts.get(k, 0) + 1
    xyzw_best_counts = {}
    for p in pairs:
        k = str(xyzw_best[p]); xyzw_best_counts[k] = xyzw_best_counts.get(k, 0) + 1

    log = {
        'meta': {
            'base_config': BASE, 'eps_grid': EPS_GRID, 'xyzw_grid': XYZW_GRID,
            'xyz_family_fixed_eps': BASE_EPS_FOR_XYZ,
            'region_anchors': [], 'n_pairs': len(pairs),
            'production_agg_top1': prod_agg_top1,
            'note': 'anchor-free couplings; production shown only as scorer sanity ref',
            'runtime_s': None,
        },
        'epsilon': {
            'aggregate_top1_by_eps': {str(e): eps_scores[e]['top1'] for e in EPS_GRID},
            'aggregate_mean_mass_by_eps': {str(e): eps_scores[e]['mean_mass_in_region'] for e in EPS_GRID},
            'global_best_eps': global_best_eps,
            'per_pair': {p: {str(e): eps_pair[p][e] for e in EPS_GRID} for p in pairs},
            'per_pair_best_eps': {p: eps_best[p] for p in pairs},
            'best_eps_counts': eps_best_counts,
            'per_pair_gain': eps_gain,
        },
        'xyz_weight': {
            'aggregate_top1_by_xyzw': {str(w): xyzw_scores[w]['top1'] for w in XYZW_GRID},
            'aggregate_mean_mass_by_xyzw': {str(w): xyzw_scores[w]['mean_mass_in_region'] for w in XYZW_GRID},
            'global_best_xyzw': global_best_xyzw,
            'per_pair': {p: {str(w): xyzw_pair[p][w] for w in XYZW_GRID} for p in pairs},
            'per_pair_best_xyzw': {p: xyzw_best[p] for p in pairs},
            'best_xyzw_counts': xyzw_best_counts,
            'per_pair_gain': xyzw_gain,
        },
        'oracle': {
            'n_pairs_with_eps_top1_variation': sum(1 for p in pairs if eps_gain[p]['gain'] > 1e-9),
            'n_pairs_with_xyzw_top1_variation': sum(1 for p in pairs if xyzw_gain[p]['gain'] > 1e-9),
            'n_distinct_best_eps': len(set(eps_best.values())),
            'n_distinct_best_xyzw': len(set(xyzw_best.values())),
            'caveat': ('per_pair_best_eps counts are tie-broken on mean_mass for the many pairs whose top1 is flat across eps; only the *_variation counts reflect pairs where top1 actually moves.'),
            'oracle_eps_top1': oracle_eps_top1,
            'global_best_eps_top1': global_eps_top1,
            'eps_oracle_uplift': oracle_eps_top1 - global_eps_top1,
            'oracle_xyzw_top1': oracle_xyzw_top1,
            'global_best_xyzw_top1': global_xyzw_top1,
            'xyzw_oracle_uplift': oracle_xyzw_top1 - global_xyzw_top1,
        },
    }
    log['meta']['runtime_s'] = time.time() - t0

    os.makedirs('outputs/logs', exist_ok=True)
    with open('outputs/logs/section5_region_hyperparam.json', 'w') as f:
        json.dump(log, f, indent=2)

    print('\n=== EPSILON aggregate top1 ===')
    for e in EPS_GRID:
        print(f'  eps={e:<6} top1={eps_scores[e]["top1"]:.4f}  mean_mass={eps_scores[e]["mean_mass_in_region"]:.4f}')
    print(f'  global best eps = {global_best_eps}')
    print('\n=== XYZ-WEIGHT aggregate top1 (eps=5e-3) ===')
    for w in XYZW_GRID:
        print(f'  w={w:<5} top1={xyzw_scores[w]["top1"]:.4f}  mean_mass={xyzw_scores[w]["mean_mass_in_region"]:.4f}')
    print(f'  global best xyzw = {global_best_xyzw}')
    print('\n=== HETEROGENEITY ===')
    print(f'  distinct best-eps values across {len(pairs)} pairs: {len(set(eps_best.values()))} -> {eps_best_counts}')
    print(f'  distinct best-xyzw values: {len(set(xyzw_best.values()))} -> {xyzw_best_counts}')
    print('\n  per-pair best eps / best xyzw / eps-gain:')
    for p in pairs:
        print(f'    {p:<48} eps*={eps_best[p]:<6} xyzw*={xyzw_best[p]:<5} '
              f'epsgain={eps_gain[p]["gain"]:.3f} ({eps_gain[p]["worst"]:.3f}->{eps_gain[p]["best"]:.3f})')
    print('\n=== ORACLE ===')
    print(f'  eps  oracle={oracle_eps_top1:.4f} vs global-best={global_eps_top1:.4f} '
          f'(uplift {oracle_eps_top1-global_eps_top1:+.4f})')
    print(f'  xyzw oracle={oracle_xyzw_top1:.4f} vs global-best={global_xyzw_top1:.4f} '
          f'(uplift {oracle_xyzw_top1-global_xyzw_top1:+.4f})')
    print(f'\n  runtime {log["meta"]["runtime_s"]:.1f}s -> outputs/logs/section5_region_hyperparam.json')


if __name__ == '__main__':
    main()
