#!/usr/bin/env python3
"""Build split-half functional connectivity for each species.

Splits each resting-state cohort into two halves of subjects and averages the connectivity within
each half. Refitting the coupling on each half separately tests whether it depends on the particular
subjects it was fitted on. Results section 1 reports the outcome.

The split is drawn from a fixed seed, so the halves are the same on every run. Averaging is done in
column blocks because the per-subject connectivity tensors do not fit in memory. NaN entries are
excluded pairwise rather than zero-filled, and a cell that is missing in every subject of a half
stays NaN.

Writes outputs/splithalf/{species}_splithalf.npz with A, B, idxA, idxB and n_subj.

    conda activate retune
    cd otter && python3 pipeline/02b_build_splithalf_fc.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]                       # .../otter
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT.parent / "data_crossspecies"
OUT = ROOT / "outputs" / "splithalf"

# (species, source .mat, top-level group holding the per-subject connectivity tensor "rr")
COHORTS = [
    ("human", "corrs_human.mat", "h"),
    ("mouse", "corrs_mouse.mat", "m"),
]
SEED = 0


def half_means(path: Path, top: str, *, seed: int = SEED, verbose: bool = True) -> dict:
    """Mean connectivity for each half of the cohort, plus the subject indices used."""
    import h5py

    with h5py.File(path, "r") as fh:
        rr = fh[f"{top}/rr"]
        n_subj, n_node, _ = rr.shape
        if verbose:
            print(f"  {n_subj} subjects, {n_node} nodes", flush=True)

        perm = np.random.default_rng(seed).permutation(n_subj)
        idx = {"A": np.sort(perm[: n_subj // 2]), "B": np.sort(perm[n_subj // 2:])}
        block = rr.chunks[1] if rr.chunks else 256

        means = {}
        for name, subjects in idx.items():
            weight = np.zeros(n_subj, np.float32)
            weight[subjects] = 1.0
            total = np.zeros((n_node, n_node))
            count = np.zeros((n_node, n_node))
            for b in range((n_node + block - 1) // block):
                j0, j1 = b * block, min((b + 1) * block, n_node)
                chunk = rr[:, :, j0:j1].astype(np.float32, copy=False)
                finite = ~np.isnan(chunk)
                filled = np.where(finite, chunk, 0.0)
                total[:, j0:j1] += (weight @ filled.reshape(n_subj, -1)).reshape(n_node, j1 - j0)
                count[:, j0:j1] += (weight @ finite.astype(np.float32).reshape(n_subj, -1)
                                    ).reshape(n_node, j1 - j0)
            mean = (total / np.maximum(count, 1)).astype(np.float32)
            mean[count == 0] = np.nan
            means[name] = mean
            if verbose:
                print(f"  half {name}: n={len(subjects)}, "
                      f"finite fraction {np.isfinite(mean).mean():.3f}", flush=True)

    both = np.isfinite(means["A"]) & np.isfinite(means["B"])
    agreement = float(np.corrcoef(means["A"][both], means["B"][both])[0, 1])
    return {"A": means["A"], "B": means["B"],
            "idxA": idx["A"], "idxB": idx["B"], "n_subj": n_subj,
            "half_half_r": agreement}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true", help="rebuild even if the output exists")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for species, filename, top in COHORTS:
        target = OUT / f"{species}_splithalf.npz"
        if target.exists() and not args.force:
            print(f"{species}: {target.relative_to(ROOT)} exists, skipping (--force to rebuild)")
            continue
        source = DATA / filename
        if not source.exists():
            print(f"{species}: source not found at {source}", file=sys.stderr)
            return 1
        print(f"{species}: reading {source.name}")
        res = half_means(source, top, seed=args.seed)
        np.savez_compressed(target, A=res["A"], B=res["B"],
                            idxA=res["idxA"], idxB=res["idxB"], n_subj=res["n_subj"])
        print(f"  half-half FC agreement r={res['half_half_r']:.4f}")
        print(f"  wrote {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
