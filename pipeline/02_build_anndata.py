"""Build & cache AnnData for both species.

Run from homer/ root:
    PYTHONPATH=src python scripts/build_anndata.py [--species human|mouse|both]

Outputs:
    outputs/anndata/{species}.h5ad
    outputs/anndata/{species}.fc.npy        (n_nodes, n_nodes, n_subjects) float32
    outputs/anndata/{species}.voxels.npz    ragged voxel index lists
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from homer.data import build_anndata  # noqa: E402

CACHE = ROOT / "outputs" / "anndata"
LOGS = ROOT / "outputs" / "logs"


def main(species_arg: str) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    species_list = ["human", "mouse"] if species_arg == "both" else [species_arg]

    summary: dict[str, dict] = {}
    for sp in species_list:
        t0 = time.time()
        print(f"[{sp}] building AnnData...", flush=True)
        A = build_anndata(sp, cache_dir=CACHE, overwrite=True)
        elapsed = time.time() - t0
        n_obs = A.uns["fc_n_obs"]
        full_cov = int((n_obs == A.uns["n_subjects"]).sum())
        partial  = int(((n_obs > 0) & (n_obs < A.uns["n_subjects"])).sum())
        zero_cov = int((n_obs == 0).sum())
        summary[sp] = {
            "elapsed_sec":      round(elapsed, 1),
            "n_subjects":       int(A.uns["n_subjects"]),
            "n_nodes":          int(A.uns["n_nodes"]),
            "garin_anchors":    int(A.var["garin_anchor"].sum()),
            "fc_mean_shape":    list(A.uns["fc_mean"].shape),
            "fc_diag_mean":     float(np.nanmean(np.diag(A.uns["fc_mean"]))),
            "fc_n_obs_min":     int(n_obs.min()),
            "fc_n_obs_max":     int(n_obs.max()),
            "fc_cells_full":    full_cov,
            "fc_cells_partial": partial,
            "fc_cells_zero":    zero_cov,
            "h5ad_path":        str(CACHE / f"{sp}.h5ad"),
        }
        print(
            f"[{sp}] done in {elapsed:.1f}s, n_subj={A.uns['n_subjects']} "
            f"n_nodes={A.uns['n_nodes']} fc_n_obs∈[{n_obs.min()},{n_obs.max()}] "
            f"zero-coverage cells={zero_cov}",
            flush=True,
        )

    log_path = LOGS / "build_anndata.json"
    log_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary written → {log_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--species", choices=["human", "mouse", "both"], default="both")
    main(p.parse_args().species)
