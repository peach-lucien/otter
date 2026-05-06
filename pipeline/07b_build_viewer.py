"""Pipeline step 07b — build the standalone 3D HTML viewer.

Loads a saved π from outputs/coupling/, builds the embedded JSON payload,
and writes the self-contained HTML viewer to outputs/viewer/index.html.

The heavy lifting (data prep + HTML template) lives in :mod:`homer.viz.viewer`.

Usage:
    python pipeline/07b_build_viewer.py
    python pipeline/07b_build_viewer.py --pi-file pi_baseline_fc_only.npy --top-k 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from homer.data import load_cached                # noqa: E402
from homer.viz.viewer import write_viewer         # noqa: E402

ANN  = ROOT / "outputs" / "anndata"
COUP = ROOT / "outputs" / "coupling"
VIEW = ROOT / "outputs" / "viewer"; VIEW.mkdir(parents=True, exist_ok=True)


def main(args):
    pi_path = COUP / args.pi_file
    print(f"Loading π from {pi_path}")
    pi = np.load(pi_path).astype(np.float64)
    print(f"  pi shape={pi.shape}, sum={pi.sum():.3f}")

    print("Loading anndata + anchors...")
    H, _ = load_cached("human", cache_dir=ANN)
    M, _ = load_cached("mouse", cache_dir=ANN)

    pi_label = args.pi_file.replace("pi_", "").replace(".npy", "")
    print(f"Building viewer (top_k={args.top_k})...")
    json_path, html_path = write_viewer(
        pi, mouse_ad=M, human_ad=H, output_dir=VIEW,
        top_k=args.top_k, pi_label=pi_label, pi_source=args.pi_file,
    )
    print(f"saved → {json_path}  ({json_path.stat().st_size / 1024:.1f} KB)")
    print(f"saved → {html_path}  ({html_path.stat().st_size / 1024:.1f} KB)")
    print(f"\nOpen the viewer by double-clicking {html_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi-file", default="pi_fc_plus_SC.npy",
                    help="which π in outputs/coupling/ to visualise")
    ap.add_argument("--top-k",   type=int, default=30,
                    help="how many top partners to embed per node")
    main(ap.parse_args())
