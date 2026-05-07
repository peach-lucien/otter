"""Pipeline step 07 — build comparison artefacts + interactive 3D viewer.

Two reporting steps in one script:

1. **Comparison table & figures** (default): Reads every results JSON in
   ``outputs/logs/`` and produces:
     outputs/comparison/comprehensive_table.csv     wide CSV: configs × headline metrics
     outputs/comparison/per_network_top1.csv        long CSV: configs × networks → top1
     outputs/comparison/comparison_summary.md       markdown summary
     outputs/figures/13_comprehensive_comparison.png  4-panel headline bars
     outputs/figures/14_config_x_network_heatmap.png  full heatmap

2. **Interactive 3D viewer** (``--viewer``): Loads a saved π from
   ``outputs/coupling/``, builds the embedded JSON payload, and writes a
   self-contained HTML viewer to ``outputs/viewer/index.html``. Pass
   ``--pi-file`` to choose a non-default π.

Usage:
    python pipeline/07_build_artefacts.py                     # comparison report only
    python pipeline/07_build_artefacts.py --viewer            # also build viewer
    python pipeline/07_build_artefacts.py --viewer-only       # viewer only
    python pipeline/07_build_artefacts.py --viewer --pi-file pi_fc_plus_SC_with_M1_hippo.npy

This is a pure reporting step — it doesn't run any solves. Re-run any time
you want a fresh table after `pipeline/05_evaluate.py` or a fresh viewer
after `pipeline/04_solve_production.py`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from homer.data import load_cached                                      # noqa: E402
from homer.viz.reports import (                                          # noqa: E402
    build_comparison_table, render_summary_md,
    make_comparison_bars_figure, make_per_network_heatmap_figure,
)
from homer.viz.viewer import write_viewer                                # noqa: E402

ANN  = ROOT / "outputs" / "anndata"
COUP = ROOT / "outputs" / "coupling"
LOG  = ROOT / "outputs" / "logs"
CMP  = ROOT / "outputs" / "comparison"; CMP.mkdir(parents=True, exist_ok=True)
FIG  = ROOT / "outputs" / "figures";    FIG.mkdir(parents=True, exist_ok=True)
VIEW = ROOT / "outputs" / "viewer";     VIEW.mkdir(parents=True, exist_ok=True)


def build_comparison():
    print(f"Loading results from {LOG}")
    wide_df, long_df, null_z, bootstrap = build_comparison_table(LOG)
    print(f"  built {len(wide_df)} config rows × {wide_df.shape[1]} columns")

    print("Null z-scores (production = fc_plus_SC):")
    for k, v in null_z.items():
        print(f"  {k:20s}: real={v['real_top1']:.0%}  null={v['null_mean']:.0%}±{v['null_std']:.0%}  z={v['z_score']:+.1f}")

    csv_path  = CMP / "comprehensive_table.csv"
    long_path = CMP / "per_network_top1.csv"
    md_path   = CMP / "comparison_summary.md"
    fig_path  = FIG / "13_comprehensive_comparison.png"
    fig2_path = FIG / "14_config_x_network_heatmap.png"

    wide_df.to_csv(csv_path, index=False, float_format="%.4f")
    long_df.to_csv(long_path, index=False, float_format="%.4f")
    md_path.write_text(render_summary_md(wide_df, long_df, null_z, bootstrap))

    fig = make_comparison_bars_figure(wide_df)
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")

    fig2 = make_per_network_heatmap_figure(long_df)
    fig2.savefig(fig2_path, dpi=120, bbox_inches="tight")

    print("\ncomparison artefacts saved:")
    for p in (csv_path, long_path, md_path, fig_path, fig2_path):
        print(f"  → {p}")


def build_viewer(pi_file: str, top_k: int):
    pi_path = COUP / pi_file
    print(f"\nLoading π from {pi_path}")
    pi = np.load(pi_path).astype(np.float64)
    print(f"  pi shape={pi.shape}, sum={pi.sum():.3f}")

    print("Loading anndata + anchors...")
    H, _ = load_cached("human", cache_dir=ANN)
    M, _ = load_cached("mouse", cache_dir=ANN)

    pi_label = pi_file.replace("pi_", "").replace(".npy", "")
    print(f"Building viewer (top_k={top_k})...")
    json_path, html_path = write_viewer(
        pi, mouse_ad=M, human_ad=H, output_dir=VIEW,
        top_k=top_k, pi_label=pi_label, pi_source=pi_file,
    )
    print(f"saved → {json_path}  ({json_path.stat().st_size / 1024:.1f} KB)")
    print(f"saved → {html_path}  ({html_path.stat().st_size / 1024:.1f} KB)")
    print(f"\nOpen the viewer by double-clicking {html_path}")


def main(args):
    if not args.viewer_only:
        build_comparison()
    if args.viewer or args.viewer_only:
        build_viewer(args.pi_file, args.top_k)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true",
                    help="also build the interactive 3D viewer")
    ap.add_argument("--viewer-only", action="store_true",
                    help="skip the comparison table; build viewer only")
    ap.add_argument("--pi-file", default="pi_fc_plus_SC.npy",
                    help="which π in outputs/coupling/ to visualise (with --viewer)")
    ap.add_argument("--top-k", type=int, default=30,
                    help="how many top partners to embed per node (with --viewer)")
    main(ap.parse_args())
