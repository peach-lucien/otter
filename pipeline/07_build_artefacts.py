"""Pipeline step 07 — build the comprehensive comparison artefacts.

Reads every results JSON in ``outputs/logs/`` and produces:

  outputs/comparison/comprehensive_table.csv     wide CSV: configs × headline metrics
  outputs/comparison/per_network_top1.csv        long CSV: configs × networks → top1
  outputs/comparison/comparison_summary.md       markdown summary
  outputs/figures/13_comprehensive_comparison.png  4-panel headline bars
  outputs/figures/14_config_x_network_heatmap.png  full heatmap

This is a pure reporting step — it does not run any solves. Re-run any time
you want a fresh table after `pipeline/05_evaluate.py`.

The heavy lifting lives in :mod:`homer.viz.reports`.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from homer.viz.reports import (                                       # noqa: E402
    build_comparison_table, render_summary_md,
    make_comparison_bars_figure, make_per_network_heatmap_figure,
)

LOG = ROOT / "outputs" / "logs"
CMP = ROOT / "outputs" / "comparison"; CMP.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "outputs" / "figures";    FIG.mkdir(parents=True, exist_ok=True)


def main():
    print(f"Loading results from {LOG}")
    wide_df, long_df, null_z, bootstrap = build_comparison_table(LOG)
    print(f"Built {len(wide_df)} config rows × {wide_df.shape[1]} columns")

    print("Null z-scores (production = fc_plus_SC):")
    for k, v in null_z.items():
        print(f"  {k:20s}: real={v['real_top1']:.0%}  null={v['null_mean']:.0%}±{v['null_std']:.0%}  z={v['z_score']:+.1f}")

    csv_path = CMP / "comprehensive_table.csv"
    long_path = CMP / "per_network_top1.csv"
    md_path  = CMP / "comparison_summary.md"
    fig_path  = FIG / "13_comprehensive_comparison.png"
    fig2_path = FIG / "14_config_x_network_heatmap.png"

    wide_df.to_csv(csv_path, index=False, float_format="%.4f")
    long_df.to_csv(long_path, index=False, float_format="%.4f")
    md_path.write_text(render_summary_md(wide_df, long_df, null_z, bootstrap))

    fig = make_comparison_bars_figure(wide_df)
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")

    fig2 = make_per_network_heatmap_figure(long_df)
    fig2.savefig(fig2_path, dpi=120, bbox_inches="tight")

    print("\nsaved:")
    for p in (csv_path, long_path, md_path, fig_path, fig2_path):
        print(f"  → {p}")


if __name__ == "__main__":
    main()
