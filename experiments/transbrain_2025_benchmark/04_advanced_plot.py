"""Visualise the OTTER × TransBrain advanced comparison (3 panels)."""
# Retired. This script read transbrain_2025_advanced.json["cycle_consistency"], which scored
# TransBrain on a different region set than OTTER, and that block is no longer in the log. The
# reported comparison uses transbrain_roundtrip_maps.json, which scores both methods on the same
# 52 mouse regions. The per-panel scripts (make_panelA/B/C/D_*.py) supersede this composite.
raise SystemExit(__doc__.splitlines()[0] if __doc__ else "retired; see header")
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def main():
    j = json.loads((ROOT / "outputs/logs/transbrain_2025_advanced.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9))

    # --- Panel 1: bidirectional cycle-consistency -------------------------
    ax = axes[0]
    cyc = j["cycle_consistency"]
    names = list(cyc)
    x = np.arange(len(names))
    otter = [cyc[n]["otter"] for n in names]
    tb = [cyc[n]["transbrain"] for n in names]
    ax.bar(x - 0.2, otter, 0.4, color="#2a9d8f", label="OTTER")
    ax.bar(x + 0.2, tb, 0.4, color="#e76f51", label="TransBrain")
    for i, v in enumerate(otter):
        ax.text(i - 0.2, v + 0.012, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(tb):
        ax.text(i + 0.2, v + 0.012, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("cycle-consistency  (mouse→human→mouse, Pearson r)")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("1 · Bidirectional cycle-consistency\n"
                 "OTTER's round-trip is more self-consistent. \n"
                 "a ground-truth-free metric", fontsize=9.5)

    # --- Panel 2: optogenetic AI circuit → human cognition ----------------
    ax = axes[1]
    opto = j.get("optogenetic", {})
    if opto:
        top_h = opto["otter_top10"]
        tb_set = set(opto["transbrain_top10"])
        sc = opto["otter_scores"]
        vals = [sc[t] for t in top_h]
        cols = ["#8ab17d" if t in tb_set else "#264653" for t in top_h]
        y = np.arange(len(top_h))[::-1]
        ax.barh(y, vals, color=cols)
        ax.set_yticks(y)
        ax.set_yticklabels([t[:22] for t in top_h], fontsize=8)
        ax.set_xlabel("Neurosynth term score (r)")
        ax.set_title(f"2 · Mouse AI optogenetic circuit → human cognition\n"
                     f"OTTER's top-10 terms (green = shared with TransBrain,\n"
                     f"{opto['top10_overlap']}/10 overlap)", fontsize=9.5)
    else:
        ax.axis("off")
        ax.set_title("2 · Optogenetic decode. Neurosynth maps unavailable")

    # --- Panel 3: trust-stratified agreement (flat result) ---------
    ax = axes[2]
    ts = j["trust_stratified"]
    tm = ts["tier_mean_topdist_mm"]
    order = ["anchored_and_validated", "anchored_only", "validated_only",
             "structural", "low_evidence"]
    order = [t for t in order if t in tm]
    vals = [tm[t] for t in order]
    ax.bar(range(len(order)), vals, color="#9c9c9c")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.6, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([t.replace("_", "\n") for t in order], fontsize=7)
    ax.set_ylabel("OTTER↔TransBrain top-region distance (mm)")
    ax.set_title(f"3 · Agreement vs OTTER's trust tiers\n"
                 f"flat (r = {ts['topdist_vs_trust_pearson']:+.2f}), the two methods\n"
                 f"differ for reasons unrelated to OTTER's confidence", fontsize=9.5)

    plt.suptitle("OTTER × TransBrain 2025, advanced comparison: cycle-consistency, "
                 "circuit annotation, and where the methods differ", fontsize=12, y=1.04)
    plt.tight_layout()
    out = ROOT / "outputs" / "figures" / "transbrain_2025_advanced.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
