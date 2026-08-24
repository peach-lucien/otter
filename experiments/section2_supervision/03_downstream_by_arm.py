#!/usr/bin/env python3
"""Section 5 reconstruction statistics for each arm of the ablation ladder.

Reads the per-arm couplings written by 02_ablation_ladder.py to
outputs/coupling/pi_ladder_<arm>.npy. Couplings are loaded, not refitted.

For each arm, the human connectome is reconstructed through the arm's coupling, once from
functional and once from structural connectivity, and five statistics are reported per modality:

  mean_r     mean row-wise reconstruction accuracy over parcels carrying a myelin value
  vs_iso     Spearman against distance to the nearest mouse parcel
  LR_rel     Spearman between the left and right members of the 200 Schaefer pairs
  vs_exp     Spearman against the Xu 2020 expansion map
  ContB_SD   mean z of the Control-B network minus the mean z of the remaining parcels

Run:
    cd otter && PYTHONPATH=src python experiments/section2_supervision/03_downstream_by_arm.py --check
    cd otter && PYTHONPATH=src python experiments/section2_supervision/03_downstream_by_arm.py

Writes outputs/logs/out_a1c_downstream.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from otter.data import load_cached                                   # noqa: E402

COUPLINGS = ROOT / "outputs" / "coupling"
LOGS = ROOT / "outputs" / "logs"
OUT = LOGS / "out_a1c_downstream.json"

# Tolerance for comparing against the committed log. The Pearson-based statistics reproduce to
# machine precision; the Spearman ones can move in the sixth decimal place across SciPy versions,
# because the rank correlation is computed by a different internal path in different releases and
# this repository pins floors rather than exact versions. An absolute tolerance of 1e-5 on a
# correlation coefficient sits well above that and well below any difference that would indicate
# the analysis itself had changed.
TOL_ABS = 1e-5
TOL_REL = 1e-9


# The five arms scored here. 02_ablation_ladder.py defines a sixth, 3_+anchors, which this script
# does not read.
ARMS = {
    "4_+packs_CANONICAL":          "canonical (alpha=0.5)",
    "6_NOCONN_spatial+anch+packs": "alpha=0 (NO connectivity)",
    "2_+spatial":                  "conn+spatial, no curation",
    "5_NOCONN_spatial_only":       "alpha=0 spatial only",
    "1_connectivity_only":         "connectivity only",
}

WHAT = ("Section 5 reconstruction statistics for each arm of the ablation ladder. The couplings "
        "are read from outputs/coupling/pi_ladder_<arm>.npy as written by 02_ablation_ladder.py "
        "and are not refitted. Reported per arm for functional and structural connectivity: "
        "mean row-wise reconstruction accuracy, its Spearman against distance to the nearest "
        "mouse parcel, the left-right reliability across the 200 Schaefer pairs, its Spearman "
        "against the Xu 2020 expansion map, and the Control-B deficit in standard deviations.")

# The expansion map is stored under either of two key spellings depending on the log version. The
# stored schaefer_ids and map_values are the same under both.
EXPANSION_KEYS = ("Xu2020 macaque→human expansion", "Xu2020 mouse→human expansion")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def row_corr(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-parcel correlation between a reconstructed and an observed connectivity row.

    The parcel's own entry is dropped from both rows. A row is scored only when more than ten
    partners remain finite and neither side is constant.
    """
    n = pred.shape[0]
    out = np.full(n, np.nan)
    for j in range(n):
        a = pred[j].copy()
        b = true[j].copy()
        a[j] = np.nan
        b[j] = np.nan
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 10 and a[ok].std() > 1e-9 and b[ok].std() > 1e-9:
            out[j] = np.corrcoef(a[ok], b[ok])[0, 1]
    return out


def reconstruct(pi: np.ndarray, Cm: np.ndarray, Ch: np.ndarray) -> np.ndarray:
    """Push the mouse connectome through the coupling and score it against the human one."""
    col = pi.sum(0)
    pit = pi / np.maximum(col, 1e-300)
    return row_corr(pit.T @ Cm @ pit, Ch)


def build_stats(cov, mye, iso, nr, net, xu):
    m = np.isfinite(cov) & np.isfinite(mye)
    ids = [k for k in range(1, 201) if (nr == k).any() and (nr == k + 200).any()]
    left = [np.nanmean(cov[nr == k]) for k in ids]
    right = [np.nanmean(cov[nr == k + 200]) for k in ids]
    eids = [k for k in range(1, 401) if (nr == k).any() and k in xu]
    cc = np.array([np.nanmean(cov[nr == k]) for k in eids])
    ev = np.array([xu[k] for k in eids])
    z = (cov[m] - np.nanmean(cov[m])) / np.nanstd(cov[m])
    sel = net[m] == "ContB"
    return dict(mean_r=float(np.nanmean(cov[m])),
                vs_iso=float(spearmanr(cov[m], iso[m]).statistic),
                LR_rel=float(spearmanr(left, right, nan_policy="omit").statistic),
                vs_exp=float(spearmanr(cc, ev, nan_policy="omit").statistic),
                ContB_SD=float(np.nanmean(z[sel]) - np.nanmean(z[~sel])))


def load_context():
    M, _ = load_cached("mouse", cache_dir=str(ROOT / "outputs/anndata"))
    H, _ = load_cached("human", cache_dir=str(ROOT / "outputs/anndata"))
    Mfc = np.asarray(M.uns["fc_mean"], float)
    Hfc = np.asarray(H.uns["fc_mean"], float)
    Msc = np.log1p(np.maximum(np.load(ROOT / "data_external/mouse_sc.npy").astype(float), 0))
    Hsc = np.log1p(np.maximum(np.load(ROOT / "data_external/human_sc.npy").astype(float), 0))
    iso = np.load(ROOT / "outputs/anndata/full_costs.npz")["M_xyz"].min(0)
    mye = np.asarray(json.loads((LOGS / "buckner_krienen_2013_tethering.json").read_text())
                     ["myelin_per_parcel"], float)
    nr = np.asarray(json.loads((ROOT / "data_external/human_sc_meta.json").read_text())
                    ["node_region"], int)
    rows = [l.split("\t") for l in
            (ROOT / "outputs/anndata/_schaefer_order.txt").read_text().splitlines() if l.strip()]
    lut = {int(p[0]): p[1].split("_", 2)[2].split("_")[0] for p in rows}
    net = np.array([lut.get(int(k), "?") for k in nr])

    battery = json.loads((LOGS / "section5_evolution_battery.json").read_text())
    key = next((k for k in EXPANSION_KEYS if k in battery), None)
    if key is None:
        raise SystemExit("no expansion map in section5_evolution_battery.json; looked for %s"
                         % (EXPANSION_KEYS,))
    xu = dict(zip(np.asarray(battery[key]["schaefer_ids"], int),
                  np.asarray(battery[key]["map_values"], float)))
    return Mfc, Hfc, Msc, Hsc, iso, mye, nr, net, xu, key


def differences(theirs, mine, path=""):
    out = []
    if isinstance(theirs, dict) and isinstance(mine, dict):
        for k in sorted(set(theirs) | set(mine)):
            if k.startswith("_"):
                continue
            if k not in theirs:
                out.append(f"{path}/{k}: only in the new run")
            elif k not in mine:
                out.append(f"{path}/{k}: only in the committed log")
            else:
                out += differences(theirs[k], mine[k], f"{path}/{k}")
    elif isinstance(theirs, float) and isinstance(mine, float):
        if not (abs(theirs - mine) <= TOL_ABS + TOL_REL * abs(theirs)):
            out.append(f"{path}: {mine!r} against {theirs!r}")
    elif theirs != mine:
        out.append(f"{path}: {mine!r} against {theirs!r}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="score without writing, and report any drift from the committed log")
    ap.add_argument("--rewrite", action="store_true",
                    help="write even when the values differ from the committed log")
    args = ap.parse_args()

    missing = [a for a in ARMS if not (COUPLINGS / f"pi_ladder_{a}.npy").exists()]
    if missing:
        raise SystemExit(
            "missing arm couplings: %s\nRun 02_ablation_ladder.py first; it writes "
            "outputs/coupling/pi_ladder_<arm>.npy for each rung." % ", ".join(sorted(missing)))

    Mfc, Hfc, Msc, Hsc, iso, mye, nr, net, xu, expansion_key = load_context()
    print("expansion map: %s (%d regions)" % (expansion_key, len(xu)))

    out = {"_what": WHAT, "_expansion_map": expansion_key, "_couplings": {}}
    for arm, label in ARMS.items():
        path = COUPLINGS / f"pi_ladder_{arm}.npy"
        pi = np.load(path).astype(np.float64)
        out["_couplings"][arm] = {"file": path.name, "sha256": sha256(path)}
        fc = build_stats(reconstruct(pi, Mfc, Hfc), mye, iso, nr, net, xu)
        sc = build_stats(reconstruct(pi, Msc, Hsc), mye, iso, nr, net, xu)
        out[arm] = {"label": label, "FC": fc, "SC": sc}
        print(f"{label:30s} FC: mean={fc['mean_r']:+.3f} LRrel={fc['LR_rel']:+.3f} "
              f"vs_exp={fc['vs_exp']:+.3f} ContB={fc['ContB_SD']:+.2f}SD | "
              f"SC: ContB={sc['ContB_SD']:+.2f} vs_exp={sc['vs_exp']:+.3f}", flush=True)

    if OUT.exists():
        committed = json.loads(OUT.read_text())
        drift = differences(committed, out)
        if drift:
            print(f"\nDIFFERS from the committed log in {len(drift)} place(s):", file=sys.stderr)
            for d in drift[:20]:
                print(f"  {d}", file=sys.stderr)
            if len(drift) > 20:
                print(f"  ... and {len(drift) - 20} more", file=sys.stderr)
            if not args.rewrite:
                print("\nNot written; the committed values stand.", file=sys.stderr)
                return 1
            print("\n--rewrite given; replacing the committed log.", file=sys.stderr)
        else:
            print(f"\nreproduces the committed {OUT.name}")

    if not args.check:
        OUT.write_text(json.dumps(out, indent=1, default=float))
        print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
