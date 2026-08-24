#!/usr/bin/env python3
"""Component-by-component correspondence of the two connectomes' diffusion embeddings.

Computes the first four diffusion components of each species' functional connectome, routes every
mouse component through each arm of the ablation ladder, and records the absolute Pearson
correlation against every human component.

Two cells of that matrix are reported per arm:

  myelin_selected_r  the cell whose component pair is fixed before the coupling is consulted, by
                     taking the component in each species that best tracks that species' own
                     T1w:T2w map
  max_r              the largest cell in the matrix, with its indices in argmax

Reads the per-arm couplings written by 02_ablation_ladder.py to
outputs/coupling/pi_ladder_<arm>.npy. Couplings are loaded, not refitted.

Run:
    cd otter && PYTHONPATH=src python experiments/margulies_2016_principal_gradient/04_gradient_components.py --check
    cd otter && PYTHONPATH=src python experiments/margulies_2016_principal_gradient/04_gradient_components.py

Writes outputs/logs/out_c1_gradient.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from otter.data import load_cached                                   # noqa: E402

COUPLINGS = ROOT / "outputs" / "coupling"
LOGS = ROOT / "outputs" / "logs"
OUT = LOGS / "out_c1_gradient.json"

# Tolerance for comparing against the committed log. The Pearson-based statistics reproduce to
# machine precision; the Spearman ones can move in the sixth decimal place across SciPy versions,
# because the rank correlation is computed by a different internal path in different releases and
# this repository pins floors rather than exact versions. An absolute tolerance of 1e-5 on a
# correlation coefficient sits well above that and well below any difference that would indicate
# the analysis itself had changed.
TOL_ABS = 1e-5
TOL_REL = 1e-9


N_COMP = 4
TOP_PCT = 10.0

# The five arms scored here, matching 03_downstream_by_arm.py. The ladder's sixth arm, 3_+anchors,
# is not read by this script.
ARMS = {
    "4_+packs_CANONICAL":          "canonical (alpha=0.5)",
    "6_NOCONN_spatial+anch+packs": "alpha=0 (NO connectivity)",
    "2_+spatial":                  "conn+spatial, no curation",
    "5_NOCONN_spatial_only":       "alpha=0 spatial only",
    "1_connectivity_only":         "connectivity only",
}

WHAT = ("Component-by-component correspondence of the leading diffusion embeddings of the two "
        "connectomes, for each arm of the ablation ladder. The first four components are computed "
        "per species, every mouse component is routed through the arm's coupling, and the absolute "
        "Pearson correlation against every human component is recorded. myelin_selected_r is the "
        "cell whose component pair is fixed by each species' own T1w:T2w map; max_r is the largest "
        "cell in the matrix. Couplings are read from outputs/coupling/pi_ladder_<arm>.npy and are "
        "not refitted.")


def _load(path: Path, name: str):
    spec = spec_from_file_location(name, path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rho(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[m], b[m]).statistic)


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
    elif isinstance(theirs, list) and isinstance(mine, list):
        if len(theirs) != len(mine):
            out.append(f"{path}: length {len(mine)} against {len(theirs)}")
        else:
            for i, (t, m) in enumerate(zip(theirs, mine)):
                out += differences(t, m, f"{path}[{i}]")
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

    mg = _load(Path(__file__).resolve().parent / "01_gradient_validation.py", "mg")
    fu = _load(ROOT / "experiments/fulcher_2019_multimodal_gradient/01_gradient_validation.py",
               "fu")

    M, _ = load_cached("mouse", cache_dir=str(ROOT / "outputs/anndata"))
    H, _ = load_cached("human", cache_dir=str(ROOT / "outputs/anndata"))

    # Reference map on each side is that species' own T1w:T2w.
    human_ref = np.asarray(json.loads((LOGS / "buckner_krienen_2013_tethering.json").read_text())
                           ["myelin_per_parcel"], float)
    t1t2 = fu.load_mouse_t1t2()
    mouse_ref = np.array([t1t2.get(a, np.nan) for a in fu.load_mouse_parcel_acronyms()], float)

    cm = mg.diffusion_components(np.asarray(M.uns["fc_mean"], float),
                                 top_pct=TOP_PCT, n_comp=N_COMP)
    ch = mg.diffusion_components(np.asarray(H.uns["fc_mean"], float),
                                 top_pct=TOP_PCT, n_comp=N_COMP)

    mouse_vs = [rho(c, mouse_ref) for c in cm]
    human_vs = [rho(c, human_ref) for c in ch]
    print("mouse comps vs mouse T1w:T2w :", [round(x, 3) for x in mouse_vs])
    print("human comps vs human T1w:T2w :", [round(x, 3) for x in human_vs])

    sel_i = int(np.argmax(np.abs(mouse_vs)))
    sel_j = int(np.argmax(np.abs(human_vs)))
    print(f"myelin-selected component pair: mouse {sel_i + 1}, human {sel_j + 1}")

    out = {"_what": WHAT,
           "mouse_comp_vs_mouse_myelin": mouse_vs,
           "human_comp_vs_human_myelin": human_vs,
           "_couplings": {},
           "arms": {}}

    for arm, label in ARMS.items():
        path = COUPLINGS / f"pi_ladder_{arm}.npy"
        pi = np.load(path).astype(np.float64)
        out["_couplings"][arm] = {"file": path.name, "sha256": sha256(path)}
        Mx = np.zeros((N_COMP, N_COMP))
        for i in range(N_COMP):
            routed = mg.route_normalized(cm[i], pi)
            for j in range(N_COMP):
                m = np.isfinite(routed) & np.isfinite(ch[j])
                Mx[i, j] = abs(pearsonr(routed[m], ch[j][m])[0])
        argmax = [int(x) + 1 for x in np.unravel_index(Mx.argmax(), Mx.shape)]
        out["arms"][arm] = {"label": label,
                            "matrix_abs_r": Mx.tolist(),
                            "myelin_selected_pair": [sel_i + 1, sel_j + 1],
                            "myelin_selected_r": float(Mx[sel_i, sel_j]),
                            "max_r": float(Mx.max()),
                            "argmax": argmax}
        print(f"\n{label}")
        print("   routed mouse comp (rows) x human comp (cols), |r|:")
        for i in range(N_COMP):
            print("    m%d " % (i + 1) + " ".join(f"{Mx[i, j]:.3f}" for j in range(N_COMP)))
        print(f"   myelin-selected pair m{sel_i + 1}xh{sel_j + 1} -> |r|={Mx[sel_i, sel_j]:.3f}"
              f"   |   overall max |r|={Mx.max():.3f} at m{argmax[0]}xh{argmax[1]}")

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
