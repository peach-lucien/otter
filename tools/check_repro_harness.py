#!/usr/bin/env python3
"""Smoke-test otter.repro before the notebooks are built on top of it.

Run this in the `retune` environment. It does no fitting and takes a couple of minutes. It checks
that the harness imports, that the released coupling is present and hashes to the value the logs
claim, and that every log the manuscript cites exists and records which coupling produced it. A
failure here would otherwise surface later as a broken notebook.

    conda activate retune
    cd otter && python3 tools/check_repro_harness.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                       # .../otter
sys.path.insert(0, str(ROOT / "src"))

# The logs the manuscript is allowed to draw on, per tools/check_manuscript_numbers.py.
CITED = """beauchamp_metric_battery_canonical beauchamp_metric_battery_loro_canonical
transbrain_benchmark_summary coupling_summary_canonical evidence_tiers_canonical fig1_coupling_matrix
out_a2_splithalf out_a1c_downstream out_a1d_robust region_level_eval_canonical
ablation_ladder_battery_canonical out_a1_ladder heldout_three_config_canonical out_a1b_loro
out_g2_regret anchor_recovery_loo_combined_canonical coletta_2020_cross_species_rsn
margulies_2016_gradient published_map_validation fulcher_2019_gradient biccn_contrast_reframe
biccn_composition_from_markers hodge_areal_type_reframe hodge_2019_layer_markers_refined
hodge_markers_like_for_like out_c1_gradient out_c2_nulls transbrain_2025_benchmark
transbrain_bn_distributions transbrain_roundtrip_maps transbrain_bn_sizes fig5_panel_values
out_a3_section5 section5_dlpfc_deficit section5_connectional_vs_molecular
section6_disorder_vs_reconstruction_DK section6_reachability_spin enigma_phase1_per_disorder
enigma_disorder_unique section6_aiopto_translation section6_circuit_translation
section6_transbrain_aiopto pagani_per_model_translation abide_magel2_casecontrol
reverse_translation_validation reverse_translation_neuromaps reverse_translation_disease
reverse_translation_symptom_dissociation fig7h_homologue_transfer""".split()

# Inputs that live outside outputs/ and that an analysis would break without.
# Paths are relative to the repository root, not to its parent, so the check does not depend on
# what the checkout directory happens to be called.
EXTERNAL_INPUTS = [
    ("split-half FC (section 1)",   "outputs/splithalf/human_splithalf.npz"),
    ("split-half FC (section 1)",   "outputs/splithalf/mouse_splithalf.npz"),
    ("split-half producer",         "pipeline/02b_build_splithalf_fc.py"),
    ("ABIDE scores, de-identified", "outputs/logs/abide_otter_scores_deidentified.csv"),
]

# The split-half caches are gitignored because they are large and rebuildable, so a fresh clone
# reports them missing until pipeline/02b_build_splithalf_fc.py has been run once.
REBUILDABLE = {"outputs/splithalf/human_splithalf.npz",
               "outputs/splithalf/mouse_splithalf.npz"}

ok = True


def _find_shas(node) -> list[str]:
    """Every pi_sha256 value anywhere in a log, however deeply the stamp is nested."""
    out: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "pi_sha256" and isinstance(value, str):
                out.append(value)
            else:
                out.extend(_find_shas(value))
    elif isinstance(node, list):
        for value in node:
            out.extend(_find_shas(value))
    return out


def report(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))


print("\n1. harness imports")
try:
    import otter.repro as R
    report("import otter.repro", True, f"canonical recipe {R.CANONICAL}")
except Exception as exc:                                          # noqa: BLE001
    report("import otter.repro", False, f"{type(exc).__name__}: {exc}")
    print("\nNothing downstream can run until this imports.")
    raise SystemExit(1)

for name in R.__all__:
    if not hasattr(R, name):
        report(f"otter.repro.{name} exists", False)

print("\n2. released coupling and its provenance")
try:
    pi, prov = R.load_canonical()
    report("load_canonical()", True, f"shape {pi.shape}, sha {prov['pi_sha256'][:16]}...")
    observed_sha = prov["pi_sha256"]
except Exception as exc:                                          # noqa: BLE001
    report("load_canonical()", False, f"{type(exc).__name__}: {exc}")
    observed_sha = None

print("\n3. every cited log is present, and says which coupling produced it")
logs = ROOT / "outputs" / "logs"
missing, unstamped, shas = [], [], Counter()
for stem in CITED:
    path = logs / f"{stem}.json"
    if not path.exists():
        missing.append(stem)
        continue
    try:
        data = json.loads(path.read_text())
    except Exception:                                             # noqa: BLE001
        unstamped.append(f"{stem} (unreadable)")
        continue
    found = _find_shas(data)
    if found:
        shas.update(found)
    else:
        unstamped.append(stem)

report(f"all {len(CITED)} cited logs present", not missing,
       "" if not missing else f"missing {len(missing)}: {', '.join(sorted(missing)[:6])}"
       + (" ..." if len(missing) > 6 else ""))
report("every present log carries a pi_sha256", not unstamped,
       "" if not unstamped else f"{len(unstamped)} unstamped: {', '.join(sorted(unstamped)[:6])}"
       + (" ..." if len(unstamped) > 6 else ""))

# A log may reference several couplings. section5_dlpfc_deficit compares four of them, so the
# test is not whether every stamp equals the released sha but whether every stamp resolves to a
# coupling that exists. A sha that resolves to nothing names an input absent from the release.
if shas:
    index = R.coupling_sha_index()
    unresolved = {sha: n for sha, n in shas.items() if sha not in index}
    report("every stamped sha resolves to a coupling in outputs/coupling", not unresolved,
           f"{len(shas)} distinct sha(s); "
           + (f"unresolved: {', '.join(s[:16] + '...' for s in unresolved)}"
              if unresolved else "all resolve"))
    for sha, count in sorted(shas.items(), key=lambda kv: -kv[1]):
        name = index.get(sha, "UNKNOWN, not a file in outputs/coupling")
        flag = "" if sha == observed_sha else "   (non-canonical, expected for ablation arms)"
        print(f"         {count:2d} stamp(s) -> {name}{flag}")

print("\n4. external inputs an analysis would break without")
for label, rel in EXTERNAL_INPUTS:
    path = ROOT / rel
    if not path.exists() and rel in REBUILDABLE:
        print(f"  [ -- ] {label}: {rel}  not built yet, "
              f"run pipeline/02b_build_splithalf_fc.py")
        continue
    report(f"{label}: {rel}", path.exists())

print("\n5. packages the ported analyses need")
for mod in ("numpy", "scipy", "sklearn", "pandas", "nibabel", "nilearn",
            "anndata", "ot", "statsmodels", "nbconvert"):
    try:
        __import__(mod)
        report(mod, True)
    except ImportError:
        report(mod, False, "add to env.yml")

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED, see FAIL lines above"))
raise SystemExit(0 if ok else 1)
