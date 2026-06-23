"""Reproducible end-to-end pipeline for the recommended HOMER model.

The "recommended model" is the production FC+SC coupling supervised with the
default region-anchor packs (``pi_fc_plus_SC_with_all_packs.npy``). It is
the π most downstream queries and the GUI should use.

This orchestrator reproduces it from scratch, running the whole chain in order
so the steps stay in sync (solve → compose → bootstrap → trust → GUI):

    solve      pipeline/04_solve_production.py
                 -> outputs/coupling/pi_fc_plus_SC.npy
    compose    experiments/anchor_packs/compose_all.py
                 -> outputs/coupling/pi_fc_plus_SC_with_all_packs.npy
                 -> outputs/logs/beauchamp_validation_all_packs.json
                 -> outputs/logs/region_level_eval_all_packs.json
    bootstrap  pipeline/06_bootstrap.py  (subject-level stability)
                 -> outputs/coupling/bootstrap_aggregate_fc_plus_SC.npz
    trust      pipeline/08a_multisource_trust.py  (multi-source evidence map)
                 -> outputs/coupling/trust_multisource_all_packs.npz
    gui        pipeline/08_build_gui.py
                 -> outputs/gui/index.html  (+ docs/index.html with --publish)

Usage:
    # Full reproduction from scratch (re-fits everything; bootstrap ~10 min)
    PYTHONPATH=src python pipeline/run_recommended_model.py

    # Reuse existing solves/bootstrap, only refresh trust map + GUI
    PYTHONPATH=src python pipeline/run_recommended_model.py --start-from trust

    # Skip the slow bootstrap re-run, reuse its existing aggregate state
    PYTHONPATH=src python pipeline/run_recommended_model.py --bootstrap-iters 0

    # Run a single step
    PYTHONPATH=src python pipeline/run_recommended_model.py --only gui --publish
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline"

# Ordered stage names, `--start-from` / `--only` reference these.
STAGES = ["solve", "compose", "bootstrap", "trust", "gui"]


def run(cmd: list[str], *, label: str) -> None:
    """Run a subprocess from the repo root with src/ on PYTHONPATH; abort on failure.

    Prepends ``src/`` to any existing PYTHONPATH rather than replacing it, so a
    caller's environment (e.g. packages installed to a non-default location) is
    preserved.
    """
    import subprocess
    pythonpath = os.pathsep.join(
        p for p in (str(ROOT / "src"), os.environ.get("PYTHONPATH", "")) if p
    )
    env = {**os.environ, "PYTHONPATH": pythonpath}
    print(f"\n{'=' * 70}\n[recommended-model] {label}\n  $ {' '.join(cmd)}\n{'=' * 70}",
          flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    if result.returncode != 0:
        sys.exit(f"[recommended-model] {label} FAILED (exit {result.returncode})")
    print(f"[recommended-model] {label} done in {time.time() - t0:.1f}s", flush=True)


def stage_solve(args) -> None:
    run([sys.executable, str(PIPELINE / "04_solve_production.py"),
         "--config", "fc_plus_SC"],
        label="solve, production FC+SC point-anchor π")


def stage_compose(args) -> None:
    run([sys.executable, str(ROOT / "experiments" / "anchor_packs" / "compose_all.py")],
        label="compose, fit recommended π with the 5 default anchor packs")


def stage_bootstrap(args) -> None:
    boot = PIPELINE / "06_bootstrap.py"
    if args.bootstrap_iters > 0:
        # Clean slate so the run is exactly N iterations, not N-on-top-of-existing.
        state = ROOT / "outputs" / "coupling" / "bootstrap_state_fc_plus_SC.npz"
        if state.exists():
            print(f"[recommended-model] clearing stale bootstrap state {state.name}")
            state.unlink()
        run([sys.executable, str(boot), "--config", "fc_plus_SC",
             "--iters", str(args.bootstrap_iters)],
            label=f"bootstrap, {args.bootstrap_iters} subject-level iterations")
    else:
        print("[recommended-model] bootstrap-iters=0, reusing existing "
              "bootstrap state, only regenerating the aggregate")
    # --report (re)builds bootstrap_aggregate_fc_plus_SC.npz from the state.
    run([sys.executable, str(boot), "--config", "fc_plus_SC", "--report"],
        label="bootstrap, aggregate per-row stability")


def stage_trust(args) -> None:
    run([sys.executable, str(PIPELINE / "08a_multisource_trust.py")],
        label="trust, multi-source per-parcel evidence map")


def stage_gui(args) -> None:
    cmd = [sys.executable, str(PIPELINE / "08_build_gui.py")]
    if args.publish:
        cmd.append("--publish")
    run(cmd, label="gui, build region-first mapping GUI"
                    + (" (+ publish to docs/)" if args.publish else ""))


STAGE_FNS = {
    "solve":     stage_solve,
    "compose":   stage_compose,
    "bootstrap": stage_bootstrap,
    "trust":     stage_trust,
    "gui":       stage_gui,
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--start-from", choices=STAGES, default="solve",
                    help="skip every stage before this one (default: solve)")
    ap.add_argument("--only", choices=STAGES, default=None,
                    help="run only this single stage")
    ap.add_argument("--bootstrap-iters", type=int, default=40,
                    help="subject-bootstrap iterations to run; 0 reuses the "
                         "existing state and only regenerates the aggregate "
                         "(default: 40)")
    ap.add_argument("--publish", action="store_true",
                    help="forward --publish to 08_build_gui.py (copies the "
                         "rendered GUI into docs/index.html for GitHub Pages)")
    args = ap.parse_args()

    if args.only:
        stages = [args.only]
    else:
        stages = STAGES[STAGES.index(args.start_from):]

    print(f"[recommended-model] stages to run: {' -> '.join(stages)}")
    t0 = time.time()
    for stage in stages:
        STAGE_FNS[stage](args)
    print(f"\n[recommended-model] all stages complete in {time.time() - t0:.1f}s")
    if "gui" in stages:
        print(f"[recommended-model] open outputs/gui/index.html")


if __name__ == "__main__":
    main()
