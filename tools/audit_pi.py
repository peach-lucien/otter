#!/usr/bin/env python3
"""Audit which coupling (pi) the repo actually uses. Run BEFORE rebuilding figures.

WHY THIS EXISTS
---------------
On 2026-07-17 the canonical coupling was switched to pi_canonical.npy by changing the
default of otter.data.load_pi(). That only reaches callers that USE load_pi(). Scripts
doing their own np.load("<hardcoded path>") silently kept the retired coupling. Several
were re-executed, reproduced their old numbers, and the unchanged output was read as
"nothing flipped" -- a false all-clear. A day later the same quantities, computed live
through load_pi(), had REVERSED a central section-3 conclusion.

The lesson: a re-run proves nothing about which input was used. "Result did not move
after a global input change" is a RED FLAG, not a reassurance.

WHAT IT CHECKS
--------------
A. SOURCE SCAN   every .py that hardcodes a pi filename, classified:
                 LIVE (feeds manuscript) / RETIRED / BY-DESIGN (may legitimately name it)
B. LOG SCAN      every outputs/logs/*.json that records a `pi_file` / `pi_sha256`,
                 checked against the canonical coupling's actual sha256. Logs with NO stamp
                 are split by whether anything LIVE reads them (a manuscript figure script,
                 a notebook or a docs page). An unstamped log a figure reads is a published
                 number on an unverifiable coupling, and fails the audit; an unstamped log
                 nothing reads is dormant and is only listed.
C. VERDICT       exit non-zero if any LIVE script or any log is on a non-canonical pi.

Usage:
    python tools/audit_pi.py              # full report
    python tools/audit_pi.py --quiet      # only problems
    python tools/audit_pi.py --json OUT   # machine-readable
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # .../otter
REPO = ROOT.parent                                   # repo root (has manuscript/)
COUP = ROOT / "outputs" / "coupling"
LOGS = ROOT / "outputs" / "logs"

CANONICAL = "pi_canonical.npy"
SHARP = "pi_canonical_sharp.npy"
RETIRED = "pi_fc_plus_SC_with_all_packs.npy"
RETIRED_ALT = "pi_fc_plus_SC.npy"

PI_RE = re.compile(r"pi_[A-Za-z0-9_\-\+]*\.npy")

# Paths that may legitimately mention the retired coupling.
#
# THE BAR FOR ADDING SOMETHING HERE. A script is exempt only if BOTH hold, and both
# have been checked by reading the script (not by re-running it):
#   1. its HEADLINE result comes from the canonical coupling, via load_pi() with no
#      hardcoded name; and
#   2. every reference to the retired coupling is an EXPLICITLY LABELLED comparison
#      arm, a CLI/docstring example, or a build step that writes the file.
# If a script merely "also mentions" canonical somewhere, that is not enough — the
# 2026-07-17 false all-clear happened exactly because a script that named both kept
# consuming the retired one. Record the reason next to every entry.
BY_DESIGN = {
    # --- loaders / registries / fetchers: expose the retired pi as a named option ---
    "src/otter/data/io.py",                        # load_pi() docstring lists it
    "src/otter/data/fetch.py",                     # bundle sentinel; must still download it
    "src/otter/data/anchor_packs/__init__.py",     # doc reference to the pack composition
    "src/otter/data/anchor_packs/registry.py",     # doc reference to the pack composition
    "src/otter/viz/viewer.py",                     # docstring example of a pi_source label

    # --- deliberate labelled COMPARISON ARMS (verified 2026-07-18) ---
    # headline via load_pi(); retired coupling appears only as a named contrast row.
    "experiments/balsters_2020_mfc_divergence/01_mfc_divergence.py",
    #   4-coupling contrast table; "All headline numbers come from" pi_canonical.npy,
    #   retired row is labelled "pre-warp (retired)".
    "experiments/fulcher_2019_multimodal_gradient/01_gradient_validation.py",
    #   RETIRED_PI_FILE is used only to define the smaller territory the retired
    #   coupling reached, so the canonical result can be scored on equal ground;
    #   logged under coverage_control with retired_pi_file/_sha256.
    "experiments/validation/00_validate_published_maps.py",
    #   same coverage control; logged under fulcher_translation_coverage_control.
    "experiments/section5_coverage_rigor/01_coverage_nulls.py",
    #   retired appears in the docstring and in the log `note` only, recording that
    #   the 6.74-log-unit gap belonged to the retired coupling. Headline is canonical.
    "experiments/section5_coverage_rigor/11_dlpfc_deficit.py",
    #   control_b_across_couplings; _reported_coupling_arm == "canonical", each arm
    #   stamped with its own pi_provenance().
    "experiments/section5_coverage_rigor/12_coverage_continuum.py",
    #   medial_lateral_robustness across couplings; _reported_coupling_arm ==
    #   "canonical" (on which the gradient is null), each arm stamped.
    "experiments/section5_coverage_rigor/13_expansion_reconciliation.py",
    #   coverage_absX_across_couplings; _reported_coupling_arm == "canonical",
    #   each arm stamped.

    # --- build steps / CLI examples: these WRITE or NAME the file, never silently read it ---
    "pipeline/04_solve_production.py",             # writes pi_fc_plus_SC.npy
    "pipeline/run_recommended_model.py",           # orchestrates the pre-warp build chain
    "pipeline/08a_multisource_trust.py",           # --pi-file usage example in docstring
}

# Analyses known to be superseded/exploratory: not part of the manuscript stack.
#
# NOTE: experiments/autism_subtypes/01_network_crossvalidation.py is NOT superseded --
# it is the baseline that experiments/whitesell_2021_dmn compares against, and it was
# repointed to canonical pi on 2026-07-18. It no longer names a retired coupling, so it
# does not surface here; if it ever does again, that is a real problem, not exploratory.
RETIRED_DIRS = (
    "experiments/autism_subtypes/",
    "experiments/pagani_2026_per_model/",
    "experiments/spatial_null_check/",
    "experiments/anchor_packs/",
    "experiments/archive/",
)

# Individual superseded analyses. Prefer this over blanket-exempting a directory, so a
# NEW script in the same directory still gets audited.
RETIRED_FILES = {
    # Convergent-negative ablation: asks whether per-region xyz zeroing fixes the
    # topology-inverted regions, fits its own couplings and compares them against the
    # pre-warp production baseline. Superseded by the anchor-warped canonical coupling,
    # which is the approach that actually worked. Not a manuscript number.
    "experiments/ablations/per_region_xyz.py",
}

# Logs from superseded analyses that are on a retired coupling and will NOT be re-run.
# These are quarantined, not forgiven: they stay in the report under their own heading so
# they cannot be mistaken for verified, and the verdict does not go red on dead work.
# Only add a log here once you have confirmed NOTHING live reads it (grep manuscript/,
# notebooks/ and docs/), and say why it is not simply being re-run.
SUPERSEDED_LOGS = {
    # The 2026-06 autism-subtype translation chain (experiments/autism_subtypes/
    # 04_subtype_translation, 05_subtype_contrast, 07_full_matrix_translation). Superseded
    # twice over: by the canonical coupling AND by experiments/pagani_2026_per_model/,
    # which corrected the subtype labels these runs had inverted. Re-running them on the
    # canonical coupling would produce numbers that are still wrong for the label reason,
    # so they are quarantined instead. Only otter/notebooks/archive/ reads them.
    "autism_subtypes_contrast.json",
    "autism_subtypes_full_matrix.json",
    "autism_subtypes_translation.json",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def classify(rel: str) -> str:
    if rel in BY_DESIGN:
        return "BY-DESIGN"
    if rel in RETIRED_FILES:
        return "RETIRED"
    if any(rel.startswith(d) for d in RETIRED_DIRS):
        return "RETIRED"
    return "LIVE"


def scan_sources() -> list[dict]:
    out = []
    for base in (ROOT / "experiments", ROOT / "pipeline", ROOT / "src", REPO / "manuscript" / "figures"):
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*.py")):
            if "__pycache__" in f.parts:
                continue
            try:
                text = f.read_text(errors="ignore")
            except OSError:
                continue
            names = set(PI_RE.findall(text))
            if not names:
                continue
            stale = {n for n in names if n in (RETIRED, RETIRED_ALT)}
            if not stale:
                continue
            # A script that WRITES the retired coupling is a historical build step, not a
            # stale consumer. Only consumers can silently contaminate a result.
            role = "ambiguous"
            consumes = produces = False
            for line in text.splitlines():
                if not any(n in line for n in stale):
                    continue
                low = line.lower()
                if "np.save" in low or "savez" in low or "out_path" in low or "= coup /" in low:
                    produces = True
                if "np.load" in low or "load_pi" in low or "read" in low or "pi_file" in low:
                    consumes = True
            if consumes and not produces:
                role = "consumer"
            elif produces and not consumes:
                role = "producer"
            elif consumes and produces:
                role = "both"
            try:
                rel = str(f.relative_to(ROOT))
            except ValueError:
                rel = str(f.relative_to(REPO))
            uses_loader = "load_pi(" in text
            cls = classify(rel)
            if cls == "LIVE" and role == "producer":
                cls = "BY-DESIGN"          # historical build step that emits the old pi
            out.append({
                "file": rel,
                "class": cls,
                "role": role,
                "stale_refs": sorted(stale),
                "also_canonical": CANONICAL in names,
                "uses_load_pi": uses_loader,
            })
    return out


def scan_log_consumers() -> dict[str, list[str]]:
    """Which outputs/logs/*.json files does something LIVE actually read?

    Live means a manuscript figure script, a notebook, or a docs page -- the places a number can
    travel from a log into the paper. `manuscript/figures/_attic/` is excluded: those scripts are
    retired, so a log read only by them is dormant, not live.

    A filename that appears only in a docstring or comment is a MENTION, not a read, and does not
    make the log live. Returns {log_basename: [consumers]}; a log absent from this map is dormant
    and may be unstamped without endangering a published number.
    """
    consumers: dict[str, list[str]] = {}
    mentions: dict[str, list[str]] = {}
    names = {f.name for f in LOGS.glob("*.json")}
    haystacks = []
    figs = REPO / "manuscript" / "figures"
    if figs.is_dir():
        haystacks += [p for p in figs.rglob("*.py") if "_attic" not in p.parts
                      and "__pycache__" not in p.parts]
    for sub in ("notebooks", "docs"):
        d = ROOT / sub
        if d.is_dir():
            haystacks += [p for p in d.rglob("*") if p.suffix in (".ipynb", ".md")
                          and "archive" not in p.parts]
    READ_TOKENS = ("open(", "read_text", "json.load", "loads(", "np.load", "LOGS /",
                   "load(", "pd.read", "Path(")
    for p in haystacks:
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        try:
            rel = str(p.relative_to(REPO))
        except ValueError:
            rel = str(p)
        for n in names:
            if n not in text:
                continue
            # Only count it if SOME line both names the log and does something with it.
            # A docstring or comment that merely mentions the filename is a mention, not a read;
            # make_fig3a_network_matrix.py names a sibling log purely to warn against confusing
            # the two, and that must not read as a provenance failure.
            hit = False
            for line in text.splitlines():
                if n not in line:
                    continue
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                if any(tok in line for tok in READ_TOKENS):
                    hit = True
                    break
            if hit:
                consumers.setdefault(n, []).append(rel)
            else:
                mentions.setdefault(n, []).append(rel)
    return consumers

def scan_logs(canon_hash: str) -> list[dict]:
    """Any log recording pi provenance is checked. Logs with no provenance are reported
    separately -- absence of provenance is itself the problem worth fixing."""
    checked, unstamped = [], []
    for f in sorted(LOGS.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        blob = json.dumps(d)
        pi_files = sorted(set(PI_RE.findall(blob)))
        h = None
        if isinstance(d, dict):
            h = d.get("pi_sha256") or d.get("_pi_sha256")
        if not pi_files and not h:
            unstamped.append(f.name)
            continue
        ok = True
        why = []
        arms = []
        bad = [p for p in pi_files if p in (RETIRED, RETIRED_ALT)]
        if h and h == canon_hash:
            # The TOP-LEVEL stamp is the headline. A log may still name a retired
            # coupling deeper in, as a labelled comparison arm (see BY_DESIGN). That
            # is reported, not failed -- but only because the headline sha256 has been
            # verified against the canonical file, not because the script said so.
            if bad:
                arms = bad
                why.append("headline canonical; retired coupling appears as a nested "
                           "comparison arm: " + ", ".join(bad))
        else:
            if h and h != canon_hash:
                ok = False
                why.append(f"sha256 {h[:12]}... != canonical {canon_hash[:12]}...")
            if bad:
                # No verified canonical headline stamp, so a retired name is unexplained.
                ok = False
                why.append("names retired coupling with no canonical headline stamp: "
                           + ", ".join(bad))
        superseded = f.name in SUPERSEDED_LOGS
        if superseded and not ok:
            ok = True                      # quarantined, still printed under its own heading
        checked.append({"log": f.name, "ok": ok, "pi_files": pi_files, "sha256": h,
                        "comparison_arms": arms, "superseded": superseded,
                        "why": "; ".join(why)})
    return checked, unstamped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--json", dest="json_out")
    a = ap.parse_args()

    canon = COUP / CANONICAL
    if not canon.exists():
        print(f"FATAL: canonical coupling missing: {canon}")
        return 2
    canon_hash = sha256(canon)

    srcs = scan_sources()
    logs, unstamped = scan_logs(canon_hash)
    consumers = scan_log_consumers()
    unstamped_live = [n for n in unstamped if n in consumers]
    unstamped_dormant = [n for n in unstamped if n not in consumers]

    live = [s for s in srcs if s["class"] == "LIVE"]
    retired = [s for s in srcs if s["class"] == "RETIRED"]
    bydesign = [s for s in srcs if s["class"] == "BY-DESIGN"]
    bad_logs = [l for l in logs if not l["ok"]]

    print("=" * 78)
    print("PI PROVENANCE AUDIT")
    print("=" * 78)
    print(f"canonical : {CANONICAL}")
    print(f"sha256    : {canon_hash}")
    print(f"retired   : {RETIRED}")
    print()
    print(f"LIVE scripts still on the retired coupling : {len(live)}   <- must fix")
    print(f"RETIRED/exploratory scripts                : {len(retired)}   (ok to leave)")
    print(f"BY-DESIGN references                       : {len(bydesign)}   (ok)")
    print(f"logs recording provenance                  : {len(logs)}  ({len(bad_logs)} stale)")
    print(f"logs with NO pi provenance recorded        : {len(unstamped)}"
          f"  ({len(unstamped_live)} read by a live figure/notebook/doc)   <- cannot verify")
    print()

    if live:
        print("-" * 78)
        print("LIVE SCRIPTS ON RETIRED PI (repoint + re-run + re-check conclusions):")
        for s in sorted(live, key=lambda x: (x["role"] != "consumer", x["file"])):
            flag = "  [also refs canonical]" if s["also_canonical"] else ""
            print(f"  [{s['role']:9s}] {s['file']}{flag}")
        print()
    if bad_logs:
        print("-" * 78)
        print("LOGS ON A NON-CANONICAL PI:")
        for l in bad_logs:
            print(f"  {l['log']}: {l['why']}")
        print()
    sup_logs = [l for l in logs if l.get("superseded")]
    if sup_logs:
        print("-" * 78)
        print("LOGS QUARANTINED AS SUPERSEDED (on a retired pi; nothing live reads them):")
        for l in sup_logs:
            print(f"  {l['log']}: {l['why'] or 'retired coupling'}")
        print()
    arm_logs = [l for l in logs if l["ok"] and not l.get("superseded")
                and l.get("comparison_arms")]
    if arm_logs and not a.quiet:
        print("-" * 78)
        print("LOGS WITH A CANONICAL HEADLINE + LABELLED RETIRED COMPARISON ARMS (ok):")
        for l in arm_logs:
            print(f"  {l['log']}: {', '.join(l['comparison_arms'])}")
        print()
    if unstamped_live:
        print("-" * 78)
        print("LIVE LOGS WITH NO PI PROVENANCE (a published number on an unverifiable coupling):")
        for n in sorted(unstamped_live):
            print(f"  {n}")
            for c in sorted(consumers[n])[:4]:
                print(f"      read by {c}")
        print("  FIX: re-run each producing script so the output carries pi_file + pi_sha256.")
        print()
    if unstamped_dormant and not a.quiet:
        print("-" * 78)
        print(f"DORMANT LOGS WITH NO PI PROVENANCE ({len(unstamped_dormant)}, nothing live reads them):")
        for n in sorted(unstamped_dormant):
            print(f"  {n}")
        print()

    ok = not live and not bad_logs and not unstamped_live
    print("=" * 78)
    print("VERDICT:", "PASS - every live analysis is on the canonical coupling, and every log a "
          "figure reads is stamped"
          if ok else "FAIL - see above; do NOT rebuild figures until this is clean")
    print("=" * 78)

    if a.json_out:
        Path(a.json_out).write_text(json.dumps({
            "canonical": CANONICAL, "sha256": canon_hash,
            "live_on_retired": live, "retired_scripts": retired,
            "by_design": bydesign, "logs": logs, "unstamped_logs": unstamped,
            "unstamped_live": unstamped_live, "unstamped_dormant": unstamped_dormant,
            "log_consumers": consumers,
            "pass": ok,
        }, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
