#!/usr/bin/env python3
"""Provenance checker: every number in the manuscript must trace to an outputs/logs JSON.

REVIVED 2026-07-29. The original lived at tools/check_manuscript_numbers.py and read
`manuscript/results_section.md`. When the manuscript moved to .docx that input disappeared and the
script was retired to manuscript/figures/_attic/superseded_text/. It has not run since. This version
reads the .docx and keeps everything that made the original worth having.

WHY THIS EXISTS (from the original, still true)
----------------------------------------------
On 2026-07-12 three classes of number were found in the manuscript that no output file supported:
  * spin p = 0.021 / 0.010 for the Fulcher structural translation. Real values 0.114 / 0.103.
  * "AUROC 0.85" for TransBrain. True value 0.8446 -> 0.84.
  * "TransBrain leads 15 of 24" — that is the mass-in-region split; by AUROC it is 16 / 8.
Each was hand-typed and drifted from its source.

THE DESIGN POINT THAT MATTERS
-----------------------------
Numbers are checked ONLY against the JSONs their own section declares, never against a global index.
With ~12,000 values across 218 logs, a global index matches almost anything and proves nothing — it
cannot catch a number sourced from the wrong file, which is the actual failure mode above. If you
loosen this, you have a script that always passes.

Sections are identified by their heading text; the logs each section may draw on are declared in
SECTION_SOURCES below. When a section gains an analysis, add its log here — a number whose section
declares no source is reported separately rather than silently passed.

Usage
-----
    cd otter && python3 tools/check_manuscript_numbers.py
    cd otter && python3 tools/check_manuscript_numbers.py --section 2
    cd otter && python3 tools/check_manuscript_numbers.py --verbose
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                 # .../otter
LOGS = ROOT / "outputs" / "logs"
# The manuscript keeps its original filename until the renamed copy replaces it. Point this
# at OTTER_first_draft.v2.docx once that swap is made.
MANUSCRIPT = ROOT.parent / "manuscript" / "HOMER_first_draft.v2.docx"

# Heading text -> section key. Matched case-insensitively against whole paragraphs.
SECTION_HEADINGS = {
    "abstract": "0",
    "otter learns a calibrated probabilistic mouse-human coupling": "1",
    "connectivity, spatial structure and curation cover each other's failures": "2",
    "π transfers areal organisation along the cortical hierarchy, but not laminar structure": "3",
    "otter complements a state-of-the-art phenotype translator": "4",
    "otter measures the connectional reorganisation of human association cortex": "5",
    "otter translates mouse experiments into human predictions and clinical targets into mouse circuits": "6",
    "discussion": "END",
}

# Which logs each section is allowed to draw on. Add to this when a section gains an analysis.
SECTION_SOURCES: dict[str, list[str]] = {
    "0": ["beauchamp_metric_battery_canonical.json", "beauchamp_metric_battery_loro_canonical.json",
          "transbrain_benchmark_summary.json", "coupling_summary_canonical.json"],
    "1": ["beauchamp_metric_battery_canonical.json", "coupling_summary_canonical.json",
          "evidence_tiers_canonical.json", "fig1_coupling_matrix.json",
          "out_a2_splithalf.json", "out_a1c_downstream.json", "out_a1d_robust.json",
          "region_level_eval_canonical.json"],
    "2": ["ablation_ladder_battery_canonical.json", "out_a1_ladder.json",
          "heldout_three_config_canonical.json", "out_a1b_loro.json", "out_g2_regret.json",
          "anchor_recovery_loo_combined_canonical.json",
          "beauchamp_metric_battery_canonical.json", "beauchamp_metric_battery_loro_canonical.json"],
    "3": ["coletta_2020_cross_species_rsn.json", "margulies_2016_gradient.json",
          "published_map_validation.json", "fulcher_2019_gradient.json",
          "biccn_contrast_reframe.json", "biccn_composition_from_markers.json",
          "hodge_areal_type_reframe.json", "hodge_2019_layer_markers_refined.json",
          "hodge_markers_like_for_like.json", "out_c1_gradient.json", "out_c2_nulls.json"],
    "4": ["transbrain_benchmark_summary.json", "transbrain_2025_benchmark.json",
          "transbrain_bn_distributions.json", "transbrain_roundtrip_maps.json",
          "transbrain_bn_sizes.json"],
    "5": ["fig5_panel_values.json", "out_a3_section5.json", "out_a1c_downstream.json",
          "section5_dlpfc_deficit.json", "section5_connectional_vs_molecular.json",
          "section6_disorder_vs_reconstruction_DK.json", "section6_reachability_spin.json",
          "enigma_phase1_per_disorder.json", "enigma_disorder_unique.json"],
    "6": ["section6_aiopto_translation.json", "section6_circuit_translation.json",
          "section6_transbrain_aiopto.json", "pagani_per_model_translation.json",
          "abide_magel2_casecontrol.json", "reverse_translation_validation.json",
          "reverse_translation_neuromaps.json", "reverse_translation_disease.json",
          "reverse_translation_symptom_dissociation.json", "fig7h_homologue_transfer.json"],
}

ALLOWLIST: dict[str, str] = {
    "0.05": "FDR / significance threshold, not a reported value",
    "0.001": "threshold in 'p < 0.001', not a reported value",
    "0.5": "probability or chance threshold, not a reported value",
    "51": "gene-homologue panel size (constant of the gene data)",
    "20": "'more than 20' primary studies (prose)",
    "40": "bootstrap resamples (a design choice, not a result)",
    "2094": "parcel count, structural constant of the human atlas",
    "1864": "parcel count, structural constant of the mouse atlas",
    "1824": "cortical parcel count, structural constant",
    "1768": "Schaefer-400 cortical parcel count",
    "1635": "common-parcel count for the OTTER/TransBrain comparison",
    "127": "Brainnetome region count", "120": "Brainnetome region count (approx, prose)",
    "400": "Schaefer-400 parcellation", "388": "Schaefer regions the coupling reaches",
    "42": "number of curated Garin anchors", "41": "combined supervision units",
    "26": "curated region packs", "21": "Garin homology classes",
    "24": "benchmark region count", "19": "Beauchamp homology pairs",
    "16": "Yeo-17 networks tested", "17": "Yeo-17 networks",
    "34": "Desikan-Killiany regions per hemisphere", "68": "DK structures (34 x 2)",
    "30": "DK cortical regions OTTER resolves", "15": "scorable Garin classes / ENIGMA count",
    "14": "published mouse measurements in the battery", "12": "functional systems, reverse test",
    "8": "dopamine PET maps / prose count", "7": "prose count", "9": "prose count",
    "10": "prose count", "11": "prose count", "13": "prose count",
    "1": "prose", "2": "prose", "3": "prose", "4": "prose", "5": "prose", "6": "prose", "0": "prose",
    "1000": "null rotations, a design choice", "2000": "bootstrap resamples, a design choice",
    "500": "null rotations", "200": "null rotations", "871": "ABIDE participants (cohort size)",
    "403": "ABIDE autism n", "468": "ABIDE control n", "113": "human FC cohort size",
    "105": "mouse FC cohort size", "233": "ABIDE cases (prose)",
    "2010": "citation year", "2013": "citation year", "2016": "citation year",
    "2018": "citation year", "2019": "citation year", "2020": "citation year",
    "2021": "citation year", "2022": "citation year", "2025": "citation year",
    "95": "confidence-interval convention, not a reported value",
}

def _key_affinity(hit: str, ctx: str) -> int:
    """How many words of the surrounding sentence appear in the matched JSON key path.

    A crude relevance score, but enough to surface the right source first when several logs happen
    to hold the same value. It ranks candidates; it never rejects one.
    """
    key = hit.split(":", 1)[-1].lower()
    words = {w for w in re.findall(r"[a-z]{4,}", ctx.lower())}
    return sum(1 for w in words if w in key)


NUM_RE = re.compile(r"(?<![\w.])(-|−)?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?![\w])")

# A number immediately followed by a citation marker is quoted from the literature, not measured
# here, so it has no business in our logs. Handled in code rather than by allowlist so the
# exemption is visible in context and cannot silently cover a measured value.
CITED_RE = re.compile(r"^\s*(?:SD|%|mm)?\s*\[ref")


def flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}{k}.")
    elif isinstance(obj, list):
        if len(obj) <= 40:                       # long arrays are raw data, not reportable numbers
            for i, v in enumerate(obj):
                yield from flatten(v, f"{prefix}[{i}].")
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)) and math.isfinite(obj):
        yield prefix.rstrip("."), float(obj)


def load_per_file() -> dict[str, list[tuple[str, float]]]:
    """{filename -> [(key, value)]}. Deliberately NOT a global index — see the module docstring."""
    out = {}
    for p in sorted(LOGS.glob("*.json")):
        try:
            out[p.name] = list(flatten(json.loads(p.read_text())))
        except Exception:
            continue
    scratch = ROOT.parent / "_audit"
    for p in sorted(scratch.glob("**/out_*.json")) if scratch.exists() else []:
        try:
            out[p.name] = list(flatten(json.loads(p.read_text())))
        except Exception:
            continue
    return out


def decimals(raw: str) -> int:
    return len(raw.split(".")[1]) if "." in raw else 0


def matches(val: float, raw: str, pairs):
    d = decimals(raw)
    hits = []
    for k, v in pairs:
        for scale in (1.0, 100.0, 0.01):         # allow proportion <-> percent
            if round(v * scale, d) == round(val, d):
                hits.append(k)
                break
    return hits


def manuscript_paragraphs():
    from docx import Document
    doc = Document(MANUSCRIPT)
    sec = None
    for i, p in enumerate(doc.paragraphs):
        t = p.text.strip()
        if not t:
            continue
        key = SECTION_HEADINGS.get(t.lower())
        if key is not None:
            sec = key
            continue
        if sec is None or sec == "END":
            continue
        yield sec, i, t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", help="only check one section")
    ap.add_argument("--verbose", action="store_true", help="show matching JSON keys")
    args = ap.parse_args()

    if not MANUSCRIPT.exists():
        print(f"manuscript not found: {MANUSCRIPT}")
        return 2
    per_file = load_per_file()
    print(f"loaded {len(per_file)} JSON files\nmanuscript: {MANUSCRIPT.name}\n")

    verified, unverified, nosource, ambiguous = 0, [], [], []
    for sec, para, text in manuscript_paragraphs():
        if args.section and sec != args.section:
            continue
        for m in NUM_RE.finditer(text):
            raw = m.group(0)
            try:
                val = float(raw.replace(",", "").replace("−", "-"))
            except ValueError:
                continue
            if f"{val:g}".lstrip("-") in ALLOWLIST:
                continue
            if CITED_RE.match(text[m.end():m.end() + 12]):
                continue                              # value quoted from a cited paper
            ctx = text[max(0, m.start() - 55):m.end() + 25]
            srcs = SECTION_SOURCES.get(sec, [])
            if not srcs:
                nosource.append((sec, para, raw, ctx)); continue
            hits = [f"{fn}:{k}" for fn in srcs if fn in per_file
                    for k in matches(val, raw, per_file[fn])]
            # Rank hits by how well the key name matches the words around the number. A value can
            # coincide with an unrelated quantity in a sibling log -- 0.84 for a Spearman rho once
            # resolved to a principal-component loading -- so the plausible source is shown first
            # and the count of alternatives is shown with it.
            hits.sort(key=lambda h: -_key_affinity(h, ctx))
            if hits:
                verified += 1
                if len(hits) > 1:
                    ambiguous.append((sec, para, raw, ctx, hits))
                if args.verbose:
                    extra = f"   (+{len(hits)-1} other match{'es' if len(hits) > 2 else ''})" if len(hits) > 1 else ""
                    print(f"  ok   §{sec} p{para}  {raw:>9}  <- {hits[0]}{extra}")
            else:
                unverified.append((sec, para, raw, ctx))

    print(f"VERIFIED   {verified:>3} numbers matched a value in a JSON their own section declares")
    print(f"UNVERIFIED {len(unverified):>3} numbers have NO match in the JSONs their section declares")
    if nosource:
        print(f"NO SOURCE  {len(nosource):>3} numbers sit in a section that declares no JSON at all")
    print()
    for label, rows in (("UNVERIFIED — trace each to a source, or remove it", unverified),
                        ("NO SOURCE — add the section's logs to SECTION_SOURCES", nosource)):
        if rows:
            print(label + ":")
            for sec, para, raw, ctx in rows:
                print(f"  §{sec:<3} p{para:<4} {raw:>10}   …{ctx.strip()}…")
            print()
    if ambiguous and args.verbose:
        print(f"AMBIGUOUS  {len(ambiguous)} numbers match more than one declared source. A match is "
              "not proof\n           the number came from that source; skim these:")
        for sec, para, raw, ctx, hits in ambiguous:
            print(f"  §{sec:<3} p{para:<4} {raw:>10}   {hits[0]}   (+{len(hits)-1})")
        print()
    return 1 if unverified else 0


if __name__ == "__main__":
    sys.exit(main())
