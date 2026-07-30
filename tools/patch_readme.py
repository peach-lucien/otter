#!/usr/bin/env python3
"""Bring README.md into line with the submitted manuscript.

The README predates the review round. It reports the older spatial null for the microstructure
translation, which reversed that result, and carries superseded TransBrain numbers. It also frames
section 5 as coverage, which the paper retired, and does not mention section 6 at all.

Each edit asserts its target appears exactly once, so a changed README fails loudly rather than
being partially patched. Numbers are taken from the manuscript sections named in the comments.

    cd homer && python3 tools/patch_readme.py            # apply
    cd homer && python3 tools/patch_readme.py --check    # report without writing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

# (label, old, new). Old strings are verbatim from the README as committed.
EDITS: list[tuple[str, str, str]] = [

    # ---- section 3. The old null reversed this result; the manuscript reports it as positive. ----
    ("microstructure bullet",
     "- **It does not translate microstructure.** Routed mouse myelin and cytoarchitecture "
     "resemble the human myelin map (r = 0.37 / 0.36) but do not clear a spatial null "
     "(spin p = 0.11 / 0.10). π was fitted on connectivity and carries connectivity. Do not "
     "read a microstructural correspondence out of it. Fine molecular detail is weaker still: "
     "broad cell classes translate (excitatory − inhibitory, r = 0.26), laminar and areal-type "
     "contrasts do not.",

     "- **It does not translate properties orthogonal to the areal hierarchy.** What travels "
     "through π is areal position, so a mouse measurement transfers if it varies along the "
     "sensory-to-association axis and does not if it varies through the cortical depth. Myelin and "
     "cytoarchitecture do transfer, each clearing a translation null that rotates the mouse input "
     "and routes it through the real π (|r| = 0.50, p = 0.005 and |r| = 0.53, p = 0.003), and "
     "reaching r = 0.47 against the human myelin map. Cell-class composition transfers when it "
     "tracks that axis (neuronal minus glial 0.35, excitatory minus inhibitory 0.34). Laminar "
     "contrasts do not (supragranular minus infragranular 0.01, supragranular minus granular "
     "0.02), and neither do spatially uniform cell classes (GABAergic 0.00, oligodendrocyte 0.07, "
     "microglial −0.03). Earlier versions of this README reported myelin as failing its null. "
     "That used a null which shuffled the coupling rather than rotating the input, and it was "
     "replaced."),

    # ---- section 4. Every figure below is from the manuscript's benchmark table. ----
    ("TransBrain bullet",
     "On region identity TransBrain leads on its own benchmark (AUROC 0.84 vs 0.79), and the "
     "difference is not significant (paired Wilcoxon p = 0.17). HOMER leads where the modality is "
     "connectional: the gradient (0.55 vs 0.42), round-trip fidelity (0.98/0.95/0.97 vs "
     "0.89/0.82/0.83), sharpness (≈ 3 vs ≈ 60 effective target regions) and absence "
     "detection.",

     "On region identity the two are level on TransBrain's own benchmark, AUROC 0.83 against 0.84, "
     "a paired per-region difference that is not significant (Wilcoxon p = 0.36). HOMER leads "
     "where the modality is connectional. It tracks the human gradient at r = 0.56 against 0.52, "
     "recovers a phenotype routed mouse to human and back at 0.97, 0.86 and 0.91 against 0.89, "
     "0.82 and 0.83, concentrates its predictions on an effective 6 target regions against 60, and "
     "places three times as much mass on the correct region, 0.21 against 0.07."),

    # ---- section 5. "Coverage" was retired. The measure is reconstruction accuracy. ----
    ("section 5 heading",
     "### Where π has no support\n\nSemi-relaxed FGW frees the human marginal, so the coupling "
     "may leave human parcels poorly\nreconstructed. Reconstruction-coverage asks how well each "
     "human parcel's connectivity fingerprint\nis rebuilt by routing mouse connectivity through "
     "π. It runs high over sensorimotor, auditory and\nvisual territory and low over prefrontal "
     "and lateral temporal cortex.",

     "### Where the mouse cannot reconstruct human connectivity\n\nReconstruction accuracy asks "
     "how well each human parcel's connectivity fingerprint is rebuilt by\nrouting mouse "
     "connectivity through π. Each column of π is normalised before the push-forward, so "
     "the\nscore reflects whether some mouse tissue is wired like the human parcel and not how "
     "much mass\nthat parcel received. It runs high over sensorimotor, auditory and visual "
     "territory and low over\nprefrontal and lateral temporal cortex. Across 1,824 cortical "
     "parcels the mean is r = 0.45."),

    ("coverage wording in the expansion paragraph",
     "That deficit tracks cortical expansion. Six of seven published maps clear a spin null: "
     "coverage\nfalls with macaque→human expansion",
     "That deficit tracks cortical expansion. Six of seven published maps clear a spin null. "
     "Reconstruction\naccuracy falls with macaque-to-human expansion"),

    ("mouse has the parts",
     "flat (−0.18 SD, p = 0.39). The mouse appears to have the parts without the wiring.",
     "flat (−0.18 SD, p = 0.39). Human dorsolateral prefrontal cortex remains molecularly "
     "mammalian while\nhaving lost its connectional counterpart, so the species difference is a "
     "reorganisation of\nconnections rather than a replacement of tissue."),

    # The disorder result moved to Methods in the paper. Keep it here but name it as a control.
    ("disorder paragraph",
     "Coverage does not, however, resolve disorders. Correlating it with case-control "
     "cortical-thickness",
     "Reconstruction accuracy does not resolve disorders. Correlating it with case-control "
     "cortical-thickness"),

    ("reachability wording",
     "disorder's thinning burden by reachability is null for all six disorders (minimum p = 0.13). "
     "The\ntest can detect a hierarchy-aligned effect when one is present: run identically, the "
     "myelin map\nflags bipolar disorder (p = 0.028) and major depression (p = 0.011) in the same "
     "data.",
     "disorder's thinning burden by reachability is null for all six disorders (minimum p = 0.13). "
     "The\ntest detects a hierarchy-aligned effect when one is present. Run identically, the "
     "myelin map flags\nbipolar disorder (p = 0.028) and major depression (p = 0.011) in the same "
     "data."),

    # ---- section 6 is absent. Add it after the confidence-grading section. ----
    ("add section 6",
     "Full results in [`docs/03_results.md`](docs/03_results.md). One notebook per figure in\n"
     "[`notebooks/`](notebooks/).",

     "### Translation in both directions\n\nπ is an operator, so it runs forward and back. "
     "Multiplying a mouse map by π gives a\ntransport-weighted average over the human brain, "
     "and transposing π turns a human map into a\nranking over mouse structures.\n\nForward, an "
     "optogenetic mouse anterior-insula activation map routes onto human anterior insula\nand "
     "ventral-attention cortex. Salience cortex is enriched by +0.86 SD against a permuted-π "
     "null\n(p = 0.001). Scored head to head on the 1,635 parcels both methods cover, HOMER "
     "reaches +0.87 SD and\na transcriptomic translator +0.28 SD. HOMER exceeds a shuffled-input "
     "null (p = 0.016) and the\ntranscriptomic translator does not (p = 0.228).\n\n"
     "Reverse, twelve human functional systems route to their "
     "established mouse substrate, with the\nground-truth structure in the top three for nine of "
     "twelve and all twelve clearing a\n1,000-rotation spatial null. Eight human dopamine PET maps "
     "each route to the striatum\n(p ≤ 0.005), and the routing is specific, since cannabinoid "
     "CB1 and GABA-A maps route to sensory\ncortex instead. Two antidepressant TMS circuits that "
     "overlap in the human cortex separate in the\nmouse, the dysphoric one onto medial prefrontal "
     "cortex and the anxiosomatic one onto amygdala and\ninsula, with the contrast clearing a spin "
     "null (C = +0.59, p = 0.0005).\n\nFull results in [`docs/03_results.md`]"
     "(docs/03_results.md). The notebooks in [`notebooks/`](notebooks/)\nderive every number "
     "above."),

    # ---- section 3. The layer-marker comparison was retired. Scored like for like, on cortex
    # under a translation spin, the markers give 0.072 with 3 of 7 significant rather than 0.23
    # with 6 of 7. The manuscript dropped the claim and removed the panel. The internal control
    # it uses instead is the eight BICCN cell-class maps, all on the same parcels and null. ----
    ("layer-marker control",
     "Grouping fourteen properties by their relation to the areal hierarchy, all nine tests in "
     "the\n\"hierarchy maps\" and \"varies along the hierarchy\" groups clear their spin nulls, "
     "and none of the\nfive orthogonal to it does. The controlling comparison sits within one "
     "dataset: individual\ncortical-layer marker genes, which retain areal variation, translate "
     "at mean r = 0.23 (6 of 7\nsignificant), whereas layer contrasts built from the same genes, "
     "which remove the shared areal\ncomponent, do not (mean r = 0.07). Granular L4 − "
     "infragranular is the expected exception, since\ncortical granularity is itself the areal "
     "hierarchy.",

     "Grouping fourteen properties by their relation to the areal hierarchy, all nine tests in "
     "the\n\"hierarchy maps\" and \"varies along the hierarchy\" groups clear their spin nulls, "
     "and none of the\nfive orthogonal to it does.\n\nThe comparison is internally controlled. "
     "Eight of the fourteen are cell-class maps scored on "
     "the\nsame 2,094 parcels against the same null, and they span the full range, from −0.03 for "
     "microglial\ndensity to +0.35 for the neuronal-glial contrast. What separates them is their "
     "relation to the\nareal hierarchy rather than how they were measured. Granular L4 minus "
     "infragranular is the\nexpected exception among the laminar contrasts, since cortical "
     "granularity is itself areal.\n\nAn earlier version of this README offered individual "
     "cortical-layer marker genes as the control,\nat mean r = 0.23 with 6 of 7 significant. That "
     "scored the markers over the whole brain against a\nnull that shuffled the coupling. Scored "
     "like for like, over Schaefer-400 cortex against a null that\nrotates the mouse input, the "
     "markers give 0.072 with 3 of 7 significant, and the dissociation does\nnot survive. The "
     "claim was withdrawn."),

    # ---- section 3. The gradient's spin p. ----
    ("gradient spin p",
     "| Principal FC gradient → human (Margulies / Huntenburg) | connectivity | \\|r\\| = 0.54, "
     "spin p = 0.032 |",
     "| Principal FC gradient → human (Margulies / Huntenburg) | connectivity | \\|r\\| = 0.54, "
     "translation spin p < 0.001, human-side spin p = 0.042 |"),

    ("worth noting",
     "Worth noting that the spatial scaffold is itself fitted to the Garin landmark pairs, so the",
     "The spatial scaffold is itself fitted to the Garin landmark pairs, so the"),

    # ---- repo map ----
    ("repo map",
     "├── experiments/         # Anchor-pack experiments + ablations\n"
     "├── notebooks/           # 8 walkthroughs: quickstart, methodology, one per figure",
     "├── experiments/         # Analyses, grouped by the manuscript section they support\n"
     "├── tools/               # Provenance, number and prose checks\n"
     "├── notebooks/           # 8 walkthroughs, in reading order"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report without writing")
    args = ap.parse_args()

    text = README.read_text()
    problems = []
    for label, old, new in EDITS:
        n = text.count(old)
        if n != 1:
            problems.append(f"{label}: found {n} occurrences, expected 1")
            continue
        if not args.check:
            text = text.replace(old, new)
        print(f"  {'would apply' if args.check else 'applied':12s} {label}")

    if problems:
        print("\nNot written. Targets did not match:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    if not args.check:
        README.write_text(text)
        print(f"\nwrote {README.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
