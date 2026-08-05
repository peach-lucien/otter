#!/usr/bin/env python3
"""Fetch Neurosynth association maps for DISEASE terms, one MNI152 volume per disease, into
disease_maps/<disease>.nii.gz. Reuses the exact machinery of 00_fetch_maps.py (same
Neurosynth v7 database, same association-test map), just with disease terms.

Run (after 00_fetch_maps.py has downloaded the Neurosynth db once):
    cd otter && python experiments/reverse_translation/00b_fetch_disease_maps.py
Re-runnable; skips diseases whose map already exists.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_s = importlib.util.spec_from_file_location("f0", ROOT / "experiments/reverse_translation/00_fetch_maps.py")
f0 = importlib.util.module_from_spec(_s); _s.loader.exec_module(f0)

OUT = ROOT / "experiments/reverse_translation/disease_maps"; OUT.mkdir(parents=True, exist_ok=True)

# disease -> candidate Neurosynth term spellings (first present with studies is used)
DISEASE_TERMS = {
    "parkinson":     ["parkinson", "parkinsons", "parkinson disease"],
    "alzheimer":     ["alzheimer", "alzheimers", "alzheimer disease"],
    "huntington":    ["huntington", "huntingtons", "huntington disease"],
    "epilepsy":      ["epilepsy", "epileptic", "seizure"],
    "addiction":     ["addiction", "craving", "drug"],
    "schizophrenia": ["schizophrenia", "schizophrenic", "psychosis"],
    "obsessive":     ["obsessive", "compulsive", "ocd"],
    "autism":        ["autism", "autistic", "autism spectrum"],
}


def main():
    todo = {d: terms for d, terms in DISEASE_TERMS.items() if not (OUT / f"{d}.nii.gz").exists()}
    if not todo:
        print("all disease maps already present in", OUT); return
    dset = f0.build_dataset()
    print(f"dataset: {len(dset.ids)} studies, {len(dset.get_labels())} term labels\n")
    for d, terms in todo.items():
        lab, used = None, None
        for term in terms:
            lab = f0.find_label(dset, term)
            if lab:
                used = term; break
        if not lab:
            print(f"  {d:14s} NO Neurosynth term matched {terms} — edit DISEASE_TERMS"); continue
        ids = dset.get_studies_by_label(labels=[lab], label_threshold=f0.LABEL_THRESHOLD)
        if len(ids) < 30:
            print(f"  {d:14s} term '{used}' has only {len(ids)} studies — weak; consider another term")
        img, mapkey = f0.association_map(dset, ids)
        img.to_filename(str(OUT / f"{d}.nii.gz"))
        print(f"  {d:14s} term '{used}' ({len(ids)} studies) map '{mapkey}' -> {d}.nii.gz")
    print("\ndone. Disease maps in", OUT)


if __name__ == "__main__":
    main()
