#!/usr/bin/env python3
"""Fetch volumetric human functional maps (Neurosynth association tests) for reverse
translation, one MNI152 volume per function, saved to human_maps/<function>.nii.gz.

WHY VOLUMETRIC NEUROSYNTH (not surface neuromaps): most reverse-translation targets
are SUBCORTICAL (accumbens, amygdala, hypothalamus, PAG). Neurosynth association maps
are native MNI152 with full subcortical coverage; cortical-surface annotations are not
usable here.

This uses NiMARE. The API drifts between versions, so the script tries several candidate
term spellings per function and several candidate map keys, and prints what it used.
If a step fails in your NiMARE version, the fix is almost always a single name in
FUNCTION_TERMS or MAP_KEYS below.

Run once (downloads the Neurosynth v7 database, ~a few hundred MB, then builds 12 maps):
    cd homer && python experiments/reverse_translation/00_fetch_maps.py
Re-runnable; skips functions whose map already exists.
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/reverse_translation/human_maps"; OUT.mkdir(parents=True, exist_ok=True)
NSDIR = ROOT / "data_external/neurosynth"; NSDIR.mkdir(parents=True, exist_ok=True)
LABEL_THRESHOLD = 0.001

# function -> candidate Neurosynth term spellings (first one present with studies is used)
FUNCTION_TERMS = {
    "reward":        ["reward", "reinforcement", "incentive"],
    "fear":          ["fear", "threat", "aversive"],
    "anxiety":       ["anxiety", "anxious"],
    "feeding":       ["feeding", "eating", "food", "appetite"],
    "spatial_memory":["navigation", "spatial memory", "spatial"],
    "motor":         ["motor", "movement", "motor control"],
    "addiction":     ["addiction", "craving", "drug"],
    "pain":          ["pain", "nociception", "painful"],
    "olfaction":     ["olfactory", "smell", "odor"],
    "vision":        ["visual", "vision"],
    "audition":      ["auditory", "sound", "hearing"],
    "interoception": ["interoception", "autonomic", "visceral"],
}
# association-test (reverse inference) preferred; forward/uniformity as fallback
MAP_KEYS = ["z_desc-specificity", "z_desc-associationResponse", "z_desc-association",
            "z_desc-consistency", "z_desc-uniformity"]


def build_dataset():
    """Robust across NiMARE versions: run the downloader for its side-effect, then build
    the Dataset from the downloaded files (stable), not from the ambiguous return value
    (dict in old NiMARE, Studyset in new). File-first guarantees the tf-idf term labels."""
    from nimare.extract import fetch_neurosynth
    from nimare.io import convert_neurosynth_to_dataset
    print("fetching Neurosynth v7 (downloads once) ...")
    ret = None
    try:
        ret = fetch_neurosynth(data_dir=str(NSDIR), version="7", overwrite=False,
                               source="abstract", vocab="terms")
    except Exception as e:
        print("  fetch_neurosynth raised:", e, "(will look for any files already downloaded)")

    def find(*patterns):
        for p in patterns:
            hits = sorted(NSDIR.rglob(p))
            if hits:
                return str(hits[0])
        return None

    coords = find("*coordinates.tsv.gz", "*coordinates*")
    meta = find("*metadata.tsv.gz", "*metadata*")
    feats = find("*features.npz", "*features*.npz", "*tfidf*.npz")
    vocab = find("*vocabulary.txt", "*vocab*.txt")
    print(f"  coordinates: {coords}\n  metadata: {meta}\n  features: {feats}\n  vocab: {vocab}")
    if coords and meta and feats and vocab:
        return convert_neurosynth_to_dataset(
            coordinates_file=coords, metadata_file=meta,
            annotations_files=[{"features": feats, "vocabulary": vocab}])

    # fallback: use the returned object if it can become a Dataset (may lack term labels)
    obj = ret[0] if isinstance(ret, (list, tuple)) and ret else ret
    if hasattr(obj, "to_dataset"):
        print("  building from returned Studyset via .to_dataset()")
        return obj.to_dataset()
    raise FileNotFoundError(
        f"could not locate Neurosynth coordinate/metadata/feature/vocab files under {NSDIR}. "
        f"Run `ls -R {NSDIR}` and adjust the glob patterns in build_dataset().")


def find_label(dset, term):
    labels = dset.get_labels()
    t = term.lower().replace(" ", "")
    for lab in labels:
        if lab.lower().replace(" ", "").endswith("__" + t):
            return lab
    return None


def association_map(dset, ids):
    from nimare.meta.cbma.mkda import MKDAChi2
    other = sorted(set(dset.ids) - set(ids))
    res = MKDAChi2().fit(dset.slice(ids), dset.slice(other))
    for key in MAP_KEYS:
        try:
            return res.get_map(key, return_type="image"), key
        except Exception:
            continue
    # last resort: first available map
    k = list(res.maps.keys())[0]
    return res.get_map(k, return_type="image"), k


def main():
    todo = {fn: terms for fn, terms in FUNCTION_TERMS.items()
            if not (OUT / f"{fn}.nii.gz").exists()}
    if not todo:
        print("all maps already present in", OUT); return
    dset = build_dataset()
    labels_all = dset.get_labels()
    print(f"dataset: {len(dset.ids)} studies, {len(labels_all)} term labels\n")
    for fn, terms in todo.items():
        lab, used = None, None
        for term in terms:
            lab = find_label(dset, term)
            if lab:
                used = term; break
        if not lab:
            print(f"  {fn:14s} NO Neurosynth term matched {terms} — edit FUNCTION_TERMS"); continue
        ids = dset.get_studies_by_label(labels=[lab], label_threshold=LABEL_THRESHOLD)
        if len(ids) < 30:
            print(f"  {fn:14s} term '{used}' has only {len(ids)} studies — weak; consider another term")
        img, mapkey = association_map(dset, ids)
        img.to_filename(str(OUT / f"{fn}.nii.gz"))
        print(f"  {fn:14s} term '{used}' ({len(ids)} studies) map '{mapkey}' -> {fn}.nii.gz")
    print("\ndone. Human maps in", OUT)


if __name__ == "__main__":
    main()
