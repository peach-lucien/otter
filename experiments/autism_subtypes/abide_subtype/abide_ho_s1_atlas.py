"""Step 1. Centroid tables for the Harvard-Oxford atlases behind the ABIDE ROI columns.

The ABIDE PCP rois_ho derivative labels every time-series column with a Harvard-Oxford id. This
step takes the centroid of each atlas label in MNI mm and records the rule that maps a
hemisphere-split cortical label L to its ABIDE id, ((L + 1) // 2) * 100 plus 1 for odd L and 2
for even L. It prints the coverage of that rule over the cortical ids and the mean x of each
hemisphere group, so the hemisphere codes can be read off the coordinates.

Inputs, from $OTTER_ABIDE_BUNDLE (default data_external/abide_ho/), all written by step 0
    abide_G.npz                          ROI ids of the connectivity bundle
    HO-cort-maxprob-thr25-2mm.nii.gz     bilateral cortical atlas
    HO-cortl-maxprob-thr25-2mm.nii.gz    hemisphere-split cortical atlas
    HO-sub-maxprob-thr25-2mm.nii.gz      subcortical atlas
    ho_labels.json                       subcortical label names

Outputs, in .scratch/abide_ho/
    cortl_cent.npy          centroids of the hemisphere-split cortical labels
    cortl_rule.json         cortical label to ABIDE id
    cort_unsplit_cent.npy   centroids of the bilateral cortical labels
    sub_cent.npy            centroids of the subcortical labels

Run from the repository root.
"""
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np


def _repo_root():
    p = Path.cwd().resolve()
    for q in [p, *p.parents]:
        if (q / "src" / "otter").is_dir():
            return q
    raise SystemExit("run this from inside the OTTER repository")


ROOT = _repo_root()
BUNDLE = Path(os.environ.get("OTTER_ABIDE_BUNDLE", ROOT / "data_external/abide_ho"))
WORK = ROOT / ".scratch/abide_ho"
WORK.mkdir(parents=True, exist_ok=True)
if not (BUNDLE / "abide_G.npz").exists():
    raise SystemExit("no abide_G.npz in %s; run abide_ho_s0_bundle.py first" % BUNDLE)

d = np.load(BUNDLE / "abide_G.npz", allow_pickle=True)
ids = d['roi_ids']; print('n ids', len(ids))
sub_ids = [i for i in ids if i < 100]; cort_ids = [i for i in ids if i >= 100]
print('subcortical ids (%d):' % len(sub_ids), sub_ids)
print('cortical-range ids (%d)' % len(cort_ids))
odd = [i for i in cort_ids if i % 100 not in (1, 2)]
print('cortical-range ids with hemi code not in {1,2}:', odd)


def centroids(path):
    im = nib.load(str(path)); dat = np.asarray(im.dataobj); aff = im.affine
    out = {}
    for L in np.unique(dat):
        if L == 0:
            continue
        ijk = np.argwhere(dat == L).mean(0)
        out[int(L)] = nib.affines.apply_affine(aff, ijk)
    return out


# ---- cortl (96 split labels) ----
cl = centroids(BUNDLE / 'HO-cortl-maxprob-thr25-2mm.nii.gz')
print('cortl labels:', len(cl))
# correspondence rule: cortl label L (1..96) -> id = ((L+1)//2)*100 + (1 if L odd else 2)
rule = {L: ((L + 1) // 2) * 100 + (1 if L % 2 == 1 else 2) for L in cl}
inv = {v: k for k, v in rule.items()}
print('rule covers all 96 cortical .1D ids:', set(inv) == set(i for i in cort_ids if i % 100 in (1, 2)))
x1 = [cl[L][0] for L in cl if rule[L] % 100 == 1]; x2 = [cl[L][0] for L in cl if rule[L] % 100 == 2]
print('ids ending 1: n=%d mean x=%+.2f  max x=%+.2f' % (len(x1), np.mean(x1), np.max(x1)))
print('ids ending 2: n=%d mean x=%+.2f  min x=%+.2f' % (len(x2), np.mean(x2), np.min(x2)))
np.save(WORK / 'cortl_cent.npy', np.array([cl[L] for L in sorted(cl)]))
json.dump({str(L): rule[L] for L in sorted(cl)}, open(WORK / 'cortl_rule.json', 'w'))

# ---- bilateral cortical labels (48), used by the positional comparison arm ----
co = centroids(BUNDLE / 'HO-cort-maxprob-thr25-2mm.nii.gz')
print('cort (unsplit) labels:', len(co))
mid = [L for L in co if abs(co[L][0]) < 5.0]
print('unsplit cort centroids within 5mm of midline: %d/48' % len(mid))
np.save(WORK / 'cort_unsplit_cent.npy', np.array([co[L] for L in sorted(co)]))

# ---- subcortical ----
so = centroids(BUNDLE / 'HO-sub-maxprob-thr25-2mm.nii.gz')
labs = json.load(open(BUNDLE / 'ho_labels.json'))['sub']
print('sub labels:', len(so), 'names:', len(labs))
for L in sorted(so):
    print('  %2d %-30s x=%+7.2f' % (L, labs[L], so[L][0]))
np.save(WORK / 'sub_cent.npy', np.array([so[L] for L in sorted(so)]))
