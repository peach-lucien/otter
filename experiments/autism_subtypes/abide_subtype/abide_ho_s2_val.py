"""Step 2. Checks on the atlas to ABIDE column correspondence.

Matches every ABIDE ROI id to a Harvard-Oxford centroid, subcortical ids through their
FreeSurfer-style names and cortical ids through the split-label rule of step 1, and reports how
many columns match. Three checks follow on the grand-mean connectivity matrix: the rank
correlation between connectivity and Euclidean distance, the rank each cortical region's
homotopic partner takes in its own connectivity row against the rank of a random contralateral
region, and the strongest partners of the one column that carries no atlas label.

Inputs
    $OTTER_ABIDE_BUNDLE/abide_G.npz     ROI ids and the grand-mean connectivity matrix
    $OTTER_ABIDE_BUNDLE/ho_labels.json  subcortical label names
    .scratch/abide_ho/cortl_rule.json, cortl_cent.npy, sub_cent.npy   from step 1

Outputs, in .scratch/abide_ho/
    cent_ids.npy    ABIDE ids that carry a centroid, in column order
    cent_xyz.npy    their centroids

Run from the repository root.
"""
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


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

d = np.load(BUNDLE / 'abide_G.npz', allow_pickle=True)
ids = list(map(int, d['roi_ids'])); FC = d['grand_fc_z'].astype(float)
labs = json.load(open(BUNDLE / 'ho_labels.json'))
rule = json.load(open(WORK / 'cortl_rule.json')); rule = {int(k): v for k, v in rule.items()}
inv = {v: k for k, v in rule.items()}
cl = np.load(WORK / 'cortl_cent.npy'); cl = {L: cl[i] for i, L in enumerate(sorted(rule))}
so = np.load(WORK / 'sub_cent.npy'); so = {L + 1: so[L] for L in range(len(so))}
# FS id -> HO sub name
fsmap = {10: 'Left Thalamus', 11: 'Left Caudate', 12: 'Left Putamen', 13: 'Left Pallidum',
         17: 'Left Hippocampus', 18: 'Left Amygdala', 26: 'Left Accumbens',
         49: 'Right Thalamus', 50: 'Right Caudate', 51: 'Right Putamen', 52: 'Right Pallidum',
         53: 'Right Hippocampus', 54: 'Right Amygdala', 58: 'Right Accumbens'}
name2lab = {labs['sub'][L]: L for L in range(1, 22)}
cent = {}; unmatched = []
for i in ids:
    if i in fsmap:
        nm = fsmap[i]
        if nm in name2lab:
            cent[i] = so[name2lab[nm]]
        else:
            unmatched.append((i, 'sub name %r not in atlas' % nm))
    elif i in inv:
        cent[i] = cl[inv[i]]
    else:
        unmatched.append((i, 'no cortl label'))
print('matched %d/%d ; unmatched: %s' % (len(cent), len(ids), unmatched))

# ---- check 1. connectivity against euclidean distance ----
mi = [k for k, i in enumerate(ids) if i in cent]
C = np.array([cent[ids[k]] for k in mi])
D = np.sqrt(((C[:, None] - C[None]) ** 2).sum(-1))
sub = FC[np.ix_(mi, mi)]
iu = np.triu_indices(len(mi), 1)
print('FC vs distance spearman rho = %+.3f (p=%.2g)' % spearmanr(sub[iu], D[iu]))

# ---- check 2. homotopy. for each cortical id, the rank of its homotopic partner ----
ranks = []
cids = [i for i in ids if i >= 100 and i % 100 in (1, 2)]
pos = {i: k for k, i in enumerate(ids)}
for i in cids:
    hom = i + 1 if i % 100 == 1 else i - 1
    row = FC[pos[i]].copy(); row[pos[i]] = -np.inf
    order = np.argsort(-row)
    ranks.append(int(np.where(order == pos[hom])[0][0]) + 1)
ranks = np.array(ranks)
print('homotopic partner rank among %d others: median=%.1f  rank1=%d/%d  top3=%d/%d'
      % (len(ids) - 1, np.median(ranks), (ranks == 1).sum(), len(cids), (ranks <= 3).sum(), len(cids)))
# null: rank of a random other-hemisphere region
rng = np.random.default_rng(0); nr = []
for i in cids:
    row = FC[pos[i]].copy(); row[pos[i]] = -np.inf
    order = np.argsort(-row)
    others = [j for j in cids if j != i and j % 100 != i % 100]
    j = others[rng.integers(len(others))]
    nr.append(int(np.where(order == pos[j])[0][0]) + 1)
print('  random contralateral-region rank: median=%.1f' % np.median(nr))

# ---- check 3. the unlabelled column ----
unl = [i for i, _ in unmatched]
for i in unl:
    p = pos[i]; row = FC[p].copy(); row[p] = -np.inf
    top = np.argsort(-row)[:6]
    print('#%d top FC partners:' % i, [(ids[t], round(float(row[t]), 3)) for t in top])
np.save(WORK / 'cent_ids.npy', np.array([i for i in ids if i in cent]))
np.save(WORK / 'cent_xyz.npy', np.array([cent[i] for i in ids if i in cent]))
