"""Step 6. Rotation null for the Magel2 case-control effect.

Draws random rotations of the mouse coordinate frame, relabels each mouse parcel with the value
of its nearest rotated neighbour, routes the rotated pattern through the coupling and repeats
the case-control test. The null therefore holds the spatial autocorrelation of the mouse pattern
fixed while breaking its correspondence with anatomy. Reported for both column mappings: the
observed p and Cliff's delta, the calibrated p by p and by effect size, the uncalibrated false
positive rate, and the median of the null.

The run is resumable. Completed draws are kept in rot_state.npz and the generator is replayed to
the recorded draw on restart, so the sequence of rotations does not depend on how the work is
split across calls. The optional argument is the wall-clock budget in seconds for one call. The
summary is written once the full number of rotations has been reached.

Inputs
    experiments/autism_subtypes/abide_subtype/abide_ho_core.py, as in step 5
    .scratch/abide_ho/rot_state.npz, when resuming

Outputs, in .scratch/abide_ho/
    rot_state.npz    completed draws
    out_rot.json     summary statistics, once the run is complete

Run from the repository root.
"""
import sys
exec(open('experiments/autism_subtypes/abide_subtype/abide_ho_core.py').read())
import json, os, time                                # noqa: E402

Gz, md = prep(QC, keep); f = scorer(Gz, near)
Gzo, mdo = prep(QC, POSITIONAL_COLS); fo = scorer(Gzo, nearo)
mv = mvall[mcols.index('Magel2')]
c0 = mx - mx.mean(0)
ST = WORK / 'rot_state.npz'
done = 0; rows = []
if ST.exists():
    z = np.load(ST); rows = z['rows'].tolist(); done = len(rows)
N = 500
BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 32.0
rng = np.random.default_rng(0)
for i in range(done):                      # replay RNG to the exact state
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
t0 = time.time()
while done < N and time.time() - t0 < BUDGET:
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    perm = (((c0 @ Q.T)[:, None, :] - c0[None, :, :]) ** 2).sum(-1).argmin(1)
    T = route(mv[perm])
    s1, _ = f(T); r1 = test(s1, md)
    s2, _ = fo(T); r2 = test(s2, mdo)
    rows.append([r1['p'], abs(r1['cliffs']), r2['p'], abs(r2['cliffs'])]); done += 1
np.savez(ST, rows=np.array(rows))
print('rotations done: %d/%d  (%.1fs)' % (done, N, time.time() - t0))
if done >= N:
    nl = np.array(rows)
    obs_c = test(f(tpl)[0], md); obs_o = test(fo(tpl)[0], mdo)
    out = {}
    for tag, (oc_, i0, i1) in {'label_matched': (obs_c, 0, 1), 'positional': (obs_o, 2, 3)}.items():
        out[tag] = {'obs_p': oc_['p'], 'obs_cliffs': oc_['cliffs'],
                    'rot_calibrated_p_by_p': float((nl[:, i0] <= oc_['p']).mean()),
                    'rot_calibrated_p_by_effect': float((nl[:, i1] >= abs(oc_['cliffs'])).mean()),
                    'uncalibrated_fpr_at_05': float((nl[:, i0] < 0.05).mean()),
                    'median_null_p': float(np.median(nl[:, i0])),
                    'median_null_abs_cliffs': float(np.median(nl[:, i1])), 'n_rot': int(len(nl))}
        print(tag, json.dumps(out[tag], indent=1))
    json.dump(out, open(WORK / 'out_rot.json', 'w'), indent=1)
