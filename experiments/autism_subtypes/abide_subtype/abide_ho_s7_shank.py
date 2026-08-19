"""Step 7. Rotation null for the Shank3 case-control effect.

Applies the rotation null of step 6 to the Shank3-- pattern on the label-matched column mapping.
Runs the full set of rotations in one pass and reports the observed p and Cliff's delta, the
calibrated p by p and by effect size, and the uncalibrated false positive rate.

Inputs
    experiments/autism_subtypes/abide_subtype/abide_ho_core.py, as in step 5

Outputs
    .scratch/abide_ho/out_shank.json   summary statistics

Run from the repository root.
"""
exec(open('experiments/autism_subtypes/abide_subtype/abide_ho_core.py').read())
import json, time                                    # noqa: E402

Gz, md = prep(QC, keep); f = scorer(Gz, near)
mv = mvall[mcols.index('Shank3--')]
c0 = mx - mx.mean(0)
rng = np.random.default_rng(0); nl = []
t0 = time.time()
for i in range(500):
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    perm = (((c0 @ Q.T)[:, None, :] - c0[None, :, :]) ** 2).sum(-1).argmin(1)
    r = test(f(route(mv[perm]))[0], md)
    nl.append([r['p'], abs(r['cliffs'])])
nl = np.array(nl); ob = test(f(route(mv))[0], md)
out = {'obs_p': ob['p'], 'obs_cliffs': ob['cliffs'],
       'rot_calibrated_p_by_p': float((nl[:, 0] <= ob['p']).mean()),
       'rot_calibrated_p_by_effect': float((nl[:, 1] >= abs(ob['cliffs'])).mean()),
       'uncalibrated_fpr_at_05': float((nl[:, 0] < 0.05).mean()), 'n_rot': 500,
       'secs': time.time() - t0}
print(json.dumps(out, indent=1))
json.dump(out, open(WORK / 'out_shank.json', 'w'), indent=1)
