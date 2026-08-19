"""Step 4. ABIDE case-control test of a translated mouse template.

Scores every participant against the Magel2 template, the worked example cached by step 3, on
both column mappings, the label-matched centroid table and the positional arm, for the
motion-passing subset and for the whole sample. Each subset is reported unadjusted, as a
Mann-Whitney U with Cliff's delta, and adjusted by ordinary least squares for diagnosis, mean
framewise displacement, age and sex. Step 5 repeats the test for every model in the table. The
diagnostics carried in ``DIAG`` are written alongside, together with the number of the 48
bilateral assignments that agree with a label-matched one in either hemisphere and in both.

Inputs
    experiments/autism_subtypes/abide_subtype/abide_ho_core.py, and everything it loads

Outputs, in .scratch/abide_ho/
    sc_qc871_label_matched.npy, sc_all1035_label_matched.npy   per-subject scores
    out_main.json                                              summary statistics

Run from the repository root.
"""
exec(open('experiments/autism_subtypes/abide_subtype/abide_ho_core.py').read())
import json                                          # noqa: E402
from scipy.stats import t as tdist                   # noqa: E402


def ols(y, Xd):
    Xd = np.column_stack([np.ones(len(y)), Xd])
    b, _, _, _ = np.linalg.lstsq(Xd, y, rcond=None)
    r = y - Xd @ b
    dof = len(y) - Xd.shape[1]
    s2 = (r @ r) / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(Xd.T @ Xd)))
    tv = b / se
    return b, 2 * tdist.sf(np.abs(tv), dof)


ALL = np.ones(len(ph), bool)
OUT = {'diag': DIAG}
# agreement variants
both = sum(1 for c in range(1, 49)
           if len(set(corr_by_cort.get(c, []))) == 1 and nearo[c - 1] == corr_by_cort[c][0])
OUT['diag']['positional']['agree_either_hemisphere_of_48'] = int(agree)
OUT['diag']['positional']['agree_both_hemispheres_of_48'] = int(both)


def adj(sc, md):
    Xd = np.column_stack([(md.DX_GROUP.to_numpy() == 1).astype(float), md.func_mean_fd.to_numpy(),
                          md.AGE_AT_SCAN.to_numpy(), (md.SEX.to_numpy() == 1).astype(float)])
    ok = np.isfinite(sc) & np.isfinite(Xd).all(1)
    b, p = ols(sc[ok], Xd[ok])
    return {'n': int(ok.sum()), 'dx_beta': float(b[1]), 'dx_p': float(p[1]),
            'fd_beta': float(b[2]), 'fd_p': float(p[2])}


for tag, mask in [('qc%d' % int(QC.sum()), QC), ('all%d' % len(ph), ALL)]:
    sc, nr, md = run(mask, tpl, near, keep)
    r = test(sc, md); r['n_rois'] = nr; r['adj'] = adj(sc, md)
    OUT[tag + '_label_matched'] = r
    sco, nro, mdo = run(mask, tpl, nearo, POSITIONAL_COLS)
    ro = test(sco, mdo); ro['n_rois'] = nro; ro['adj'] = adj(sco, mdo)
    OUT[tag + '_positional'] = ro
    np.save(WORK / ('sc_%s_label_matched.npy' % tag), sc)
    print(tag, 'label-matched  cliffs=%+.4f p=%.5f nroi=%d adjp=%.5f'
          % (r['cliffs'], r['p'], nr, r['adj']['dx_p']))
    print(tag, 'positional     cliffs=%+.4f p=%.5f nroi=%d adjp=%.5f'
          % (ro['cliffs'], ro['p'], nro, ro['adj']['dx_p']))
json.dump(OUT, open(WORK / 'out_main.json', 'w'), indent=1)
