"""Step 5. Every mutation pattern, head motion and symptom severity.

Repeats the case-control test for every mouse mutation pattern in the TransBrain table on both
column mappings, so that every model is reported on the same footing. The Magel2 score is then
correlated with mean framewise displacement over the whole subset and, within the case group,
with ADOS total.

Inputs
    experiments/autism_subtypes/abide_subtype/abide_ho_core.py, which provides ``prep`` and
    ``scorer``, the push-forward ``route``, the motion-passing mask ``QC`` and the model
    patterns ``mvall`` with their names ``mcols``

Outputs, in .scratch/abide_ho/
    out_ctrl.json               summary statistics
    sc_obs_label_matched.npy    Magel2 scores on the label-matched mapping

Run from the repository root.
"""
exec(open('experiments/autism_subtypes/abide_subtype/abide_ho_core.py').read())
import json                                          # noqa: E402
from scipy.stats import spearmanr                    # noqa: E402

Gz, md = prep(QC, keep); f = scorer(Gz, near)
Gzo, mdo = prep(QC, POSITIONAL_COLS); fo = scorer(Gzo, nearo)
res = {}
for j, c in enumerate(mcols):
    T = route(mvall[j])
    sc, nr = f(T); r = test(sc, md); r['n_rois'] = nr; res['label_matched_' + c] = r
    sco, nro = fo(T); ro = test(sco, mdo); ro['n_rois'] = nro; res['positional_' + c] = ro
    print('%-10s label-matched d=%+.4f p=%.5f nroi=%d | positional d=%+.4f p=%.5f nroi=%d'
          % (c, r['cliffs'], r['p'], nr, ro['cliffs'], ro['p'], nro))
# motion and ADOS for Magel2 on the label-matched mapping
sc, _ = f(tpl); fd = md.func_mean_fd.to_numpy()
m = np.isfinite(sc) & np.isfinite(fd)
rho, pp = spearmanr(sc[m], fd[m])
res['motion'] = {'score_vs_fd_rho': float(rho), 'score_vs_fd_p': float(pp)}
a = md.ADOS_TOTAL.to_numpy(float); asd = (md.DX_GROUP.to_numpy() == 1)
mm = np.isfinite(a) & np.isfinite(sc) & asd
rr, pa = spearmanr(sc[mm], a[mm])
res['ados'] = {'n': int(mm.sum()), 'rho': float(rr), 'p': float(pa)}
print('motion: score~FD rho=%+.3f p=%.3g | ADOS rho=%+.3f p=%.3f n=%d' % (rho, pp, rr, pa, mm.sum()))
json.dump(res, open(WORK / 'out_ctrl.json', 'w'), indent=1)
np.save(WORK / 'sc_obs_label_matched.npy', sc)
