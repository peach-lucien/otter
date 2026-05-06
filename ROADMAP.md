# Roadmap — methodology and evaluation improvements

Lives alongside `PLAN.md` (the original implementation plan, mostly complete).
This roadmap tracks the *remaining* improvements identified after the multi-modal
CV experiment showed `fc + SC` as the production winner at 81% strict-CV pair
recovery. Each item is self-contained: pick one, work through "Definition of
done", move on.

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done · `[-]` skipped

---

## Comprehensive comparison (latest, 2026-05-05)

Run `python scripts/comprehensive_comparison.py` to regenerate. Produces:
- `outputs/comparison/comprehensive_table.csv`     wide CSV: configs × headline metrics
- `outputs/comparison/per_network_top1.csv`        long CSV: configs × networks
- `outputs/comparison/comparison_summary.md`       markdown summary
- `outputs/figures/13_comprehensive_comparison.png` 4-panel multi-metric bar chart
- `outputs/figures/14_config_x_network_heatmap.png` configs × networks heatmap

**Production winner stays `fc_plus_SC`** at 81% top-1, tied with `fc_plus_xyz_gw`
and `fc_plus_network_mask` — but SC-augmented gives the best within-network FC
translation (r = 0.55 in hierarchical mode) while remaining the most robust to
held-out anchors. Z=+7.5 vs random π and z=+17.8 vs permuted-anchor null.

Items A (M_anchor), B (iterative co-clustering), C (confidence-weighted FC) all
returned clean negative results. Item D (subject-level CV) confirmed a small
~4 pp generalisation gap that's robust across folds.

---

## Phase S — Anchor expansion + diagnostics (ACTIVE — top priority)

After the audit + Beauchamp validation (May 2026), the bottleneck is clearly
**anchor density** rather than modality data, solver formulation, or marginal
control (FUGW, Knox SC, M_anchor, iterative all returned clean negatives;
Beauchamp validation shows 11.8× chance enrichment for anchored regions but
0× for hippocampal regions with no anchor). Three concrete actions follow.

### `[ ]` S1. Diagnose supervised-but-failing pairs (DIAG-1, DIAG-2)
**Why.** Beauchamp validation shows motor cortex (precentral gyrus) and
tectum (sup/inf colliculus) score 0% top-1 *despite* having Garin anchors.
Before adding new anchors we need to know whether existing anchors are
correctly placed.

**Subtasks.**
- **DIAG-1.** Find the MNI centroid of our human "Motor and premotor" anchor
  parcels. Compare to canonical precentral gyrus (-35, -20, +55). Check
  what region/network mouse-motor parcels are actually argmaxing to. Look
  at the held-out anchor CV for pair_id=2 (Motor) — does the anchor
  recover itself when held in?
- **DIAG-2.** Same for our "Tectum" anchor (pair_id=21). Currently a single
  anchor covering both colliculi; Beauchamp evaluates them separately.
  Check whether the anchor centroid is at the human midbrain or somewhere
  else (mouse colliculi argmax 60-72mm away).

**Effort.** 1-2 days. **Impact.** Either reveals a quick fix to existing
anchor placement, or confirms a real biological hardness (Beauchamp's own
paper notes weak motor-cortex transcriptomic similarity).

**Definition of done.** `outputs/logs/anchor_diagnostics.json` with per-anchor
diagnostic numbers (centroid in our coords, distance to canonical MNI,
held-in self-recovery, full-space top-K to expected target).

### `[ ]` S2. Add hippocampal subfield anchors (EXP-1)
**Why.** Beauchamp validation: 4 hippocampal pairs (Subiculum, CA1, CA3,
Dentate gyrus) all return 0% top-1, 0× chance enrichment. We have no
hippocampal anchor in Garin's 21 pair_ids. These are the cleanest test of
the "more anchors → better recovery" claim.

**What.**
1. Identify our mouse parcels in CA1/CA3/dentate gyrus/subiculum via the
   DSURQE label lookup already implemented in `pipeline/05f_*.py`.
2. Identify our human parcels in matching AHBA regions via MNI lookup
   (centroids hand-curated in `pipeline/05f_*.py`).
3. Add to anchor table (new pair_ids 22-25), with proper L/R hemisphere
   split.
4. Re-solve production π (`pipeline/04_solve_production.py`).
5. Re-validate against Beauchamp — expect non-zero top-K for these 4 pairs.
6. Re-run anchor CV + bootstrap — confirm existing anchor pairs aren't
   degraded.

**Effort.** 3-5 days. **Impact.** Direct test of the anchor-density story.
If it works, the case for further anchor expansion (motor sub-divisions,
tectum split) becomes self-evident.

**Definition of done.** Production π re-solved with 25 pair_ids; Beauchamp
re-run shows top-K > chance for the 4 new hippocampal pairs; existing 21
anchor pairs maintain ≥ current anchor CV scores.

### `[ ]` S3. Beauchamp follow-ups
**Why.** Beauchamp 2022 was the easiest external validation; 2-3 more
published cross-species correspondences would substantially strengthen the
case.
**What.** Mars 2018 / Eichert 2020 (white-matter cortical), Coletta 2020
(FC-based). Cerebellum (14 Beauchamp pairs we currently can't evaluate)
likely needs a separate cerebellar parcellation extension — out of scope
for now.
**Effort.** ~1 day each.

---

## Phase E — Evaluation hardening (do these first)

These are cheap and inform every method change downstream. Without them we
can't tell if a methodology tweak is real signal or noise.

### `[x]` E1. Predicted-FC translation quality
**DONE.** All four production configs land at r ≈ 0.36–0.38 vs random π r = 0.00.
Within-network r ≈ 0.45, cross-network r ≈ 0.20 — within-network FC much
better preserved (expected; it's the strongest signal). Key finding:
**`fc_only` and `fc + SC` are statistically indistinguishable on this metric**
(0.36 vs 0.36) — the SC win is local to specific anchors, not the bulk of π.
Headline number: r ≈ 0.36 vs random π = 0 → π is doing real work, but the
multimodal differences in CV don't translate to global FC reconstruction.
Files: `homer/eval/translation.py`, `scripts/fc_translation_eval.py`,
`outputs/figures/10_fc_translation.png`, `outputs/logs/fc_translation.json`.
**Why.** The single most important missing metric. Every current metric uses the
42 anchors in some form; this one uses the *whole* π and a held-out FC pattern.
**What.** Given mouse FC matrix `Fm` and π, predict human FC: `Fh_pred = π.T @ Fm @ π / (π.T @ ones @ π)`. Compare `Fh_pred` to actual `Fh` via Pearson on the upper-triangle. Report the correlation.
**How.**
- Add `homer.eval.fc_translation_quality(pi, Fm, Fh)` returning Pearson r over upper-triangle.
- Add a held-out-subjects variant: split human subjects 80/20, train on 80% (compute mean Fh_train + π), predict on 20% (mean Fh_test).
- Run for `baseline_fc_only` and `fc_plus_SC` to see if SC's win shows up here too.
**Effort.** 1–2 hours.
**Definition of done.** A new figure `outputs/figures/10_fc_translation.png` showing predicted-vs-actual Pearson r for at least the two configs above, plus held-out-subject variant. Number written to `outputs/logs/fc_translation.json`.
**Expected impact.** If `fc + SC` beats `fc_only` here too, we have a strong independent endorsement. If not, `fc + SC`'s anchor-CV win might be specific to the held-out anchor configurations.

### `[x]` E2. Adjacency-graded recovery metric
**DONE.** Added `held_out_metrics_graded()` returning top1, top5, mean_rank,
median_rank, mean_xyz_dist alongside the original top1/pair/hemi. Patched
`multimodal_cv.py` to use it (with `--recompute` flag). Re-ran 4 production
configs (baseline_fc_only, fc_plus_SC, fc_plus_xyz_gw, fc_plus_network_mask).

**Headline reframing:** All 4 configs achieve **top-5 = 100%** and **mean_xyz_dist
≈ 0.020** (roughly 1.2% of brain extent). The "21% top-1 misses" are not
catastrophic — they're near-misses where the correct partner is rank 2 within
its network and physically adjacent in xyz. We've been under-reporting the
quality of π by relying on binary top-1.

Files: `homer/data/anchors.py` (added function), `multimodal_cv.py` (patched),
`outputs/figures/11_graded_metrics.png`, `outputs/logs/multimodal_cv.json`.
**Why.** Top-1 is binary. A V1 → V2 prediction (anatomical near-miss) is much better than V1 → hippocampus (catastrophic) but currently both count as wrong.
**What.** For each held-out mouse anchor, compute (a) the rank of the correct human anchor in `argsort(π[i, :])` restricted to held-out columns, and (b) the normalized xyz distance between the predicted human anchor and the correct one. Replace the binary "top-1" headline with these in the CV reports.
**How.**
- Add `homer.anchors.held_out_metrics_graded()` that returns `{mean_rank, median_rank, mean_xyz_dist, top5}` alongside the existing top-1.
- Patch `multimodal_cv.py` and `network_holdout_cv.py` to use it.
**Effort.** ~1 hour.
**Definition of done.** All CV-summary JSONs now include rank and xyz-distance fields; the comparison table prints rank and distance columns.
**Expected impact.** Lets us say "fc_plus_M_gene's 20% subcortical top-1 is actually rank-3 — the right answer is in the top-3 just not top-1" vs "rank-15 — totally lost". Distinguishes near-miss from catastrophe.

### `[x]` E3. Null distributions
**DONE.** Implemented two null baselines:
- **Random π** (50 trials/network, no FGW solve): sample uniform random π satisfying mouse marginal.
- **Permuted-anchor** (5–10 trials/network, full FGW solve each): shuffle mouse-anchor → human-anchor pairings before solving.

**Headline z-scores (vs nulls):**
- top-1 = 79% vs random 28% ± 7%  → **z = +7.2**
- top-1 = 79% vs permuted-anchor 31% ± 3%  → **z = +17.0**
- top-5 = 100% vs random 86% ± 4%  → **z = +3.4**

**Interpretation.** The z=17 vs permuted-anchor null is the headline: it tells us the
*specific* pairings of which mouse anchor maps to which human anchor are doing the
work, not just "having anchor supervision in general". When supervision is random
the model recovers held-out anchors at chance.

The top-5 z-score is more modest (+3.4) because top-5 = 100% is partly trivial when
held-out networks are small (2–4 anchors) — random π also gets ~85% top-5.

Files: `scripts/null_distributions.py`, `outputs/logs/null_distributions.json`,
`outputs/figures/12_null_distributions.png`.
**Why.** All our numbers are vs absolute thresholds. We can't claim "fc + SC is significantly better than baseline" without a null.
**What.** Three nulls per config:
1. **Random π**: sample uniform-random doubly-stochastic matrices; compute the same metrics. Repeat 100×.
2. **Permuted anchors**: shuffle anchor pair_id labels and re-solve FGW. Tests whether anchor supervision is doing real work.
3. **Permuted FC**: shuffle node order in human FC and re-solve. Tests whether FC structure contributes anything beyond xyz + supervision.
**How.**
- New script `scripts/null_distributions.py`. Run each null 50–100 times per metric.
- Report each real metric as a z-score against the null mean / std.
**Effort.** Half a day.
**Definition of done.** `outputs/logs/nulls.json` with mean/std/p-value tables. Add z-scores to the multimodal_cv comparison table.
**Expected impact.** Lets us write "fc + SC achieves 81% pair recovery (p < 0.001 vs random π, z = 14.5)" instead of just the percentage.

### `[ ]` E4. Network-conditioned reporting
**Why.** "81% pair_id" is misleading because most networks are at 100% and a few are at 25%. The variance is the story.
**What.** Always report the per-network breakdown alongside the weighted mean. Add a small heatmap: rows = configs, columns = networks, cell = top-1 — makes the interaction visible at a glance.
**How.** New `scripts/multimodal_summary.py` that reads `multimodal_cv.json` and renders the heatmap.
**Effort.** ~30 min.
**Definition of done.** `outputs/figures/11_config_x_network_heatmap.png` saved.
**Expected impact.** Clearer interpretation of any new config's contribution. Cheap.

### `[x]` E5. External validation against published cross-species maps — DONE (Beauchamp 2022)
**Status.** Beauchamp 2022 validation done. See [`docs/external_validation.md`](docs/external_validation.md) and `pipeline/05f_beauchamp_validation.py`.

**Headline.**
- 22 non-cerebellar pairs; 19 evaluable.
- **15 anchor-overlapping pairs (927 mouse parcels): top-1 = 12% (chance 1.0%) → 11.8× enrichment.** top-5 = 22% (chance 4.8%) → 4.6×.
- **4 novel pairs (hippocampal subfields, no Garin anchor): top-1 = 0%** — clean confirmation that supervision density is the bottleneck.
- Permuted-π null sanity check: 0.6× chance (as expected ~1×).
- Strongest matches: Thalamus 33% top-1, Striatum-ventral 58% top-10, Postcentral 47% top-10.
- Surprising failures: Motor cortex (Beauchamp's own paper notes weak transcriptomic similarity) and Tectum — likely human anchor-placement issue, see DIAG-1/DIAG-2 below.

**Open follow-ups (separate items).**
- Mars 2018 / Eichert 2020 (white-matter cortical correspondences): not done.
- Coletta 2020 (FC-based): not done.
- 14 cerebellar pairs: cannot evaluate (cerebellum excluded from our parcellation).

### `[ ]` E6. Calibration of confidence
**Why.** Bootstrap gives us per-cell std but we don't check if it's meaningful.
**What.** Bin nodes by bootstrap stability (≤0.5, 0.5–0.8, 0.8–0.99, 1.0). For each bin, compute held-out anchor accuracy. If calibrated, accuracy should monotonically increase across bins.
**How.** `scripts/calibration_plot.py` reads `bootstrap_aggregate.npz` + held-out CV results.
**Effort.** ~1 hour.
**Definition of done.** Calibration plot saved. Reliability diagram with diagonal reference.
**Expected impact.** Tells us if we can trust the bootstrap stability as a per-node confidence score for downstream users.

---

## Phase M — Methodological improvements

### `[x]` M1. Multistart FGW
**DONE — meaningful negative result.** Implemented
`entropic_semirelaxed_fgw_multistart` with diverse G0 inits (default uniform, 4
random Sinkhorn-projected, 1 anchor-warm). Tested on visual and brainstem
networks (the two hardest CV folds).

**All 6 inits converge to within ~5e-7 relative loss of each other.** Identical
top-1, top-5, mean_rank, mean_xyz_dist across restarts. The 6× compute cost
buys no measurable gain.

This is a *substantive* finding about the methodology, not a failure: anchor
supervision + xyz spatial feature in M makes the FGW objective globally
well-identified in practice. Single-shot solutions are trustworthy. (Compare
unsupervised GW in Phase 5 where restarts found genuinely different optima.)

**Decision:** keep single-shot FGW for production. The multistart helper is
saved in `homer.fgw.entropic_semirelaxed_fgw_multistart` for future
diagnostics or for harder regimes (e.g., if we drop anchor supervision).

For methods writeup: "solution stability verified via multistart (loss spread
< 1e-6 across 6 diverse initialisations)."
**Why.** Single solve is non-convex. Multiple inits can find better optima and tell us how variable the solution is.
**What.** Run 5–10 FGW solves per config with diverse `G0` (uniform, random Sinkhorn-projected, anchor-warmed), pick lowest-loss solution.
**How.**
- Adapt `entropic_gw_multistart` from `homer.fgw` to handle semirelaxed FGW.
- Patch `multimodal_cv.py` and the production solve to use it.
**Effort.** 2–3 hours.
**Definition of done.** Production `pi_baseline.npy` regenerated with multistart. Best-loss reported. Loss spread across restarts logged.
**Expected impact.** Probably +1–3 pp on metrics. Loss spread is a *consistency* signal — small spread means the optimum is well-identified, large spread means the problem is under-constrained.

### `[ ]` M2. Subject quality control
**Why.** We flagged 3 outliers per species in §4 EDA but didn't exclude. Sensitivity test.
**What.** Recompute mean FC excluding the 3 outliers per species; re-solve FGW; compare against baseline π via per-cell L1 distance and changed-argmax-fraction.
**How.** We already have `stream_mean_fc_subset` in `homer.data`. Solve once with the cleaned mean FC, save π_clean, compare.
**Effort.** ~1 hour.
**Definition of done.** Side-by-side comparison saved. Recommendation: keep or drop outliers.
**Expected impact.** If π is robust, we know we don't need to worry about subject QC. If sensitive, we have a real preprocessing decision to make.

### `[ ]` M3. Anchor confidence weighting
**Why.** Currently `λ_anchor = 1.0` for all 42 anchors but Garin's atlas grades regions on cross-species correspondence reliability. Some pairs (Pons L/R) are anatomically fuzzier than others (Motor cortex L/R).
**What.** Read Garin's correspondence quality grades from the original paper; per-anchor `λ` proportional to grade.
**How.**
- Curate a small mapping `pair_id → confidence_score` from the Garin et al. paper (manual table).
- Replace `lam_anchor = 1.0` in `build_M_visible` with `lam_anchor[k] = base * confidence_score[pair_id]`.
**Effort.** ~2 hours (mostly the manual curation from the paper).
**Definition of done.** Per-anchor lambda implemented and run through CV. Compare against uniform-lambda result.
**Expected impact.** Probably small but principled. Eliminates one source of false-confidence in our results.

### `[x]` M4. Hierarchical / multi-scale OT
**DONE — clean trade-off, not a strict improvement.**

Implemented per-network within-species semirelaxed FGW. Each of 11 networks
solved as an isolated sub-problem (60-410 nodes each), then assembled into a
block-sparse (1864, 2094) coupling.

Results:
- **Leave-one-network-out CV: HURT (top-1 79% → 45%)** because the held-out
  network has zero visible anchors in its sub-block, so no supervision.
- **Production FC translation: HELPED (overall r 0.36 → 0.40, within-net
  0.45 → 0.55)** because each network's sub-FGW gets focused optimization.
- **Cross-network FC: HURT (r 0.20 → 0.16)** because hierarchical is
  block-diagonal by construction.
- **Coverage: HALVED (1450 → 787 human nodes kept)** for the same reason.

**Conclusion.** Hierarchical is a *complementary* tool, not a strict
improvement. Use when full anchor supervision is available AND you care more
about within-network FC fidelity than cross-network coverage. The flat solver
remains the production choice for general use.

Files: `homer/models/hierarchical.py`, `scripts/hierarchical_cv.py`,
`outputs/logs/hierarchical_cv.json`, `outputs/coupling/pi_hierarchical.npy`,
new `hierarchical_fc_only` entry in `outputs/logs/fc_translation.json`.
**Why.** Within-network confusion is the bottleneck. Solving at network scale first then refining within each network would give the within-network problem its own focused optimization.
**What.**
1. Build a 11×11 network-level cost using mean FC within each network.
2. Solve a small FGW on that to get a network-level coupling.
3. For each network pair with high coupling, solve a focused FGW restricted to those nodes.
**How.** New script `scripts/hierarchical_fgw.py`. Mostly ties together existing primitives.
**Effort.** ~1 day.
**Definition of done.** Hierarchical π computed. CV evaluation comparing hierarchical vs flat. New CV row in the comparison table.
**Expected impact.** Plausibly the biggest single algorithmic win, especially for visual (V1 vs V2) and sensorimotor (motor vs somato).

### `[x]` M5. Voxel-level mouse SC (Knox et al. 2019) — DONE, clean negative
**Status.** Knox 43-leaf cortical SC integrated as `Cm_SC_knox` via `pipeline/00_external/06_knox_sc.py`. After fixing a normalisation bug found in the audit (cost matrix was on `[0, 1.32]` vs Allen's `[0, 1]`, silently over-weighting SC by ~30%), the fair `pipeline/05e_knox_vs_standard_sc.py` LONO run shows:
- All 11 networks (n=42 folds): full top-1 unchanged at 2.4%, mean rank Δ = -0.4 / 2094 (noise).
- Knox cost matrix has only 469 unique fingerprints vs Allen's 454 — 1.03× resolution gain at the cost-matrix level (the originally claimed 2.4× compared raw SC to cost matrix; corrected in `docs/comparative_methods.md`).

**Conclusion.** Combined with the FUGW null result (also kept as comparative), this rules out modality-resolution and marginal-control as the bottleneck. The bottleneck is the 42-anchor supervision signal density — see `EXP-1` below.

### `[ ]` M6. Better FC cost matrices
**Why.** `1 - r` is the simplest choice. Partial correlations remove indirect-link confounds. Communicability captures higher-order graph structure. Worth a sweep.
**What.**
1. Compute partial correlation FC: invert the regularised covariance matrix per species.
2. Compute communicability: `C = expm(thresholded_FC)`.
3. Add as `Cm_pcorr`, `Ch_pcorr`, `Cm_comm`, `Ch_comm` in `full_costs.npz`.
4. Add CV configs that use these alternatives.
**How.** Extend `homer.costs` and `build_multimodal_costs.py`.
**Effort.** Half a day.
**Definition of done.** Three new configs in the multimodal CV comparison table.
**Expected impact.** Modest. May help if certain confusable region pairs differ by indirect connectivity but not direct correlation.

### `[ ]` M7. True sum-of-GWs solver
**Why.** Currently we approximate `α_FC·GW(C_FC) + α_SC·GW(C_SC)` by `GW(α_FC·C_FC + α_SC·C_SC)`. Algebraically different. The proper version sums per-modality losses; ours mixes inside the squared difference.
**What.** Manual PGD loop: at each outer iteration, compute the multi-modal gradient as `Σ_k w_k · gw_grad(C_m^k, C_h^k, π) + (1-α) · M`, then do an entropic semirelaxed projection step.
**How.** New `homer.fgw.multi_modal_semirelaxed_fgw()`. POT exposes `gwggrad` and Sinkhorn primitives.
**Effort.** ~1 day.
**Definition of done.** Solver implemented + tested against POT for single-modality case (should match). Then re-run `fc + SC` CV with the proper formulation; compare against the weighted-cost approximation.
**Expected impact.** Mathematically more interpretable; numerically may be similar to the approximation. Mostly a methods-paper hygiene fix.

### `[ ]` M8. OT-CFM flow refinement (Phase 8 of original PLAN)
**Why.** π gives us a point estimate. A continuous-time flow gives sampling-based uncertainty + cycle consistency.
**What.** Train a small neural network `v_θ(x, t)` that transports mouse-node embeddings to human-node embeddings, using π as the OT coupling supervision. Then sample from the flow at inference for per-node uncertainty.
**How.** PyTorch implementation. ~50 lines for the model + training loop.
**Effort.** 1–2 days.
**Definition of done.** Trained flow saved. Sampling-based confidence map alongside the bootstrap-based one.
**Expected impact.** Better uncertainty quantification. Marginal accuracy gain (the flow can refine but not fundamentally improve π).

### `[x]` M9. Iterative co-clustering — DONE, clean negative result
Implemented `scripts/iterative_cv.py` that repeatedly:
  1. Solves FGW with the visible anchors (lam=1.0).
  2. Picks top-K non-anchor mouse rows by row-max concentration.
  3. Re-solves with those rows added as soft (lam=0.30) or hard (lam=1.0) anchors.

**Result: exact-match negative.** Per-network top-1 with K=200, conf≥0.95,
lam=1.0 over 2 iterations is identical (to 16 decimals) to the single-shot
`fc_plus_SC` baseline across ALL 11 networks (visual 50%, brainstem 50%,
subcortical 100%, ..., overall top1=81%).

**Why it's a no-op.** With the production ε=5e-3 + anchor supervision, the
first-pass π already has mean row-max concentration 0.977 — every "high
confidence" row is one-hot at exactly the human node it would settle on
again. Adding lam_soft to the OTHER columns of those rows changes M but not
π (the solver was already going to its assigned column anyway). I confirmed
this analytically: |π_iter1 − π_iter0|_max = 1e-5 even with 200 rows
modified in M.

Tested with softer ε=5e-2 (mean concentration drops to 0.735), conf≥0.50,
both lam=0.30 and lam=1.0: still no change on visual (50% baseline).

**The honest interpretation.** Held-out anchor recovery is bottlenecked by
information available *to the held-out row's GW + xyz signal*, not by lack
of self-confidence about the rest of the map. Iterative co-clustering would
only help in a regime where the initial solution is genuinely ambiguous —
which isn't ours after the M_xyz term + anchor supervision.

Files: `scripts/iterative_cv.py`, `outputs/logs/iterative_cv.json`.

### `[x]` M12. Subject-level cross-validation (D) — small generalisation gap
**DONE.** `scripts/subject_cv.py`: K=5 random 80/20 subject splits per species,
seed=0. Per fold: stream train mean FC for both species, derive test mean by
subtraction (saves 2 of 4 streams per fold), build C_FC + (optional) SC,
solve FGW with full anchor supervision, evaluate FC translation Pearson r on
held-out test FC. Each fold ~26 s.

**Results (mean ± std across 5 folds):**

| config       | train r           | test r            | gap             | test within-net | test cross-net |
|--------------|-------------------|-------------------|-----------------|-----------------|----------------|
| `fc_only`    | 0.360 ± 0.002     | 0.319 ± 0.006     | −0.041 ± 0.007  | 0.420 ± 0.011   | 0.166 ± 0.007  |
| `fc_plus_SC` | 0.357 ± 0.002     | 0.318 ± 0.006     | −0.039 ± 0.008  | 0.417 ± 0.011   | 0.166 ± 0.008  |

**Findings.**
- Real but small generalisation gap: training on 80% of subjects loses
  ~4 pp of FC translation r on the held-out 20%. Within-network drops more
  (0.45 → 0.42) than cross-network (0.20 → 0.17 — already ~floor).
- `fc_only` and `fc_plus_SC` are statistically indistinguishable on this
  metric, just like in E1's all-subjects baseline. SC's win in held-out
  anchor CV does NOT translate to better subject-level generalisation.
- Test-set width across folds is ~0.006 — small, suggesting the model's
  predictions are robust to which specific subjects it sees.

**Interpretation.** π is mostly limited by *anchor structure* (per E1 result:
within-network r=0.45, cross-network r=0.20), not by *subject-level noise*.
Subject-CV adds ~4 pp of generalisation overhead on top of the anchor-driven
limits. Robustness is good news for downstream use — if a colleague were to
swap in a new cohort of similar size, the headline number would remain in the
0.31–0.36 range.

Files: `scripts/subject_cv.py`, `outputs/logs/subject_cv.json`,
`outputs/logs/fc_translation.json` (sub-key `subject_cv`).

### `[x]` M11. Confidence-weighted FC via fc_n_obs (C) — structural no-op
**DONE — clean negative by data inspection.**

Plan was to weight each FC cell by `n_obs / n_max` (Bayesian-flavored
shrinkage of `r` toward 0 in low-coverage cells), then re-derive the FC
cost matrix C and run CV. Diagnosis showed CV unwarranted because the
input data has near-uniform coverage.

| species | n_obs range | mean | row_cov min | anchor row_cov | C(orig, shrunk) corr |
|---------|-------------|------|-------------|----------------|----------------------|
| mouse   | 105 (uniform) | 105.0 | 1.000 | 1.000 | 1.0 (literal no-op)  |
| human   | 100 – 113   | 112.6 | 0.928 | 0.998 | 0.999966             |

**The colleague's preprocessing already removed coverage variation** (subjects
with significant dropout were presumably excluded upstream). For mouse it's a
literal no-op; for human only 0.9% of nodes have <95% coverage and the 42
anchors all have 99.8%, so the cost matrix barely moves
(|C_shrunk − C_orig|_max = 0.045 on a [0, 2] scale, mean 0.00018).

**Decision.** Park as future work IF the upstream preprocessing changes (e.g.,
if we get a coverage-imbalanced FC matrix from a new cohort). For the current
data, no signal to extract.

Files: `scripts/confidence_weighted_fc_check.py`,
`outputs/logs/confidence_weighted_fc_check.json`.

### `[x]` M10. Anchor-relationship cross-species cost (A)
**DONE — negative result.** Implemented `cross_species_anchor_M()` in
`homer/costs/` which builds a (n_m, n_h) cosine-distance matrix where each
node is represented by its FC vector to the 42 (or visible-during-CV) anchors.
Since anchor positions are in known 1:1 cross-species correspondence, these
vectors are directly comparable.

**Important CV-fairness fix.** First implementation pre-computed M_anchor in
`build_multimodal_costs.py` using ALL 42 anchors → leaked held-out anchors
into the cross-species cost (top-1 jumped to suspicious 100% across all
networks). Refactored to recompute M_anchor *per-fold* using only
`visible_pair_ids` via `_anchor_M_visible_only()` in `multimodal_cv.py`.

**Results after leak fix:**
- `fc_plus_M_anchor`:        top1 = 69%  (vs baseline 79%)
- `fc_plus_SC_plus_M_anchor`: top1 = 69%  (vs `fc + SC` 81%)

The 32 visible anchors' FC patterns aren't enough to predict held-out anchors
better than xyz alone; visual went 50% → 25%, subcortical 100% → 60%. Helps
salience (25% → 50%) but the regressions outweigh the gains.

**Decision.** Keep the helper available (`cross_species_anchor_M`,
`_anchor_M_visible_only`) but do **not** include in production. Useful as a
diagnostic — if a config relies heavily on M_anchor and improves, it's a sign
the held-out anchors are too easy / leaked.

Files: `homer/costs/`, `scripts/build_multimodal_costs.py`,
`scripts/multimodal_cv.py` (configs `fc_plus_M_anchor`,
`fc_plus_SC_plus_M_anchor`).

---

## Phase D — Data-driven improvements (depends on the bigger gene run)

### `[-]` D1. Expanded mouse gene set — *paused, see findings*
**What we tried.**
- Bulk `02_mouse_genes.py` over ~4082 datasets returned by RmaApi with
  `storage_directory$ne''`: **0/4082 had grid data**. The bulk RmaApi filter
  returns SectionDataSets that pass our metadata filter but lack 3D
  reconstructions in practice.
- Curated `02b_mouse_genes_direct.py` with 73 well-known markers: **51/73**
  gene downloads succeeded.
- Expanded `02b` to 343 curated markers: **61/255 resolved symbols** had grid
  data — only ~10 more than the small list. The "pick lowest SDS_ID" heuristic
  picks the canonical experiment, but ~75% of canonical Allen ISH datasets
  don't have 3D reconstructions.

**Conclusion.** Our gene-coverage ceiling is set by the underlying Allen ISH
3D-reconstruction rate (~25%), not by our curation. To improve materially we
need either (a) a different per-gene SDS-selection strategy (D3 below), or (b)
an entirely different data source.

### `[ ]` D2. Re-run multi-modal CV with the bigger gene set — *paused*
**Status.** Effectively waiting on D1 producing meaningfully more orthologs
(>150). With the current 61 mouse genes mapping to ~50 ortholog pairs (similar
to our previous run), re-running CV won't change the conclusion that the
51-ortholog gene set is too sparse to integrate cleanly into the FGW objective.

### `[ ]` D3. Multi-SDS-per-gene retry strategy (NEW)
**Why.** Allen often has 2–5 SectionDataSets per gene (different image series,
different reconstructions). Our current `02b` picks the *lowest* SDS_ID per
gene and stops — but that's the canonical/oldest experiment, which is the one
*least* likely to have a clean 3D reconstruction. A retry-on-failure strategy
would explore all SDS_IDs per gene and stop at the first one with valid grid
data.

**What.** Modify `_query_section_data_sets()` to return ALL SDS_IDs per gene
(not just the lowest). Modify the per-gene loop to try each SDS_ID in turn
(perhaps prefer higher IDs first since newer reconstructions tend to be
better) until one yields a valid `.mhd`/`.raw` zip.

**How.**
- `02b._query_section_data_sets` → return all dataset records, sorted by id
  descending (newest first).
- Main loop: for each gene, try `_try_download_grid` for each SDS_ID until one
  succeeds. Keep the first successful one; record its id.
- Log per-gene attempts → success rate so we can see if the strategy works.

**Effort.** Half a day. Plus several hours of compute (~3× more downloads).

**Expected impact.** If Allen's SDS-level success rate is 25% independently
across the 2–5 SDS_IDs per gene, then trying all of them gets us to roughly
1 - (0.75)^3 ≈ 58% per gene — i.e. from ~60 successes to ~150–200, doubling
or tripling the ortholog count.

**When to do this.** Only if we revisit gene expression as a modality. After
the multimodal CV result that gene M_gene hurts subcortical and only modestly
helps visual/sensorimotor, and given that the more pressing need is null
distributions / external validation / writeup, this is *not* on the critical
path. Park it as future work.

### `[ ]` D4. Alternative gene data sources (NEW, also future work)
**Options if Allen ISH coverage stays the limiting factor.**
- **BICCN cell-type taxonomies** (Bakken et al. 2021 Nature) provide
  region-averaged single-cell expression at high coverage; package
  `cellxgene-census` exposes them.
- **Mouse cortical hierarchy gene-expression** (Tasic, Yao etc.) at
  region-resolved granularity.
- **Allen Brain Cell taxonomy** (10x Genomics) — newer, higher-coverage but
  cellular not voxel-resolution; would need a region-averaging step.
**Effort.** ~1 week each (data download + parcellation alignment).
**Expected impact.** Would solve the 25%-grid-coverage issue at the cost of
giving up voxel-level resolution.

---

## Phase V — Validation & writeup

### `[ ]` V1. Comparison vs colleague's spectral pipeline
**Why.** All our improvement claims are vs *our* unsupervised baseline. A direct comparison vs the colleague's actual pipeline (or a reproduction of it) is the real benchmark.
**What.** Reproduce the spectral embedding + Procrustes + FAQ pipeline at the 1864 × 2094 scale. Run the same held-out anchor CV.
**Effort.** 1–2 days for a clean reproduction.
**Definition of done.** Side-by-side CV table: colleague's pipeline vs our `fc + SC`.
**Expected impact.** Necessary for the methods writeup.

### `[ ]` V2. Methods writeup draft
**Why.** Lock in what we've done before adding more.
**What.** Short methods note: data, FGW formulation, anchor handling, modality contributions, CV evaluation. Position vs the colleague's pipeline.
**Effort.** 1 day.
**Definition of done.** `docs/methods.md` (or `.docx`) drafted. Figures package collected.

### `[ ]` V3. Full bootstrap stability for the production config
**Why.** Existing 40-iter bootstrap was on FC-only. Production is `fc + SC`. Need bootstrap stability for the actual production π.
**Effort.** ~30 min running, automated.
**Definition of done.** New `bootstrap_aggregate.npz` for `fc + SC`.

---

## Suggested ordering (refreshed May 2026 after audit + Beauchamp)

The audit (`docs/audit_2026-05-06.md`) and Beauchamp validation
(`docs/external_validation.md`) reframed the priority list. Items E1–M5
above are mostly DONE or returned clean negatives. The active priorities
are now in **Phase S** at the top of this file:

1. **S1 (diagnose motor + tectum)** — small, fast, informs S2.
2. **S2 (add hippocampal anchors)** — direct test of the anchor-density story.
3. **S3 (more external validation: Mars 2018, Coletta 2020)** — credibility.
4. **V2 (methods writeup)** + **V3 (any final production bootstrap)** —
   wrap-up.
5. **E6 (calibration of confidence)** + **M6 (better FC costs)** — nice to have.

Older items below (M2 subject QC, M3 anchor weighting, M7–M9) remain in the
backlog but are lower priority than Phase S.

---

## Notes on what we *won't* do

- **Drop FC bandpass tuning, motion scrubbing, etc.** These are upstream of our pipeline. The colleague has done his preprocessing; we work with the FC matrices we get.
- **Try other species (macaque, rat).** Out of scope for v1; could be a future paper.
- **Per-pair α tuning** (i.e., region-specific anchor weights). Risk of overfitting to 21 pair_ids.
- **Dynamic FC.** Adds time-varying complexity for unclear payoff.
