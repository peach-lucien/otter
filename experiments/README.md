# Experiments

Reproducible one-off experiments. The `src/homer/` library is the production code; everything here is research scripts that produced a specific result documented in `docs/`.

## Layout

```
experiments/
├── anchor_packs/          # Per-pack experiment runners (default + opt-in)
├── ablations/             # Methodology ablations (soft anchors, marginals, xyz)
├── archive/               # Older detours superseded by the current pipeline
└── outputs/               # Cached intermediate results from these runs
```

## anchor_packs/ — region-anchor experiments

Each script fits the production point-anchor π plus one anchor pack and reports the Beauchamp delta. They're the experiments behind `docs/04_anchor_packs.md`.

| Script | Pack | Pids | Reference |
|---|---|---|---|
| `biccn_motor.py` | M1 + M2 | 30, 31 | Bakken 2021 *Nature* |
| `tectum.py` | SC + IC | 32, 33 | May 2006; Schreiner 2007 |
| `olfactory.py` | Piriform + AON | 34, 35 | Mori 2014; Carlén 2017 |
| `cingulate.py` | sgACC + RSC (opt-in) | 36, 37 | Vogt 2019 |
| `amygdala.py` | Cortical subplate | 38 | Janak & Tye 2015 |
| `hippocampal.py` | Subi + CA1 + CA3 + DG | 39-42 | Strange 2014 |
| `lateral_pfc.py` | OFC + dlPFC (opt-in for dlPFC) | 45, 46 | Wallis 2012; Carlén 2017 |
| `compose_all.py` | All default packs (headline result) | 30-35, 38-42 | — |

Run any with:
```bash
PYTHONPATH=src python experiments/anchor_packs/<name>.py
```

`compose_all.py` is the "headline result" experiment that produces the recommended production-with-packs π. See `docs/03_results.md` for the numbers.

## ablations/ — methodology ablations

Three ablations that justify or contextualise design choices in the production model. Each produced a documented result in `docs/archive/iteration_log.md`.

| Script | Question | Result |
|---|---|---|
| `soft_region_anchors.py` | Hard 0/1 wall vs soft `lam_outside < 1` constraint? | Soft is better-calibrated (43% lower mean rank); soft `0.15` is now default. |
| `marginal_weighting.py` | Does volume- or stability-weighted source marginal help? | No (0.0 pp difference). Uniform is fine. |
| `per_region_xyz.py` | Can per-region xyz weighting fix topology-inverted regions? | Convergent negative — local intervention doesn't reproduce global xyz effect. |

Run with:
```bash
PYTHONPATH=src python experiments/ablations/<name>.py
```

## archive/ — older detours

Stepping-stone experiments from earlier iterations. Kept for reproducibility but not part of the current narrative. See `docs/archive/iteration_log.md` for context.
