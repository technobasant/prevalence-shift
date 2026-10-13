# Measurements — prevalence under label shift

Every figure here is rendered by `report.py` from `out/results.json`, which `sim.py` writes.
Re-run: `uv run --with numpy,scipy python sim.py --reps 1000 && python3 report.py`.

## Apparatus

- Seed `20261003`; 1,000 replications per scenario; 12 periods.
- Per period: 5,000 reports scored; a uniform random gold sample of 200 human labels.
- Validation set for ACC: 2,000 labelled items at the training prior, one per replication (per model).
- Scores: x | y ~ Normal(±d/2, 1). Posterior = Bayes posterior at the training prior, so the classifier is perfectly calibrated where it was trained. Hard label: posterior ≥ 0.5.
- Python 3.13.15, numpy 2.5.3, scipy 1.18.1, arm64 (Apple M5). Whole run: 7.2 s CPU.

## Scenario A — prevalence rises 10% → 25%, calibrated posteriors

Classifier: d = 2.5, trained at prior 0.10; TPR 0.645, FPR 0.0166, accuracy 95.0% at the training prior. TPR − FPR = 0.628. True slope 0.01364 per period. Counted share = FPR + (TPR − FPR)·p crosses the diagonal at FPR / (1 − TPR + FPR) = 0.0447. Coverage is over 12,000 period estimates.

| Estimator | Bias at 10% | Bias at 25% | RMSE | Trend slope ÷ true | Slope SD | 95% CI coverage | Mean CI width |
|---|---:|---:|---:|---:|---:|---:|---:|
| CC | -0.0205 | -0.0764 | 0.0517 | 0.628 | 0.00040 | 0.0% | 0.0183 |
| PCC | +0.0000 | -0.0635 | 0.0376 | 0.577 | 0.00033 | — | — |
| ACC | +0.0004 | +0.0008 | 0.0130 | 1.003 | 0.00099 | 95.0% | 0.0503 |
| SLD | +0.0000 | -0.0002 | 0.0067 | 1.000 | 0.00057 | — | — |
| GOLD | +0.0007 | +0.0004 | 0.0268 | 1.002 | 0.00229 | 94.3% | 0.1033 |
| PPI | -0.0001 | +0.0007 | 0.0196 | 1.003 | 0.00166 | 94.7% | 0.0763 |

Mean estimate by period (first, middle, last):

| Estimator | Period 1 | Period 6 | Period 12 |
|---|---:|---:|---:|
| truth | 0.1000 | 0.1682 | 0.2500 |
| CC | 0.0795 | 0.1221 | 0.1736 |
| PCC | 0.1000 | 0.1393 | 0.1865 |
| ACC | 0.1004 | 0.1686 | 0.2508 |
| SLD | 0.1000 | 0.1680 | 0.2498 |
| GOLD | 0.1007 | 0.1677 | 0.2504 |
| PPI | 0.0999 | 0.1683 | 0.2507 |

PPI CI width ÷ gold-only CI width: 0.739; equivalent gold-sample multiplier (width ratio squared): 1.83×, i.e. about 367 gold labels' worth from 200.

## Scenario B — same drift, over-confident posteriors (temperature 2)

Classifier: d = 2.5, trained at prior 0.10; TPR 0.645, FPR 0.0166, accuracy 95.0% at the training prior. TPR − FPR = 0.628. True slope 0.01364 per period. Counted share = FPR + (TPR − FPR)·p crosses the diagonal at FPR / (1 − TPR + FPR) = 0.0447. Coverage is over 12,000 period estimates.

| Estimator | Bias at 10% | Bias at 25% | RMSE | Trend slope ÷ true | Slope SD | 95% CI coverage | Mean CI width |
|---|---:|---:|---:|---:|---:|---:|---:|
| CC | -0.0206 | -0.0763 | 0.0518 | 0.627 | 0.00039 | 0.0% | 0.0183 |
| PCC | -0.0159 | -0.0737 | 0.0486 | 0.613 | 0.00035 | — | — |
| ACC | +0.0008 | +0.0015 | 0.0131 | 1.003 | 0.00095 | 95.3% | 0.0504 |
| SLD | -0.0197 | -0.0532 | 0.0387 | 0.775 | 0.00044 | — | — |
| GOLD | +0.0003 | -0.0003 | 0.0266 | 0.999 | 0.00215 | 94.8% | 0.1034 |
| PPI | +0.0010 | -0.0003 | 0.0193 | 0.993 | 0.00156 | 95.3% | 0.0764 |

Mean estimate by period (first, middle, last):

| Estimator | Period 1 | Period 6 | Period 12 |
|---|---:|---:|---:|
| truth | 0.1000 | 0.1682 | 0.2500 |
| CC | 0.0794 | 0.1223 | 0.1737 |
| PCC | 0.0841 | 0.1260 | 0.1763 |
| ACC | 0.1008 | 0.1693 | 0.2515 |
| SLD | 0.0803 | 0.1326 | 0.1968 |
| GOLD | 0.1003 | 0.1671 | 0.2497 |
| PPI | 0.1010 | 0.1683 | 0.2497 |

PPI CI width ÷ gold-only CI width: 0.739; equivalent gold-sample multiplier (width ratio squared): 1.83×, i.e. about 367 gold labels' worth from 200.

## Scenario C — flat prevalence 15%, classifier upgraded after period 6

Old model TPR 0.645 / FPR 0.0166; new model TPR 0.869 / FPR 0.0087. True change: 0.

| Estimator | Mean, periods 1–6 | Mean, periods 7–12 | Apparent change | Relative | False-trend rate (α = 0.05) |
|---|---:|---:|---:|---:|---:|
| CC | 0.1109 | 0.1377 | +0.0269 | 24.2% | 100.0% |
| ACC_stale | 0.1510 | 0.1940 | +0.0431 | 28.6% | 100.0% |
| ACC_refreshed | 0.1510 | 0.1500 | -0.0009 | -0.6% | 30.4% |
| PPI | 0.1500 | 0.1501 | +0.0000 | 0.0% | 5.8% |

False-trend rate: share of replications where a two-sided z-test on (mean of periods 7–12 − mean of periods 1–6) rejects 'no change', using each estimator's own standard errors and treating periods as independent.
For ACC_refreshed that independence is wrong — every period scored with one validation set shares its TPR/FPR error. With that variance counted once per half instead of averaged down, the rate is 5.0%.
