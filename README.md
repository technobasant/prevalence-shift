# prevalence-shift

[![ci](https://github.com/technobasant/prevalence-shift/actions/workflows/ci.yml/badge.svg)](https://github.com/technobasant/prevalence-shift/actions/workflows/ci.yml)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://prevalence-shift.streamlit.app)

When a category's share of incoming reports changes over time, which estimator reports the change honestly? This is a Monte Carlo simulation with a fixed seed that answers that for one idealised classifier, and the answer is not the one a dashboard built on predicted labels gives you.

A classifier that is 95% accurate at its training prior **counted a rise from 10% to 25% as 8% to 17%**. Its trend slope came out at 0.628 of the true one — exactly TPR − FPR — and its naive 95% intervals covered the truth in **0.0%** of 12,000 period estimates. Counting labels does not measure prevalence; it measures prevalence passed through the classifier's error rates. Then a model upgrade on a flat 15% series showed up as a **+24% "trend" that never happened**.

Write-up: [The classifier was 95% accurate. The trend it counted was a third too flat.](https://basantbhattarai.com.np/writing/count-prevalence-not-classifications/)

**Try it in the browser: [prevalence-shift.streamlit.app](https://prevalence-shift.streamlit.app).** It runs `sim.py` live: move the classifier and the trend, try a weaker model or a rarer category, or [open the model upgrade](https://prevalence-shift.streamlit.app/?view=upgrade). At the published settings it reproduces every figure below and says so. After a quiet spell it takes about half a minute to wake.

## What it compares

Every period draws 5,000 reports whose true prevalence is known, scores them with a fixed binary classifier, and estimates the prevalence six ways:

| Estimator | What it does |
|---|---|
| CC | classify and count: the mean of hard predictions |
| PCC | probabilistic classify and count: the mean of posteriors |
| ACC | adjusted classify and count — `(CC − FPR) / (TPR − FPR)`, error rates from a validation set |
| SLD | Saerens–Latinne–Decaestecker EM re-estimation of the prior from posteriors |
| GOLD | the mean of a small random human-labelled sample from the period (200 labels) |
| PPI | prediction-powered inference: the counted share plus the mean (label − prediction) on the gold sample |

Three scenarios, 1,000 replications each, 12 periods, seed `20261003`:

- **A.** True prevalence rises linearly from 10% to 25%; posteriors are calibrated at the training prior (10%).
- **B.** The same drift, with over-confident posteriors (temperature 2) and identical hard labels.
- **C.** A flat 15%, with the classifier upgraded after period 6.

## Results

Every figure below is from [MEASUREMENTS.md](MEASUREMENTS.md), which `report.py` renders from `out/results.json`. None is typed by hand.

**Scenario A — 10% → 25%, calibrated posteriors.** Classifier TPR 0.645, FPR 0.0166, 95.0% accurate at its training prior.

| Estimator | Bias at 10% | Bias at 25% | RMSE | Trend slope ÷ true | 95% CI coverage |
|---|---:|---:|---:|---:|---:|
| CC | −0.0205 | −0.0764 | 0.0517 | 0.628 | **0.0%** |
| PCC | +0.0000 | −0.0635 | 0.0376 | 0.577 | — |
| ACC | +0.0004 | +0.0008 | 0.0130 | 1.003 | 95.0% |
| SLD | +0.0000 | −0.0002 | 0.0067 | 1.000 | — |
| GOLD | +0.0007 | +0.0004 | 0.0268 | 1.002 | 94.3% |
| PPI | −0.0001 | +0.0007 | 0.0196 | 1.003 | 94.7% |

PPI's interval is 0.739× the width of the gold-only interval from the same 200 labels — the equivalent of about 367 gold labels.

**Scenario B — same drift, over-confident posteriors.** SLD's slope drops to **0.775** and its bias at 25% to −0.0532: it is only as good as the calibration it assumes. ACC (slope 1.003, coverage 95.3%) and PPI (0.993, 95.3%) are unaffected, because neither reads the posterior.

**Scenario C — flat 15%, classifier upgraded after period 6.** The true change is zero.

| Estimator | Apparent change | Relative | False-trend rate (α = 0.05) |
|---|---:|---:|---:|
| CC | +0.0269 | 24.2% | 100.0% |
| ACC, stale error rates | +0.0431 | 28.6% | 100.0% |
| ACC, refreshed error rates | −0.0009 | −0.6% | 30.4% |
| PPI | +0.0000 | 0.0% | 5.8% |

The 30.4% for refreshed ACC is a flaw in my first test, not in ACC: every period scored with one validation set shares that set's TPR/FPR error, so the periods are not independent. Counting that variance once per half instead of averaging it down gives **5.0%**.

## Reproduce

```bash
uv run python sim.py --reps 1000            # writes out/results.json (~7–15 s CPU)
uv run python report.py                     # renders MEASUREMENTS.md from it
uv run python scripts/check_reproduces.py   # every figure vs the committed results
```

`uv run --with numpy,scipy python sim.py --reps 1000`, the form MEASUREMENTS.md prints, also works. On the machine that produced the figures, a rerun under Python 3.12 instead of 3.13 reproduced `results.json` bit for bit. CI reruns the full 1,000 replications on every push and fails if any committed figure moves.

The browser app in [`app/`](app) runs locally with `uv run --with-requirements app/requirements.txt streamlit run app/streamlit_app.py`; CI checks that it still reproduces scenarios A, B and C.

## Method notes

- Scores: `x | y ~ Normal(±d/2, 1)` with separation d = 2.5 (3.5 for the upgraded model). The posterior is the Bayes posterior at the training prior, so the classifier is perfectly calibrated where it was trained; the hard label is posterior ≥ 0.5.
- ACC's TPR/FPR come from a 2,000-item validation set at the training prior, drawn once per replication per model.
- Interval coverage is counted over every period of every replication (12,000 estimates per scenario). CC's interval is the naive binomial one a dashboard would draw.
- The false-trend test is a two-sided z-test on (mean of periods 7–12 − mean of periods 1–6), using each estimator's own standard errors.

## Limits

- **Generated data, idealised classifier.** Gaussian scores with equal variance and a single threshold. What transfers is the direction and mechanism of each bias — CC's slope is TPR − FPR, SLD inherits calibration error, a model change is a measurement break — not the specific magnitudes.
- **Pure label shift.** The share changes; what a report in each category looks like does not. ACC and SLD rely on that. PPI needs no assumption about the classifier, only a uniform random gold sample drawn from the same period — but evolving categories are not simulated here.
- **Binary, one category at a time.** Real taxonomies are multi-class, with mixed sources, campaign duplicates and categories too rare for a per-period estimate.
- **One machine** for the timings (Apple M5). The figures themselves are deterministic for the seed.

## The literature this relies on

- Forman, [adjusted classify-and-count](https://link.springer.com/article/10.1007/s10618-008-0097-y) (Data Mining and Knowledge Discovery, 2008)
- Saerens, Latinne and Decaestecker, [EM re-estimation of class priors](https://doi.org/10.1162/089976602753284446) (Neural Computation, 2002), and [Esuli, Molinari and Sebastiani's reassessment of SLD](https://dl.acm.org/doi/10.1145/3433164)
- Lipton, Wang and Smola, [detecting and correcting for label shift with black-box predictors](https://arxiv.org/abs/1802.03916) (2018)
- Angelopoulos et al., [prediction-powered inference](https://arxiv.org/abs/2301.09633) (2023), and [Egami and colleagues](https://arxiv.org/abs/2306.04746) on design-based supervised learning
- [QuaPy](https://github.com/HLT-ISTI/QuaPy), a library of quantification methods
- The [Federal Reserve's ScamClassifier](https://fedpaymentsimprovement.org/strategic-initiatives/payments-security/scams/scamclassifier-model/), a public scam taxonomy

The question came from trend measurement in [the consulting engagement](https://basantbhattarai.com.np/work/consulting-short-text-and-alerting/) this lab rebuilds the idea of. No data or code from it is used here.

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
