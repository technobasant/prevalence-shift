"""Prevalence under label shift: classify-and-count vs corrected estimators.

A seeded Monte Carlo. Each period draws N reports whose true category
prevalence p_t is known, scores them with a fixed binary classifier, and
estimates p_t six ways:

  CC     classify and count: mean of hard predictions
  PCC    probabilistic classify and count: mean of posteriors
  ACC    adjusted classify and count (Forman 2008), TPR/FPR from a validation set
  SLD    Saerens-Latinne-Decaestecker EM prior adjustment on posteriors
  GOLD   mean of a small random human-labelled sample from the period
  PPI    prediction-powered inference (Angelopoulos et al. 2023):
         mean(yhat over N) + mean(y - yhat over the gold sample)

Scores: x | y ~ Normal(mu_y, 1), mu_1 = +d/2, mu_0 = -d/2. The posterior is the
Bayes posterior at the TRAINING prior (perfectly calibrated there), optionally
made over-confident by a temperature. Hard prediction: posterior >= 0.5.

Every figure in MEASUREMENTS.md comes from `out/results.json`, written here.

    uv run --with numpy,scipy python sim.py --reps 1000
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import norm

OUT = Path(__file__).parent / "out"
Z = norm.ppf(0.975)


def logit(p):
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class Classifier:
    """Gaussian scores with separation d; posterior calibrated at pi_train."""

    def __init__(self, d: float, pi_train: float, temperature: float = 1.0):
        self.d, self.pi_train, self.temp = d, pi_train, temperature
        # LLR(x) = d * x for unit-variance Gaussians at +/- d/2
        self.threshold = -logit(pi_train) / d          # posterior = 0.5 here
        self.tpr = 1 - norm.cdf(self.threshold - d / 2)
        self.fpr = 1 - norm.cdf(self.threshold + d / 2)

    def scores(self, rng, y):
        return rng.normal(np.where(y == 1, self.d / 2, -self.d / 2), 1.0)

    def posterior(self, x):
        # temperature > 1 scales logits up: over-confident, same hard decisions
        return sigmoid(self.temp * (self.d * x + logit(self.pi_train)))

    def predict(self, x):
        return (x >= self.threshold).astype(np.int8)


def validation_rates(rng, clf, n_val, pi_val):
    """Estimate TPR/FPR on a labelled validation set at the training prior."""
    y = (rng.random(n_val) < pi_val).astype(np.int8)
    yhat = clf.predict(clf.scores(rng, y))
    pos, neg = y == 1, y == 0
    tpr = yhat[pos].mean()
    fpr = yhat[neg].mean()
    return tpr, fpr, int(pos.sum()), int(neg.sum())


def sld(post, pi_train, tol=1e-6, max_iter=1000):
    """EM prior adjustment (Saerens et al. 2002), binary case."""
    pi = pi_train
    for _ in range(max_iter):
        w1, w0 = pi / pi_train, (1 - pi) / (1 - pi_train)
        adj = w1 * post / (w1 * post + w0 * (1 - post))
        new = adj.mean()
        if abs(new - pi) < tol:
            return new
        pi = new
    return pi


def acc(q, n_items, rates):
    """Adjusted classify-and-count with a delta-method standard error that
    includes the validation set's uncertainty in TPR and FPR."""
    tpr_hat, fpr_hat, n_pos, n_neg = rates
    denom = tpr_hat - fpr_hat
    p_ = (q - fpr_hat) / denom
    var_t = tpr_hat * (1 - tpr_hat) / n_pos
    var_f = fpr_hat * (1 - fpr_hat) / n_neg
    var_q = q * (1 - q) / n_items
    return float(np.clip(p_, 0, 1)), float(np.sqrt((var_q + p_**2 * var_t + (1 - p_) ** 2 * var_f) / denom**2))


def acc_shared_var(q, rates):
    """The part of ACC's variance that comes from the validation set. Every
    period scored with the same TPR/FPR estimate shares it, so it does not
    average down across periods."""
    tpr_hat, fpr_hat, n_pos, n_neg = rates
    denom = tpr_hat - fpr_hat
    p_ = (q - fpr_hat) / denom
    return (p_**2 * tpr_hat * (1 - tpr_hat) / n_pos + (1 - p_) ** 2 * fpr_hat * (1 - fpr_hat) / n_neg) / denom**2


def estimate_period(rng, clf, p_true, n_items, n_gold, rates, alt_rates=None):
    """All estimators, with standard errors where an honest one exists."""
    y = (rng.random(n_items) < p_true).astype(np.int8)
    x = clf.scores(rng, y)
    yhat = clf.predict(x)
    post = clf.posterior(x)

    out = {}
    q = yhat.mean()
    out["CC"] = (q, np.sqrt(q * (1 - q) / n_items))          # the naive error bar
    out["PCC"] = (post.mean(), np.nan)

    out["ACC"] = acc(q, n_items, rates)
    if alt_rates is not None:
        out["ACC_alt"] = acc(q, n_items, alt_rates)
        out["ACC_alt_shared_var"] = acc_shared_var(q, alt_rates)
        out["q_var"] = q * (1 - q) / n_items

    out["SLD"] = (sld(post, clf.pi_train), np.nan)

    gold = rng.choice(n_items, size=n_gold, replace=False)   # uniform random sample
    g = y[gold].mean()
    out["GOLD"] = (g, np.sqrt(g * (1 - g) / n_gold))

    rect = y[gold] - yhat[gold]                               # rectifier
    ppi = q + rect.mean()
    out["PPI"] = (ppi, np.sqrt(yhat.var(ddof=1) / n_items + rect.var(ddof=1) / n_gold))
    return out


def run_drift(seed, reps, n_items, n_gold, n_val, temperature, periods, p0, p1, d, pi_train):
    rng = np.random.default_rng(seed)
    clf = Classifier(d, pi_train, temperature)
    truth = np.linspace(p0, p1, periods)
    names = ["CC", "PCC", "ACC", "SLD", "GOLD", "PPI"]
    est = {k: np.empty((reps, periods)) for k in names}
    se = {k: np.empty((reps, periods)) for k in names}
    for r in range(reps):
        rates = validation_rates(rng, clf, n_val, pi_train)   # one validation set per replication
        for t, p in enumerate(truth):
            for k, (v, s) in estimate_period(rng, clf, p, n_items, n_gold, rates).items():
                est[k][r, t], se[k][r, t] = v, s

    true_slope = (p1 - p0) / (periods - 1)
    tt = np.arange(periods)
    res = {"classifier": {"d": d, "pi_train": pi_train, "temperature": temperature,
                          "threshold": clf.threshold, "tpr": clf.tpr, "fpr": clf.fpr,
                          "accuracy_at_train_prior": pi_train * clf.tpr + (1 - pi_train) * (1 - clf.fpr)},
           "truth": truth.tolist(), "true_slope": true_slope, "estimators": {}}
    for k in names:
        e = est[k]
        err = e - truth
        slopes = np.polyfit(tt, e.T, 1)[0]                    # per-replication OLS slope
        row = {
            "mean_by_period": e.mean(0).tolist(),
            "bias_first": float(err[:, 0].mean()),
            "bias_last": float(err[:, -1].mean()),
            "mean_abs_bias": float(np.abs(err.mean(0)).mean()),
            "rmse": float(np.sqrt((err**2).mean())),
            "slope_mean": float(slopes.mean()),
            "slope_ratio": float(slopes.mean() / true_slope),
            "slope_sd": float(slopes.std(ddof=1)),
        }
        if not np.isnan(se[k]).all():
            lo, hi = e - Z * se[k], e + Z * se[k]
            row["ci_coverage"] = float(((lo <= truth) & (truth <= hi)).mean())
            row["ci_mean_width"] = float((hi - lo).mean())
        res["estimators"][k] = row
    return res


def run_break(seed, reps, n_items, n_gold, n_val, periods, p_flat, d_old, d_new, pi_train, break_at):
    """Flat true prevalence; the classifier is upgraded at `break_at`."""
    rng = np.random.default_rng(seed)
    old, new = Classifier(d_old, pi_train), Classifier(d_new, pi_train)
    names = ["CC", "ACC_stale", "ACC_refreshed", "PPI"]
    est = {k: np.empty((reps, periods)) for k in names}
    se = {k: np.empty((reps, periods)) for k in names}
    shared = np.empty((reps, periods))      # validation-set variance of ACC_refreshed
    qvar = np.empty((reps, periods))        # sampling variance of CC
    denoms = np.empty((reps, periods))
    for r in range(reps):
        rates_old = validation_rates(rng, old, n_val, pi_train)
        rates_new = validation_rates(rng, new, n_val, pi_train)   # a bridging validation run
        for t in range(periods):
            clf = old if t < break_at else new
            current = rates_old if t < break_at else rates_new
            a = estimate_period(rng, clf, p_flat, n_items, n_gold, rates_old, alt_rates=current)
            est["CC"][r, t], se["CC"][r, t] = a["CC"]
            est["PPI"][r, t], se["PPI"][r, t] = a["PPI"]
            est["ACC_stale"][r, t], se["ACC_stale"][r, t] = a["ACC"]          # old matrix forever
            est["ACC_refreshed"][r, t], se["ACC_refreshed"][r, t] = a["ACC_alt"]  # matrix of the model in use
            shared[r, t], qvar[r, t] = a["ACC_alt_shared_var"], a["q_var"]
            denoms[r, t] = current[0] - current[1]

    res = {"old": {"tpr": old.tpr, "fpr": old.fpr}, "new": {"tpr": new.tpr, "fpr": new.fpr},
           "p_flat": p_flat, "break_at": break_at, "estimators": {}}
    for k in names:
        e, s = est[k], se[k]
        before, after = e[:, :break_at], e[:, break_at:]
        delta = after.mean(1) - before.mean(1)
        # SE of a difference of period means, treating periods as independent
        se_d = np.sqrt((s[:, break_at:] ** 2).sum(1) / (periods - break_at) ** 2
                       + (s[:, :break_at] ** 2).sum(1) / break_at ** 2)
        zval = delta / se_d
        extra = {}
        if k == "ACC_refreshed":
            # Correct test: sampling variance averages down across periods, the
            # validation-set variance of each half does not (one estimate per half).
            nb, na = break_at, periods - break_at
            v_after = (qvar[:, break_at:] / denoms[:, break_at:] ** 2).sum(1) / na**2 + shared[:, break_at:].mean(1)
            v_before = (qvar[:, :break_at] / denoms[:, :break_at] ** 2).sum(1) / nb**2 + shared[:, :break_at].mean(1)
            z2 = delta / np.sqrt(v_after + v_before)
            extra["false_trend_rate_shared_validation"] = float((np.abs(z2) > Z).mean())
        res["estimators"][k] = {
            **extra,
            "before_mean": float(before.mean()),
            "after_mean": float(after.mean()),
            "apparent_change_abs": float(delta.mean()),
            "apparent_change_rel": float(delta.mean() / before.mean()),
            "false_trend_rate": float((np.abs(zval) > Z).mean()),
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261003)
    args = ap.parse_args()

    common = dict(reps=args.reps, n_items=5000, n_gold=200, n_val=2000)
    t0 = time.process_time()
    results = {
        "config": {**common, "seed": args.seed, "periods": 12, "p0": 0.10, "p1": 0.25,
                   "d": 2.5, "pi_train": 0.10, "break": {"p_flat": 0.15, "d_old": 2.5,
                                                          "d_new": 3.5, "break_at": 6}},
        "env": {"python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "machine": platform.machine()},
        "drift_calibrated": run_drift(args.seed, temperature=1.0, periods=12, p0=0.10, p1=0.25,
                                      d=2.5, pi_train=0.10, **common),
        "drift_overconfident": run_drift(args.seed + 1, temperature=2.0, periods=12, p0=0.10,
                                         p1=0.25, d=2.5, pi_train=0.10, **common),
        "break": run_break(args.seed + 2, periods=12, p_flat=0.15, d_old=2.5, d_new=3.5,
                           pi_train=0.10, break_at=6, **common),
    }
    results["cpu_seconds"] = time.process_time() - t0
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k in ("config", "env", "cpu_seconds")}, indent=2))


if __name__ == "__main__":
    main()
