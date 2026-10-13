"""Render MEASUREMENTS.md from out/results.json. No number is typed by hand."""
import json
from pathlib import Path

ROOT = Path(__file__).parent
r = json.loads((ROOT / "out" / "results.json").read_text())
cfg, env = r["config"], r["env"]


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def f(x, d=4):
    return f"{x:+.{d}f}"


lines = []
w = lines.append
w("# Measurements — prevalence under label shift\n")
w("Every figure here is rendered by `report.py` from `out/results.json`, which `sim.py` writes.")
w("Re-run: `uv run --with numpy,scipy python sim.py --reps 1000 && python3 report.py`.\n")
w("## Apparatus\n")
w(f"- Seed `{cfg['seed']}`; {cfg['reps']:,} replications per scenario; 12 periods.")
w(f"- Per period: {cfg['n_items']:,} reports scored; a uniform random gold sample of {cfg['n_gold']} human labels.")
w(f"- Validation set for ACC: {cfg['n_val']:,} labelled items at the training prior, one per replication (per model).")
w("- Scores: x | y ~ Normal(±d/2, 1). Posterior = Bayes posterior at the training prior, "
  "so the classifier is perfectly calibrated where it was trained. Hard label: posterior ≥ 0.5.")
w(f"- Python {env['python']}, numpy {env['numpy']}, scipy {env['scipy']}, {env['machine']} (Apple M5). "
  f"Whole run: {r['cpu_seconds']:.1f} s CPU.\n")

for key, title in (("drift_calibrated", "Scenario A — prevalence rises 10% → 25%, calibrated posteriors"),
                   ("drift_overconfident", "Scenario B — same drift, over-confident posteriors (temperature 2)")):
    d = r[key]
    c = d["classifier"]
    w(f"## {title}\n")
    w(f"Classifier: d = {c['d']}, trained at prior {c['pi_train']:.2f}; TPR {c['tpr']:.3f}, FPR {c['fpr']:.4f}, "
      f"accuracy {pct(c['accuracy_at_train_prior'])} at the training prior. "
      f"TPR − FPR = {c['tpr'] - c['fpr']:.3f}. True slope {d['true_slope']:.5f} per period. "
      f"Counted share = FPR + (TPR − FPR)·p crosses the diagonal at FPR / (1 − TPR + FPR) = "
      f"{c['fpr'] / (1 - c['tpr'] + c['fpr']):.4f}. Coverage is over {cfg['reps'] * cfg['periods']:,} period estimates.\n")
    w("| Estimator | Bias at 10% | Bias at 25% | RMSE | Trend slope ÷ true | Slope SD | 95% CI coverage | Mean CI width |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    for k, v in d["estimators"].items():
        cov = pct(v["ci_coverage"]) if "ci_coverage" in v else "—"
        wid = f"{v['ci_mean_width']:.4f}" if "ci_mean_width" in v else "—"
        w(f"| {k} | {f(v['bias_first'])} | {f(v['bias_last'])} | {v['rmse']:.4f} | {v['slope_ratio']:.3f} | "
          f"{v['slope_sd']:.5f} | {cov} | {wid} |")
    w("")
    w("Mean estimate by period (first, middle, last):\n")
    w("| Estimator | Period 1 | Period 6 | Period 12 |")
    w("|---|---:|---:|---:|")
    w(f"| truth | {d['truth'][0]:.4f} | {d['truth'][5]:.4f} | {d['truth'][11]:.4f} |")
    for k, v in d["estimators"].items():
        m = v["mean_by_period"]
        w(f"| {k} | {m[0]:.4f} | {m[5]:.4f} | {m[11]:.4f} |")
    w("")
    if "PPI" in d["estimators"] and "GOLD" in d["estimators"]:
        g, p = d["estimators"]["GOLD"], d["estimators"]["PPI"]
        w(f"PPI CI width ÷ gold-only CI width: {p['ci_mean_width'] / g['ci_mean_width']:.3f}; "
          f"equivalent gold-sample multiplier (width ratio squared): "
          f"{(g['ci_mean_width'] / p['ci_mean_width']) ** 2:.2f}×, i.e. about "
          f"{round(cfg['n_gold'] * (g['ci_mean_width'] / p['ci_mean_width']) ** 2)} gold labels' worth from {cfg['n_gold']}.\n")

b = r["break"]
w("## Scenario C — flat prevalence 15%, classifier upgraded after period 6\n")
w(f"Old model TPR {b['old']['tpr']:.3f} / FPR {b['old']['fpr']:.4f}; "
  f"new model TPR {b['new']['tpr']:.3f} / FPR {b['new']['fpr']:.4f}. True change: 0.\n")
w("| Estimator | Mean, periods 1–6 | Mean, periods 7–12 | Apparent change | Relative | False-trend rate (α = 0.05) |")
w("|---|---:|---:|---:|---:|---:|")
for k, v in b["estimators"].items():
    w(f"| {k} | {v['before_mean']:.4f} | {v['after_mean']:.4f} | {f(v['apparent_change_abs'])} | "
      f"{pct(v['apparent_change_rel'])} | {pct(v['false_trend_rate'])} |")
acc = b["estimators"]["ACC_refreshed"]
w("")
w("False-trend rate: share of replications where a two-sided z-test on (mean of periods 7–12 − mean of "
  "periods 1–6) rejects 'no change', using each estimator's own standard errors and treating periods as "
  "independent.")
w(f"For ACC_refreshed that independence is wrong — every period scored with one validation set shares its "
  f"TPR/FPR error. With that variance counted once per half instead of averaged down, the rate is "
  f"{pct(acc['false_trend_rate_shared_validation'])}.\n")

(ROOT / "MEASUREMENTS.md").write_text("\n".join(lines))
print("\n".join(lines))
