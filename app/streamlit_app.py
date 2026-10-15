"""Interactive front end for the prevalence-shift simulation.

Runs the functions in `sim.py` (imported, not copied) with the knobs exposed: how
separable the classifier is, how fast the category grows, and how many human labels
each period gets. The defaults are the published configuration, 1,000 replications
from the published seed, so the first screen is a live rerun of MEASUREMENTS.md, and
the app checks it against out/results.json with the tolerance CI uses.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from sim import Classifier, run_break, run_drift  # noqa: E402

REPO = "https://github.com/technobasant/prevalence-shift"
POST = "https://basantbhattarai.com.np/writing/count-prevalence-not-classifications/"
PUBLISHED = json.loads((ROOT / "out" / "results.json").read_text())
CFG = PUBLISHED["config"]
NAMES = {
    "CC": "Classify and count",
    "PCC": "Probabilistic count (PCC)",
    "ACC": "Adjusted count (ACC)",
    "SLD": "EM prior adjustment (SLD)",
    "GOLD": "Gold labels only",
    "PPI": "Prediction-powered (PPI)",
}
COLOURS = ["#8c8c8c", "#d62728", "#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd", "#17becf"]
BREAK_NAMES = {
    "CC": "Classify and count",
    "ACC_stale": "ACC, old error rates kept",
    "ACC_refreshed": "ACC, error rates re-measured",
    "PPI": "Prediction-powered (PPI)",
}


def close(a, b) -> bool:
    """check_reproduces.py's comparison: every number, floats to a relative 1e-6."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-12)
    return a == b


@st.cache_data(show_spinner=False)
def drift(seed, reps, n_items, n_gold, temperature, periods, p0, p1, d, pi_train):
    return run_drift(seed, reps, n_items, n_gold, CFG["n_val"], temperature, periods, p0, p1, d, pi_train)


@st.cache_data(show_spinner=False)
def upgrade(seed, reps, n_items, n_gold, periods, p_flat, d_old, d_new, pi_train, break_at):
    return run_break(seed, reps, n_items, n_gold, CFG["n_val"], periods, p_flat, d_old, d_new, pi_train, break_at)


def reproduced(live: dict, scenario: str) -> None:
    if close(live, PUBLISHED[scenario]):
        st.success(
            "These are the published settings, and every number matches `out/results.json`, the "
            "file CI regenerates and checks on every push."
        )


st.set_page_config(page_title="Counting labels flattens a trend", layout="wide")
st.title("Counting predicted labels flattens a trend")
st.markdown(
    "A category's true share of incoming items changes over time. A fixed classifier labels "
    "every item, and the share is estimated six ways. **Classify and count** is what a "
    "dashboard of predicted labels shows; the others correct for the classifier's error rates "
    "or lean on a small human-labelled sample. "
    f"Code: [prevalence-shift]({REPO}) · write-up: [The classifier was 95% accurate. The trend "
    f"it counted was a third too flat.]({POST})"
)

with st.sidebar:
    st.header("Classifier")
    d = st.slider(
        "Separation between the classes (d)", 1.0, 5.0, CFG["d"], 0.1,
        help="Distance between the two classes' score distributions. Larger is a better classifier.",
    )
    pi_train = st.slider("Share it was trained at", 0.02, 0.50, CFG["pi_train"], 0.01)
    temperature = st.select_slider(
        "Posterior over-confidence", options=[1.0, 1.5, 2.0, 3.0], value=1.0,
        help="1.0 is calibrated at the training share. Higher sharpens the posteriors without "
        "changing a single hard label. The published scenario B uses 2.0.",
    )
    clf = Classifier(d, pi_train, temperature)
    acc_train = pi_train * clf.tpr + (1 - pi_train) * (1 - clf.fpr)
    st.caption(
        f"Accuracy at the training share **{acc_train:.1%}** · "
        f"TPR {clf.tpr:.3f} · FPR {clf.fpr:.3f}"
    )

    st.header("World")
    p0, p1 = st.slider(
        "True share, first → last period", 0.01, 0.60, (CFG["p0"], CFG["p1"]), 0.01, format="%.2f"
    )
    periods = st.slider("Periods", 6, 24, CFG["periods"])
    n_items = st.select_slider("Items per period", options=[1000, 2000, 5000, 10000], value=CFG["n_items"])
    n_gold = st.select_slider(
        "Human-labelled items per period", options=[50, 100, 200, 500, 1000], value=CFG["n_gold"]
    )
    reps = st.select_slider(
        "Replications", options=[100, 200, 500, 1000], value=CFG["reps"],
        help="The published figures use 1,000. Fewer is faster and noisier.",
    )
    seed = int(st.number_input("Seed", value=CFG["seed"], step=1))
    st.caption(
        "As in sim.py, each scenario draws from its own stream: the calibrated trend from the "
        "seed, the over-confident one from seed + 1, the upgrade from seed + 2."
    )

tab_trend, tab_upgrade = st.tabs(
    ["A rising trend", "A model upgrade on a flat series"], on_change="rerun"
)

with tab_trend:
    if tab_trend.open:
        if p0 == p1:
            st.warning("Pick different first and last shares to see a trend.")
        else:
            with st.spinner(f"Simulating {reps:,} replications…"):
                res = drift(seed + (0 if temperature == 1.0 else 1), reps, n_items, n_gold, temperature,
                            periods, p0, p1, d, pi_train)
            est = res["estimators"]

            rows = []
            for k, name in NAMES.items():
                rows += [{"Period": t + 1, "Series": name, "Share": v}
                         for t, v in enumerate(est[k]["mean_by_period"])]
            # Truth last, so its dashed line is drawn over the estimators that sit on it.
            rows += [{"Period": t + 1, "Series": "Truth", "Share": v} for t, v in enumerate(res["truth"])]
            order = ["Truth", *NAMES.values()]
            is_truth = alt.datum.Series == "Truth"
            chart = (
                alt.Chart(pd.DataFrame(rows))
                .mark_line()
                .encode(
                    x=alt.X("Period:Q", scale=alt.Scale(domain=[1, periods], nice=False),
                            axis=alt.Axis(tickMinStep=1, format="d")),
                    y=alt.Y("Share:Q", scale=alt.Scale(zero=False), axis=alt.Axis(format="%"),
                            title="Mean estimated share"),
                    color=alt.Color("Series:N", scale=alt.Scale(domain=order, range=COLOURS),
                                    legend=alt.Legend(orient="bottom", title=None, columns=4)),
                    strokeDash=alt.condition(is_truth, alt.value([6, 4]), alt.value([1, 0])),
                    strokeWidth=alt.condition(is_truth, alt.value(3), alt.value(2)),
                    tooltip=["Series", "Period", alt.Tooltip("Share:Q", format=".2%")],
                )
                .properties(height=400)
            )
            st.altair_chart(chart, width="stretch")

            cc = est["CC"]
            st.markdown(
                f"Counting labels recovered **{cc['slope_ratio']:.0%}** of the true trend "
                f"(TPR − FPR = {clf.tpr - clf.fpr:.3f}), reading "
                f"{cc['mean_by_period'][0]:.1%} → {cc['mean_by_period'][-1]:.1%} for a true "
                f"{p0:.0%} → {p1:.0%}. Its naive 95% interval covered the truth in "
                f"**{cc['ci_coverage']:.1%}** of period estimates."
            )
            table = []
            for k, name in NAMES.items():
                e = est[k]
                table.append({
                    "Estimator": name,
                    "Share of true trend": f"{e['slope_ratio']:.3f}",
                    "Bias, first period": f"{e['bias_first']:+.4f}",
                    "Bias, last period": f"{e['bias_last']:+.4f}",
                    "RMSE": f"{e['rmse']:.4f}",
                    "95% interval covers truth": f"{e['ci_coverage']:.1%}" if "ci_coverage" in e else "—",
                    "Interval width": f"{e['ci_mean_width']:.4f}" if "ci_mean_width" in e else "—",
                })
            st.dataframe(pd.DataFrame(table), hide_index=True)
            st.caption(
                f"{reps:,} replications × {periods} periods × {n_items:,} items, "
                f"{n_gold} human labels per period. Coverage is the share of period estimates "
                "whose 95% interval contains the true share; PCC and SLD have no honest "
                "standard error, so no interval."
            )
            scenario = {1.0: "drift_calibrated", 2.0: "drift_overconfident"}.get(temperature)
            if scenario:
                reproduced(res, scenario)

with tab_upgrade:
    if tab_upgrade.open:
        st.markdown(
            "The true share stays flat. Partway through, the classifier is replaced by a better "
            "one. Every change an estimator reports is an artefact, and every *significant* "
            "change is a false trend."
        )
        c1, c2, c3 = st.columns(3)
        brk = CFG["break"]
        p_flat = c1.slider("Flat true share", 0.02, 0.50, brk["p_flat"], 0.01)
        d_new = c2.slider("New model's separation", 1.0, 5.0, brk["d_new"], 0.1,
                          help=f"The old model is the sidebar's d = {d}.")
        break_at = c3.slider("Upgrade after period", 2, periods - 2, min(brk["break_at"], periods - 2))
        with st.spinner(f"Simulating {reps:,} replications…"):
            br = upgrade(seed + 2, reps, n_items, n_gold, periods, p_flat, d, d_new, pi_train, break_at)
        st.caption(
            f"Old model TPR {br['old']['tpr']:.3f}, FPR {br['old']['fpr']:.3f} → new model TPR "
            f"{br['new']['tpr']:.3f}, FPR {br['new']['fpr']:.3f}."
        )
        table = [{
            "Estimator": BREAK_NAMES[k],
            "Mean before": f"{v['before_mean']:.4f}",
            "Mean after": f"{v['after_mean']:.4f}",
            "Apparent change": f"{v['apparent_change_rel']:+.1%}",
            "Runs calling it a significant trend": f"{v['false_trend_rate']:.1%}",
        } for k, v in br["estimators"].items()]
        st.dataframe(pd.DataFrame(table), hide_index=True)
        shared = br["estimators"]["ACC_refreshed"]["false_trend_rate_shared_validation"]
        st.markdown(
            "Re-measured ACC's false-trend rate is a flaw in the naive test, not in ACC: every "
            "period scored with one validation set shares that set's error, so the periods are "
            "not independent. Counting that variance once per half instead of averaging it down "
            f"gives **{shared:.1%}**, against a nominal 5%."
        )
        reproduced(br, "break")

st.divider()
st.caption(
    "Synthetic, idealised classifier: Gaussian scores, one threshold, pure label shift. What "
    "transfers is the direction and mechanism of each bias, not the magnitudes. "
    f"[Source]({REPO}/tree/main/app) · [Measurements]({REPO}/blob/main/MEASUREMENTS.md)"
)
