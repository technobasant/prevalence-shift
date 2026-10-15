"""Interactive front end for the prevalence-shift simulation.

Runs the functions in `sim.py` (imported, not copied). Pick a scenario or move the
classifier and the trend, and watch how far a count of predicted labels falls short of
the truth and which estimators recover it. The defaults are the published configuration,
1,000 replications from the published seed, so the first screen is a live rerun of
MEASUREMENTS.md, checked against out/results.json at the tolerance CI uses.

`?view=upgrade` opens the model-upgrade view directly.
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

# Community Cloud pulls each push into the running process and re-reads this script on
# every run, but the imported sim module and st.cache_data results outlive the push: the
# app would keep serving the previous simulation while saying it runs this repository's
# code. Fingerprint what it runs, and start afresh when that changes.
_stamp = tuple(p.stat().st_mtime_ns for p in (ROOT / "sim.py", ROOT / "out" / "results.json"))
if getattr(sys.modules.get("sim"), "app_stamp", None) != _stamp:
    sys.modules.pop("sim", None)
    st.cache_data.clear()

import sim  # noqa: E402
from sim import Classifier, run_break, run_drift  # noqa: E402

sim.app_stamp = _stamp

REPO = "https://github.com/technobasant/prevalence-shift"
POST = "https://basantbhattarai.com.np/writing/count-prevalence-not-classifications/"
PUBLISHED = json.loads((ROOT / "out" / "results.json").read_text())
CFG = PUBLISHED["config"]
BRK = CFG["break"]

NAMES = {
    "CC": "Classify and count",
    "PCC": "Probabilistic count",
    "ACC": "Adjusted count (ACC)",
    "SLD": "EM prior adjustment (SLD)",
    "GOLD": "Gold labels only",
    "PPI": "Prediction-powered (PPI)",
}
COLOURS = {
    "Truth": "#5f6670",
    "Classify and count": "#d1495b",
    "Probabilistic count": "#e9a23b",
    "Adjusted count (ACC)": "#3a7bd5",
    "EM prior adjustment (SLD)": "#2a9d8f",
    "Gold labels only": "#8e6cc9",
    "Prediction-powered (PPI)": "#17a2b8",
}
SHORT = {"CC": "CC", "PCC": "PCC", "ACC": "ACC", "SLD": "SLD", "GOLD": "Gold", "PPI": "PPI"}
BREAK_NAMES = {
    "CC": "Classify and count",
    "ACC_stale": "ACC, old error rates kept",
    "ACC_refreshed": "ACC, error rates re-measured",
    "PPI": "Prediction-powered (PPI)",
}

TREND, UPGRADE = "A rising trend", "A model upgrade"
PUBLISHED_TREND = {"d": CFG["d"], "pi_train": CFG["pi_train"], "temperature": 1.0,
                   "shares": (CFG["p0"], CFG["p1"]), "periods": CFG["periods"]}
PRESETS = {
    "The published trend": PUBLISHED_TREND,
    "An over-confident model": {**PUBLISHED_TREND, "temperature": 2.0},
    "A weaker classifier": {**PUBLISHED_TREND, "d": 1.5},
    "A rare category": {**PUBLISHED_TREND, "pi_train": 0.02, "shares": (0.01, 0.04)},
}
PRESET_NOTES = {
    "The published trend": "Scenario A: a 95%-accurate classifier, trained at a 10% share, "
    "watching a category grow from 10% to 25%.",
    "An over-confident model": "Scenario B: the same hard labels, with posteriors sharpened by "
    "a temperature of 2. Watch SLD, which trusts the posteriors.",
    "A weaker classifier": "Less separation between the classes. Counting multiplies every "
    "change by TPR − FPR, so the trend nearly disappears.",
    "A rare category": "A category that goes from 1% to 4%. The classifier mostly says no, "
    "and the count barely moves.",
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
            "These are the published settings, and every number matches `out/results.json`, "
            "the file CI regenerates and checks on every push.",
            icon=":material/verified:",
        )


def html_legend(names: list[str]) -> str:
    """A legend that wraps on a phone instead of being clipped inside the chart."""
    items = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:.4rem;margin:0 1.1rem .3rem 0">'
        f'<span style="width:1rem;height:.22rem;border-radius:1px;background:{COLOURS[n]}">'
        f"</span>{n}{' (dashed)' if n == 'Truth' else ''}</span>"
        for n in names
    )
    return f'<div style="font-size:.85rem;display:flex;flex-wrap:wrap">{items}</div>'


def apply_preset() -> None:
    name = st.session_state.get("preset")
    if name:
        st.session_state.update(PRESETS[name])


def clear_preset() -> None:
    st.session_state["preset"] = None


st.set_page_config(
    page_title="Counting labels flattens a trend · prevalence-shift",
    page_icon=":material/trending_down:",
    layout="wide",
)
st.title("Counting predicted labels flattens a trend")
st.markdown(
    "A category's true share of incoming items changes over time, and a fixed classifier "
    "labels every item. **Classify and count**, what a dashboard of predicted labels shows, "
    "gets the trend wrong in a predictable way. Five other estimators try to fix it."
)
st.caption(
    f"[prevalence-shift on GitHub]({REPO}) · [The write-up: the classifier was 95% "
    f"accurate]({POST}) · runs sim.py live, seeded"
)

view = st.query_params.get("view")
tab_trend, tab_upgrade = st.tabs(
    [TREND, UPGRADE], default=UPGRADE if view == "upgrade" else TREND, on_change="rerun"
)

with tab_trend:
    if tab_trend.open:
        for key, value in PUBLISHED_TREND.items():
            st.session_state.setdefault(key, value)
        st.session_state.setdefault("preset", "The published trend")
        st.pills("Start from a scenario", list(PRESETS), key="preset", on_change=apply_preset)
        if st.session_state.get("preset"):
            st.caption(PRESET_NOTES[st.session_state["preset"]])

        controls, results = st.columns([1, 2], gap="large")
        with controls, st.container(border=True):
            st.markdown("**The classifier**")
            d = st.slider("Separation between the classes", 1.0, 5.0, step=0.1, key="d",
                          on_change=clear_preset,
                          help="Distance between the two classes' score distributions, in "
                          "standard deviations. Larger is a better classifier.")
            pi_train = st.slider("Share it was trained at", 0.02, 0.50, step=0.01, key="pi_train",
                                 on_change=clear_preset)
            temperature = st.select_slider(
                "Posterior over-confidence", options=[1.0, 1.5, 2.0, 3.0], key="temperature",
                on_change=clear_preset,
                help="1.0 is calibrated at the training share. Higher sharpens the posteriors "
                "without changing a single hard label.",
            )
            clf = Classifier(d, pi_train, temperature)
            acc_train = pi_train * clf.tpr + (1 - pi_train) * (1 - clf.fpr)
            st.caption(f"Accuracy at the training share **{acc_train:.1%}** · "
                       f"TPR {clf.tpr:.3f} · FPR {clf.fpr:.3f}")
            st.markdown("**The world**")
            p0, p1 = st.slider("True share, first → last period", 0.01, 0.60, step=0.01,
                               key="shares", on_change=clear_preset, format="%.2f")
            periods = st.slider("Periods", 6, 24, key="periods", on_change=clear_preset)
            with st.expander("Simulation"):
                n_items = st.select_slider("Items per period", options=[1000, 2000, 5000, 10000],
                                           value=CFG["n_items"])
                n_gold = st.select_slider("Human-labelled items per period",
                                          options=[50, 100, 200, 500, 1000], value=CFG["n_gold"])
                reps = st.select_slider("Replications", options=[100, 200, 500, 1000],
                                        value=CFG["reps"], help="The published figures use 1,000.")
                seed = int(st.number_input("Seed", value=CFG["seed"], step=1))
                st.caption("As in sim.py, the calibrated trend draws from the seed and the "
                           "over-confident one from seed + 1.")

        with results:
            if p0 == p1:
                st.warning("Pick different first and last shares to see a trend.")
                st.stop()
            with st.spinner(f"Simulating {reps:,} replications…"):
                res = drift(seed + (0 if temperature == 1.0 else 1), reps, n_items, n_gold,
                            temperature, periods, p0, p1, d, pi_train)
            est = res["estimators"]
            cc, acc, sld, ppi = est["CC"], est["ACC"], est["SLD"], est["PPI"]
            first, last = cc["mean_by_period"][0], cc["mean_by_period"][-1]
            with st.container(border=True):
                st.badge("Classify and count", color="red" if cc["slope_ratio"] < 0.9 else "green")
                st.markdown(f"## Counting recovered {cc['slope_ratio']:.0%} of the trend")
                st.markdown(
                    f"A classifier that is **{acc_train:.1%}** accurate where it was trained read "
                    f"a true {p0:.0%} → {p1:.0%} as **{first:.1%} → {last:.1%}**. Counting "
                    f"multiplies every change by TPR − FPR = {clf.tpr - clf.fpr:.3f}, and its "
                    f"naive 95% interval covered the true share in **{cc['ci_coverage']:.1%}** of "
                    "period estimates."
                )
                st.caption(
                    f"Adjusted count recovered {acc['slope_ratio']:.0%} of the trend, EM prior "
                    f"adjustment {sld['slope_ratio']:.0%}, and prediction-powered inference "
                    f"{ppi['slope_ratio']:.0%}, its interval covering the truth "
                    f"{ppi['ci_coverage']:.1%} of the time."
                )
            m1, m2, m3 = st.columns(3)
            m1.metric("Counted share of the trend", f"{cc['slope_ratio']:.3f}", border=True,
                      help="Mean fitted slope of the counted share ÷ true slope.")
            m2.metric("EM prior adjustment (SLD)", f"{sld['slope_ratio']:.3f}", border=True,
                      help="Reads the posteriors, so it inherits their calibration.")
            m3.metric("Prediction-powered (PPI)", f"{ppi['slope_ratio']:.3f}", border=True,
                      help=f"Interval coverage {ppi['ci_coverage']:.1%} with {n_gold} labels "
                      "per period.")

            shown = st.pills("Show", list(NAMES), selection_mode="multi", default=list(NAMES),
                             format_func=SHORT.get, key="shown",
                             help="CC: classify and count. PCC: probabilistic count. ACC: "
                             "adjusted count. SLD: EM prior adjustment. Gold: the human-labelled "
                             "sample alone. PPI: prediction-powered inference.")
            series = ["Truth", *[NAMES[k] for k in NAMES if k in (shown or [])]]
            rows = []
            for k, name in NAMES.items():
                if name in series:
                    rows += [{"Period": t + 1, "Series": name, "Share": v}
                             for t, v in enumerate(est[k]["mean_by_period"])]
            # Truth last, so its dashed line is drawn over the estimators that sit on it.
            rows += [{"Period": t + 1, "Series": "Truth", "Share": v}
                     for t, v in enumerate(res["truth"])]
            is_truth = alt.datum.Series == "Truth"
            chart = (
                alt.Chart(pd.DataFrame(rows))
                .mark_line()
                .encode(
                    x=alt.X("Period:Q", scale=alt.Scale(domain=[1, periods], nice=False),
                            axis=alt.Axis(tickMinStep=1, format="d")),
                    y=alt.Y("Share:Q", scale=alt.Scale(zero=False), axis=alt.Axis(format="%"),
                            title="Mean estimated share"),
                    color=alt.Color("Series:N", scale=alt.Scale(domain=list(COLOURS),
                                                                range=list(COLOURS.values())),
                                    legend=None),
                    strokeDash=alt.condition(is_truth, alt.value([6, 4]), alt.value([1, 0])),
                    strokeWidth=alt.condition(is_truth, alt.value(3), alt.value(2.2)),
                    tooltip=["Series", "Period", alt.Tooltip("Share:Q", format=".2%")],
                )
                .properties(height=360)
            )
            st.altair_chart(chart, width="stretch")
            st.html(html_legend(series))
            scenario = {1.0: "drift_calibrated", 2.0: "drift_overconfident"}.get(temperature)
            if scenario:
                reproduced(res, scenario)

        with st.expander("All six estimators, in numbers"):
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
                f"{reps:,} replications × {periods} periods × {n_items:,} items, {n_gold} human "
                "labels per period. PCC and SLD have no honest standard error, so no interval."
            )

with tab_upgrade:
    if tab_upgrade.open:
        st.markdown(
            "The true share stays flat. Partway through, the classifier is replaced by a better "
            "one. Every change an estimator reports is an artefact, and every *significant* "
            "change is a false trend."
        )
        controls, results = st.columns([1, 2], gap="large")
        with controls, st.container(border=True):
            st.markdown("**The upgrade**")
            p_flat = st.slider("Flat true share", 0.02, 0.50, BRK["p_flat"], 0.01)
            d_old = st.slider("Old model's separation", 1.0, 5.0, BRK["d_old"], 0.1)
            d_new = st.slider("New model's separation", 1.0, 5.0, BRK["d_new"], 0.1)
            break_at = st.slider("Upgrade after period", 2, CFG["periods"] - 2, BRK["break_at"])
            st.caption(
                f"{CFG['reps']:,} replications × {CFG['periods']} periods × {CFG['n_items']:,} "
                f"items, {CFG['n_gold']} human labels per period, seed + 2 as in sim.py."
            )
        with results:
            with st.spinner(f"Simulating {CFG['reps']:,} replications…"):
                br = upgrade(CFG["seed"] + 2, CFG["reps"], CFG["n_items"], CFG["n_gold"],
                             CFG["periods"], p_flat, d_old, d_new, CFG["pi_train"], break_at)
            e = br["estimators"]
            cc_b, ppi_b = e["CC"], e["PPI"]
            with st.container(border=True):
                st.badge("Classify and count",
                         color="red" if cc_b["false_trend_rate"] > 0.2 else "green")
                st.markdown(
                    f"## Counting reported a {cc_b['apparent_change_rel']:+.0%} change that "
                    "never happened"
                )
                st.markdown(
                    f"The true share stayed at {p_flat:.0%}. After the upgrade, counting moved "
                    f"from {cc_b['before_mean']:.1%} to {cc_b['after_mean']:.1%}, and "
                    f"**{cc_b['false_trend_rate']:.0%}** of runs called it a significant trend. "
                    f"Prediction-powered inference moved {ppi_b['apparent_change_rel']:+.1%} and "
                    f"called a false trend in {ppi_b['false_trend_rate']:.1%} of runs, against "
                    "a nominal 5%."
                )
                st.caption(
                    f"Old model TPR {br['old']['tpr']:.3f}, FPR {br['old']['fpr']:.3f} → new "
                    f"model TPR {br['new']['tpr']:.3f}, FPR {br['new']['fpr']:.3f}."
                )
            moves = pd.DataFrame([
                {"Estimator": BREAK_NAMES[k], "When": when, "Mean share": v[col]}
                for k, v in e.items()
                for when, col in (("Before the upgrade", "before_mean"),
                                  ("After the upgrade", "after_mean"))
            ])
            order = list(BREAK_NAMES.values())
            y = alt.Y("Estimator:N", sort=order, title=None, axis=alt.Axis(labelLimit=220))
            x = alt.X("Mean share:Q", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False),
                      title="Mean estimated share")
            link = alt.Chart(moves).mark_line(color="#c9ced6", strokeWidth=3).encode(
                x=x, y=y, detail="Estimator:N")
            dots = alt.Chart(moves).mark_circle(size=140, opacity=1).encode(
                x=x, y=y,
                color=alt.Color("When:N", sort=["Before the upgrade", "After the upgrade"],
                                scale=alt.Scale(range=["#7d8590", "#d1495b"]),
                                legend=alt.Legend(orient="top", title=None, labelLimit=0)),
                tooltip=["Estimator", "When", alt.Tooltip("Mean share:Q", format=".2%")])
            truth = alt.Chart(pd.DataFrame({"Mean share": [p_flat]})).mark_rule(
                strokeDash=[6, 4], color=COLOURS["Truth"], strokeWidth=2).encode(x="Mean share:Q")
            st.altair_chart((link + dots + truth).properties(height=260), width="stretch")
            st.caption("Dashed: the true share, which never moves.")
            st.dataframe(pd.DataFrame([{
                "Estimator": BREAK_NAMES[k],
                "Apparent change": f"{v['apparent_change_rel']:+.1%}",
                "Runs calling it a significant trend": f"{v['false_trend_rate']:.1%}",
            } for k, v in e.items()]), hide_index=True)
            shared = e["ACC_refreshed"]["false_trend_rate_shared_validation"]
            st.caption(
                "Re-measured ACC's false-trend rate is a flaw in the naive test, not in ACC: every "
                "period scored with one validation set shares that set's error. Counting that "
                f"variance once per half instead of averaging it down gives {shared:.1%}."
            )
            reproduced(br, "break")

st.divider()
st.caption(
    "Synthetic, idealised classifier: Gaussian scores, one threshold, pure label shift. What "
    "transfers is the direction and mechanism of each bias, not the magnitudes. "
    f"[Source of this app]({REPO}/tree/main/app) · [Measurements]({REPO}/blob/main/MEASUREMENTS.md)"
)
