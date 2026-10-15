"""The app renders, reruns the published scenarios exactly, and runs everything else live
without claiming a match.

Run with the app's own dependencies, as Streamlit Community Cloud installs them:
``uv pip install -r app/requirements.txt pytest`` then ``python -m pytest app``.
"""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).with_name("streamlit_app.py"))
MATCH = "every number matches `out/results.json`"


def run_app(view: str | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    if view:
        at.query_params["view"] = view
    at.run()
    assert not at.exception
    return at


def text(at: AppTest) -> str:
    return "\n".join([m.value for m in at.markdown] + [c.value for c in at.caption])


def matched(at: AppTest) -> bool:
    return any(MATCH in s.value for s in at.success)


def test_defaults_rerun_scenario_a_exactly() -> None:
    at = run_app()
    assert matched(at)
    assert "## Counting recovered 63% of the trend" in text(at)
    assert "covered the true share in **0.0%**" in text(at)
    assert "0.628" in [m.value for m in at.metric]


def test_the_over_confident_scenario_reruns_scenario_b_exactly() -> None:
    at = run_app()
    at.pills[0].set_value("An over-confident model").run()
    assert not at.exception
    assert matched(at)
    assert "0.775" in [m.value for m in at.metric]  # SLD inherits the bad calibration


def test_a_scenario_off_the_published_grid_runs_live_without_claiming_a_match() -> None:
    at = run_app()
    at.pills[0].set_value("A weaker classifier").run()
    assert not at.exception
    assert not matched(at)
    assert float(at.metric[0].value) < 0.3  # counting keeps TPR − FPR ≈ 0.22 of the trend


def test_the_upgrade_view_reruns_scenario_c_exactly() -> None:
    at = run_app("upgrade")
    assert matched(at)
    assert "## Counting reported a +24% change that never happened" in text(at)
