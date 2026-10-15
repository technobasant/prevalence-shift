"""The app renders, and at its default settings reproduces the committed results.

Run with the app's own dependencies, as Streamlit Community Cloud installs them:
``uv pip install -r app/requirements.txt pytest`` then ``python -m pytest app``.
"""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).with_name("streamlit_app.py"))


def test_defaults_rerun_the_published_scenario() -> None:
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception
    assert any("every number matches `out/results.json`" in s.value for s in at.success)
    text = " ".join(m.value for m in at.markdown)
    assert "recovered **63%** of the true trend" in text
    assert "covered the truth in **0.0%**" in text


def test_a_changed_setting_runs_live_without_claiming_a_match() -> None:
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.sidebar.select_slider[2].set_value(100).run()  # replications: 1,000 → 100
    assert not at.exception
    assert not at.success
