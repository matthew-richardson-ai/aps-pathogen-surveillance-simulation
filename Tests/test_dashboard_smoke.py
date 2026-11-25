"""Smoke-test that Streamlit can build the full student-facing APS page.

This complements algorithm unit tests: the Python test suite alone cannot
catch integration errors in the dashboard's initial widget/Plotly rendering.
"""
from pathlib import Path
import sys

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "OB"))


def test_dashboard_initial_render_has_no_streamlit_exceptions():
    app = AppTest.from_file(str(ROOT / "OB" / "live_dashboard.py"), default_timeout=30)
    app.run()
    if app.exception:
        pytest.fail(f"Streamlit rendering raised: {[item.message for item in app.exception]}")
    assert any("Adaptive Pathogen Surveillance" in title.value for title in app.title)
