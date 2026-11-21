from __future__ import annotations

from streamlit.testing.v1 import AppTest

from .conftest import ROOT


def test_dashboard_runs_and_shows_real_results() -> None:
    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    at.sidebar.slider[0].set_value(400)  # simulation length → shorter run
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert any("SIMULATED" in w.value for w in at.warning)
    assert len(at.dataframe) >= 1
    caption = " ".join(c.value for c in at.caption)
    assert "400 steps" in caption
