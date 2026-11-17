"""The Streamlit page, run headless. Skipped without streamlit or downloaded data."""

from pathlib import Path

import pytest

from load_data import RAW_DIR

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")
HAVE_DATA = any(RAW_DIR.glob("E0_*.csv")) if RAW_DIR.exists() else False


@pytest.mark.skipif(not HAVE_DATA, reason="needs downloaded data")
def test_the_page_shows_a_matchweek_and_a_match():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert len(at.dataframe) == 1 and len(at.dataframe[0].value) > 0
    probabilities = [float(m.value.rstrip("%")) for m in at.metric if m.value.endswith("%")]
    assert len(probabilities) == 3 and abs(sum(probabilities) - 100) <= 2  # rounded


@pytest.mark.skipif(not HAVE_DATA, reason="needs downloaded data")
def test_the_same_team_twice_is_refused():
    at = AppTest.from_file(APP, default_timeout=120).run()
    home = at.selectbox[1]  # [0] is the matchweek
    at.selectbox[2].set_value(home.value).run()
    assert any("two different teams" in e.value for e in at.error)
