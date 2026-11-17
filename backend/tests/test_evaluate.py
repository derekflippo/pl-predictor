import numpy as np
import pandas as pd
import pytest

from evaluate import compare, markdown_table
from load_data import DATA_DIR
from train import TEST, VALIDATE, before_test, load_settings, load_table

HAVE_FEATURES = (DATA_DIR / "features.csv").exists()


def fake_result():
    seasons = ["2015-16", "2016-17"]
    score = {"log_loss": 0.95, "accuracy": 0.53, "brier": 0.57}
    entry = {n: score for n in ("model", "elo_only", "base_rates", "bookmakers")}
    return {"per_season": [{"season": s, **entry} for s in seasons], "overall": entry}


def test_markdown_table_has_a_row_per_season_and_a_total():
    lines = markdown_table(fake_result()).splitlines()
    assert len(lines) == 2 + 2 + 1
    assert lines[2].startswith("| 2015-16 | 0.950 | 0.950 | 0.950 | 0.950 | 53.0% | 53.0% |")
    assert lines[-1].startswith("| **All** |")


@pytest.mark.skipif(not HAVE_FEATURES, reason="needs backend/data/features.csv")
def test_validation_comparison():
    r = compare(before_test(load_table()), VALIDATE, load_settings())
    rows = r["rows"]
    assert len(rows) == 1520 and set(rows["league"]) == {"E0"}
    assert rows["season"].max() < TEST[0]
    assert [e["games"] for e in r["per_season"]] == [380] * 4
    for p in r["probs"].values():
        assert p.sum(axis=1) == pytest.approx(np.ones(len(rows)))
    o = r["overall"]
    assert o["model"]["log_loss"] < o["base_rates"]["log_loss"]
    # The tuned model must match the validation score saved with its settings
    saved = load_settings()["tuning"]["final_validation_log_loss"]
    assert o["model"]["log_loss"] == pytest.approx(saved, abs=1e-5)
