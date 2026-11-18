import numpy as np
import pandas as pd
import pytest

from explain import contributions, elo_curve, label, weight_table
from train import fit


def toy_model():
    """elo_diff decides the result, elo_expected follows it, noise does nothing."""
    rng = np.random.default_rng(0)
    gap = rng.normal(0, 150, size=600)
    rows = pd.DataFrame({"elo_diff": gap,
                         "elo_expected": 1 / (1 + 10 ** (-gap / 400)),
                         "noise": rng.normal(size=600)})
    rows["result"] = np.select([gap > 60, gap < -60], ["H", "A"], "D")
    features = ["elo_diff", "elo_expected", "noise"]
    return fit(rows, features, c=0.05, l1_ratio=0.5), features, rows


def test_labels_are_readable():
    assert label("home_sotf_5") == "Home shots on target per game, last 5"
    assert label("away_ga_10") == "Away goals conceded per game, last 10"
    assert label("home_venue_ppg_5") == "Home team's home points per game, last 5"
    assert label("away_came_up") == "Away just promoted"
    assert label("elo_diff") == "Elo gap (home minus away)"


def test_weight_table_is_sorted_by_home_minus_away():
    model, features, _ = toy_model()
    table = weight_table(model, features)
    assert list(table.columns) == ["label", "H", "D", "A", "toward_home"]
    assert table["toward_home"].abs().is_monotonic_decreasing
    assert table.index[0] in ("elo_diff", "elo_expected")  # both carry the signal
    assert table.iloc[0]["toward_home"] > 0
    assert table.index[-1] == "noise"  # it carries no signal, so it's last and tiny
    assert abs(table.loc["noise", "toward_home"]) < 0.05 * table.iloc[0]["toward_home"]


def test_contributions_add_up_to_the_log_odds():
    model, features, rows = toy_model()
    row = rows.iloc[[7]]
    pushes, start = contributions(model, features, row)
    p = model.predict_proba(row[features])[0]
    classes = list(model[-1].classes_)
    log_odds = np.log(p[classes.index("H")] / p[classes.index("A")])
    assert start + pushes["toward_home"].sum() == pytest.approx(log_odds)
    assert pushes["toward_home"].abs().is_monotonic_decreasing


def test_missing_values_are_filled_before_explaining():
    model, features, rows = toy_model()
    row = rows.iloc[[3]].copy()
    row["noise"] = np.nan
    pushes, _ = contributions(model, features, row)
    filled = pushes.set_index("feature").loc["noise", "value"]
    assert filled == pytest.approx(rows["noise"].median())


def test_elo_curve_moves_toward_the_stronger_team():
    model, features, _ = toy_model()
    curve = elo_curve(model, features, home_advantage=0, gaps=np.array([-300, 0, 300]))
    probs = curve[["home", "draw", "away"]].to_numpy()
    assert probs.sum(axis=1) == pytest.approx([1, 1, 1])
    assert curve["home"].is_monotonic_increasing and curve["away"].is_monotonic_decreasing
    assert curve["draw"].iloc[1] > max(curve["draw"].iloc[0], curve["draw"].iloc[2])
