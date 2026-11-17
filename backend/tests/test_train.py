import numpy as np
import pandas as pd
import pytest

from load_data import DATA_DIR
from train import (MODEL_FEATURES, OUTCOMES, bookmaker_probs, evaluate, fit, load_table,
                   match_scores, paired, predict, split)

HAVE_FEATURES = (DATA_DIR / "features.csv").exists()


def toy_rows(n=300, seed=0):
    """One feature x: high x means home wins, low x means away wins."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    result = np.select([x > 0.5, x < -0.5], ["H", "A"], "D")
    return pd.DataFrame({"x": x, "result": result})


def test_split_keeps_test_seasons_out():
    seasons = ["2001-02", "2002-03", "2014-15", "2015-16", "2018-19", "2019-20", "2025-26"]
    table = pd.DataFrame({"season": seasons})
    train, validate = split(table)
    assert list(train["season"]) == ["2002-03", "2014-15"]
    assert list(validate["season"]) == ["2015-16", "2018-19"]


def test_came_down_and_window_sizes_are_left_out():
    for name in ("came_down", "games_5", "games_10"):
        assert f"home_{name}" not in MODEL_FEATURES
        assert f"away_{name}" not in MODEL_FEATURES
    assert "home_elo" in MODEL_FEATURES and len(MODEL_FEATURES) == 34


def test_bookmaker_probabilities_remove_the_margin():
    rows = pd.DataFrame({"odds_home": [2.0], "odds_draw": [3.5], "odds_away": [4.0]})
    probs = bookmaker_probs(rows)[0]
    implied = np.array([1 / 2.0, 1 / 3.5, 1 / 4.0])  # adds up to 1.036: a 3.6% margin
    assert probs.sum() == pytest.approx(1)
    assert probs == pytest.approx(implied / implied.sum())


def test_match_scores_hand_example():
    scores = match_scores(["H", "A"], np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]]))
    assert scores["log_loss"].tolist() == pytest.approx([-np.log(0.5), -np.log(0.2)])
    # H happened: (0.5-1)^2 + 0.3^2 + 0.2^2 = 0.38. A happened: 0.5^2 + 0.3^2 + 0.8^2 = 0.98
    assert scores["brier"].tolist() == pytest.approx([0.38, 0.98])
    assert scores["correct"].tolist() == [True, False]


def test_paired_difference_and_standard_error():
    mean, se = paired([1.0, 2.0, 3.0], [0.0, 1.0, 1.0])
    assert mean == pytest.approx(4 / 3)
    assert se == pytest.approx(np.std([1, 1, 2], ddof=1) / np.sqrt(3))


def test_filling_and_scaling_use_training_rows_only():
    train = toy_rows()
    train.loc[0, "x"] = np.nan
    model = fit(train, ["x"])
    imputer, scaler = model[0], model[1]
    assert imputer.statistics_[0] == pytest.approx(train["x"].median())
    assert scaler.mean_[0] == pytest.approx(train["x"].fillna(train["x"].median()).mean())


def test_probabilities_are_valid_and_in_home_draw_away_order():
    rows = toy_rows()
    probs = predict(fit(rows, ["x"]), pd.DataFrame({"x": [2.0, 0.0, -2.0]}), ["x"])
    assert ((probs >= 0) & (probs <= 1)).all()
    assert probs.sum(axis=1) == pytest.approx([1, 1, 1])
    assert [OUTCOMES[i] for i in probs.argmax(axis=1)] == ["H", "D", "A"]


@pytest.mark.skipif(not HAVE_FEATURES, reason="needs backend/data/features.csv")
def test_first_model_beats_average_shares_and_never_sees_test_seasons():
    r = evaluate(load_table())
    seasons = set(r["train"]["season"]) | set(r["validate"]["season"])
    assert max(seasons) == "2018-19"
    assert r["train_log_loss"] < r["train_base_log_loss"]
    c = r["comparisons"]
    assert c["model"]["log_loss"] < c["base_rates"]["log_loss"]
