import numpy as np
import pandas as pd
import pytest

from metrics import base_rates, bookmaker_probs, calibration, match_scores, paired, summarize


def test_bookmaker_probabilities_remove_the_margin():
    rows = pd.DataFrame({"odds_home": [2.0], "odds_draw": [3.5], "odds_away": [4.0]})
    probs = bookmaker_probs(rows)[0]
    implied = np.array([1 / 2.0, 1 / 3.5, 1 / 4.0])  # adds up to 1.036: a 3.6% margin
    assert probs.sum() == pytest.approx(1)
    assert probs == pytest.approx(implied / implied.sum())


def test_missing_odds_stop_with_the_game_named():
    rows = pd.DataFrame({"season": ["2030-31", "2030-31"], "home_team": ["A", "C"],
                         "away_team": ["B", "D"], "odds_home": [2.0, np.nan],
                         "odds_draw": [3.5, 3.4], "odds_away": [4.0, 3.0]})
    with pytest.raises(ValueError, match="1 game.*2030-31 C D"):
        bookmaker_probs(rows)


def test_match_scores_hand_example():
    scores = match_scores(["H", "A"], np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]]))
    assert scores["log_loss"].tolist() == pytest.approx([-np.log(0.5), -np.log(0.2)])
    # H happened: (0.5-1)^2 + 0.3^2 + 0.2^2 = 0.38. A happened: 0.5^2 + 0.3^2 + 0.8^2 = 0.98
    assert scores["brier"].tolist() == pytest.approx([0.38, 0.98])
    assert scores["correct"].tolist() == [True, False]
    assert summarize(scores) == pytest.approx({"log_loss": (-np.log(0.5) - np.log(0.2)) / 2,
                                               "brier": 0.68, "accuracy": 0.5})


def test_paired_difference_and_standard_error():
    mean, se = paired([1.0, 2.0, 3.0], [0.0, 1.0, 1.0])
    assert mean == pytest.approx(4 / 3)
    assert se == pytest.approx(np.std([1, 1, 2], ddof=1) / np.sqrt(3))


def test_base_rates_repeat_the_training_shares():
    train = pd.DataFrame({"result": ["H", "H", "D", "A"]})
    assert base_rates(train, 2).tolist() == [[0.5, 0.25, 0.25]] * 2


def test_calibration_groups_by_predicted_probability():
    probs = np.array([[0.14, 0.24, 0.62], [0.12, 0.24, 0.64], [0.65, 0.20, 0.15]])
    cal = calibration(["A", "H", "H"], probs)
    away = cal[(cal["outcome"] == "A") & (cal["low"].round(1) == 0.6)].iloc[0]
    assert away["games"] == 2 and away["actual"] == 0.5 and away["predicted"] == pytest.approx(0.63)
