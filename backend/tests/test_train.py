import numpy as np
import pandas as pd
import pytest

from load_data import DATA_DIR
from train import (C_GRID, GROUPS, MODEL_FEATURES, OUTCOMES, TEST, TRAIN_FROM, VALIDATE,
                   add_derived, before_test, bookmaker_probs, clearly_better, fit, load_settings,
                   load_table, make_model, match_scores, model_features, one_se_choice, paired,
                   predict, train_final, walk_forward)

HAVE_FEATURES = (DATA_DIR / "features.csv").exists()


def toy_rows(n=300, seed=0):
    """One feature x: high x means home wins, low x means away wins."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    result = np.select([x > 0.5, x < -0.5], ["H", "A"], "D")
    return pd.DataFrame({"x": x, "result": result})


def season_table():
    """Two leagues, one row each per season from 2001-02 to 2020-21."""
    seasons = [f"{y}-{str(y + 1)[2:]}" for y in range(2001, 2021)]
    return pd.DataFrame({"season": seasons * 2, "league": ["E0"] * 20 + ["E1"] * 20})


def test_walk_forward_trains_only_on_earlier_seasons():
    seen = {}

    def spy(train, rows):
        seen[rows["season"].iloc[0]] = (set(train["season"]), set(train["league"]), len(rows))
        return np.full((len(rows), 3), 1 / 3)

    rows, probs = walk_forward(season_table(), VALIDATE, spy)
    assert list(rows["season"]) == VALIDATE and set(rows["league"]) == {"E0"}
    assert probs.shape == (4, 3)
    for season in VALIDATE:
        train_seasons, leagues, n = seen[season]
        assert min(train_seasons) == TRAIN_FROM and max(train_seasons) < season
        assert leagues == {"E0", "E1"} and n == 1


def test_before_test_drops_the_test_seasons():
    kept = before_test(season_table())
    assert kept["season"].max() == "2018-19" and TEST[0] not in set(kept["season"])


def test_model_features():
    for name in ("came_down", "games_5", "games_10", "elo"):
        assert f"home_{name}" not in MODEL_FEATURES and f"away_{name}" not in MODEL_FEATURES
    assert {"elo_diff", "elo_sum"} <= set(MODEL_FEATURES) and len(MODEL_FEATURES) == 34
    in_groups = [f for group in GROUPS.values() for f in group]
    assert sorted(in_groups + ["elo_diff"]) == sorted(MODEL_FEATURES)
    settings = {"features": ["elo_diff"], "championship_rows": True}
    assert model_features(settings) == ["elo_diff", "championship"]


def test_elo_difference_and_sum():
    table = add_derived(pd.DataFrame({"home_elo": [1600.0], "away_elo": [1450.0],
                                      "league": ["E1"]}))
    assert table.loc[0, "elo_diff"] == 150 and table.loc[0, "elo_sum"] == 3050
    assert table.loc[0, "championship"] == 1


def test_c_grid_runs_from_strong_to_light_penalty():
    assert C_GRID[0] == 0.0001 and C_GRID[-1] == 1 and len(C_GRID) == 17


def test_one_se_rule_picks_the_smallest_c_close_to_the_best():
    base = 1.0 + np.random.default_rng(0).normal(size=1000)
    wobble = np.r_[0.01, -0.01].repeat(500)  # per-game differences, so the SE is about 0.0003
    losses = {
        0.001: base + 0.05,               # clearly worse than the best
        0.01: base + 0.0001 + wobble,     # 0.0001 worse: within one SE of the best
        0.1: base,                        # the best
    }
    assert one_se_choice(losses) == (0.01, 0.1)


def test_clearly_better_needs_more_than_one_standard_error():
    old = np.ones(100)
    assert clearly_better(old - 0.01, old + np.r_[0.001, -0.001].repeat(50)) is True
    assert clearly_better(old, old) is False


def test_elastic_net_uses_saga():
    lr = make_model(c=0.1, l1_ratio=0.5)[-1]
    assert lr.penalty == "elasticnet" and lr.solver == "saga" and lr.l1_ratio == 0.5
    assert make_model(c=0.1)[-1].penalty == "l2"


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


@pytest.mark.parametrize("l1_ratio", [None, 0.5])
def test_probabilities_are_valid_and_in_home_draw_away_order(l1_ratio):
    rows = toy_rows()
    model = fit(rows, ["x"], l1_ratio=l1_ratio)
    probs = predict(model, pd.DataFrame({"x": [2.0, 0.0, -2.0]}), ["x"])
    assert ((probs >= 0) & (probs <= 1)).all()
    assert probs.sum(axis=1) == pytest.approx([1, 1, 1])
    assert [OUTCOMES[i] for i in probs.argmax(axis=1)] == ["H", "D", "A"]


def test_saved_settings_are_usable():
    settings = load_settings()
    assert set(settings["features"]) <= set(MODEL_FEATURES)
    assert settings["c"] in C_GRID
    assert settings["l1_ratio"] in (None, 0.5, 1.0)
    assert settings["tuning"]["validation_seasons"] == f"{VALIDATE[0]} to {VALIDATE[-1]}"


@pytest.mark.skipif(not HAVE_FEATURES, reason="needs backend/data/features.csv")
def test_final_model_trains_on_finished_premier_league_matches():
    table = load_table()
    settings = load_settings()
    model = train_final(table, settings)
    rows = table[(table["league"] == "E0") & (table["season"] == VALIDATE[0])]
    probs = predict(model, rows, model_features(settings))
    assert probs.sum(axis=1) == pytest.approx(np.ones(len(rows)))
