from datetime import date

import numpy as np
import pandas as pd
import pytest

from load_data import DATA_DIR
from predict import (PredictError, check_date, check_season, check_teams, favourite,
                     latest_result, missing_results, next_games, predict_games, season_teams,
                     train_model)
from train import TRAIN_FROM, load_settings

HAVE_DATA = (DATA_DIR / "matches.csv").exists()


def toy_fixtures():
    return pd.DataFrame({
        "season": "2026-27",
        "matchweek": [1, 2, 2, 3],
        "date": pd.to_datetime(["2026-08-15", "2026-08-22", "2026-08-23", "2026-08-29"]),
        "kickoff": ["15:00", "15:00", "16:30", "15:00"],
        "home_team": ["A", "B", "C", "A"],
        "away_team": ["B", "C", "A", "C"],
    })


def toy_matches(played=(("A", "B", "2026-08-15", "H"),)):
    return pd.DataFrame({
        "league": "E0", "season": "2026-27",
        "home_team": [p[0] for p in played], "away_team": [p[1] for p in played],
        "date": pd.to_datetime([p[2] for p in played]), "result": [p[3] for p in played],
        "home_goals": 1, "away_goals": 0,
    })


def test_favourite_names_the_team_or_says_close_call():
    assert favourite("Arsenal", "Chelsea", np.array([0.50, 0.28, 0.22])) == "Arsenal"
    assert favourite("Arsenal", "Chelsea", np.array([0.20, 0.25, 0.55])) == "Chelsea"
    assert favourite("Arsenal", "Chelsea", np.array([0.30, 0.40, 0.30])) == "Draw"
    assert favourite("Arsenal", "Chelsea", np.array([0.39, 0.26, 0.35])) == "Close call"


def test_next_matchweek_skips_played_games_and_past_dates():
    week, games = next_games(toy_fixtures(), toy_matches(), date(2026, 8, 20))
    assert week == 2 and list(games["home_team"]) == ["B", "C"]


def test_a_named_matchweek_keeps_only_unplayed_games():
    played = (("A", "B", "2026-08-15", "H"), ("B", "C", "2026-08-22", "D"))
    week, games = next_games(toy_fixtures(), toy_matches(played), date(2026, 8, 1), matchweek=2)
    assert week == 2 and list(games["home_team"]) == ["C"]
    with pytest.raises(PredictError, match="no unplayed games"):
        next_games(toy_fixtures(), toy_matches(), date(2026, 8, 1), matchweek=1)


def test_no_predictions_before_the_season_has_results():
    matches = toy_matches().assign(season="2025-26")
    with pytest.raises(PredictError, match="No 2026-27 results yet"):
        check_season(toy_fixtures(), matches)
    check_season(toy_fixtures(), toy_matches())  # a played game: no error


def test_team_names_are_checked():
    teams = {"Arsenal", "Chelsea", "Man United"}
    check_teams("Arsenal", "Chelsea", teams)
    with pytest.raises(PredictError, match="Unknown team name: Chelsae.*Man United"):
        check_teams("Arsenal", "Chelsae", teams)
    with pytest.raises(PredictError, match="the same"):
        check_teams("Arsenal", "Arsenal", teams)


def test_a_named_game_cannot_be_dated_before_the_latest_result():
    latest = pd.Timestamp("2026-11-08")
    check_date(date(2026, 11, 8), latest)
    check_date(date(2026, 12, 26), latest)
    with pytest.raises(PredictError, match="on or after the latest result, 08 Nov 2026"):
        check_date(date(2026, 9, 1), latest)


def test_missing_results_are_games_listed_since_the_last_download():
    fixtures = pd.concat([toy_fixtures(), pd.DataFrame({
        "season": ["2026-27"], "matchweek": [1], "date": pd.to_datetime(["2026-08-10"]),
        "kickoff": ["15:00"], "home_team": ["B"], "away_team": ["A"]})])  # postponed
    missing = missing_results(fixtures, toy_matches(), date(2026, 8, 25))
    assert list(missing["home_team"]) == ["B", "C"]  # matchweek 2, not the postponed game
    assert missing_results(fixtures, toy_matches(), date(2026, 8, 16)).empty


def test_season_teams_come_from_the_latest_premier_league_season():
    matches = pd.concat([toy_matches().assign(season="2025-26", home_team="Old"),
                         toy_matches(), toy_matches().assign(league="E1", home_team="Champ")])
    assert season_teams(matches) == {"A", "B"}


@pytest.fixture(scope="module")
def trained():
    from load_data import load_matches
    matches = load_matches()
    settings = load_settings()
    model, table, state = train_model(matches, settings)
    return matches, settings, model, table, state


@pytest.mark.skipif(not HAVE_DATA, reason="needs backend/data/matches.csv")
def test_model_trains_on_finished_matches_from_2002_03(trained):
    matches, _, _, table, state = trained
    assert len(table) == len(matches)  # one feature row per played match
    assert table["result"].notna().all() and table["season"].min() < TRAIN_FROM
    assert latest_result(state) == matches["date"].max()


@pytest.mark.skipif(not HAVE_DATA, reason="needs backend/data/matches.csv")
def test_predictions_are_valid_probabilities(trained):
    matches, settings, model, _, state = trained
    teams = sorted(season_teams(matches))
    games = pd.DataFrame({"date": [pd.Timestamp(matches["date"].max()) + pd.Timedelta(days=7)] * 2,
                          "home_team": teams[:2], "away_team": teams[2:4]})
    out = predict_games(games, state, model, settings)
    probs = out[["p_home", "p_draw", "p_away"]].to_numpy()
    assert ((probs > 0) & (probs < 1)).all()
    assert probs.sum(axis=1) == pytest.approx([1, 1])
    assert list(out["home_team"]) == teams[:2]


@pytest.mark.skipif(not HAVE_DATA, reason="needs backend/data/matches.csv")
def test_a_game_listed_before_the_latest_result_is_treated_as_played_on_it(trained):
    matches, settings, model, _, state = trained
    teams = sorted(season_teams(matches))
    latest = matches["date"].max()
    games = pd.DataFrame({"date": [latest - pd.Timedelta(days=30), latest],
                          "home_team": [teams[0]] * 2, "away_team": [teams[1]] * 2})
    out = predict_games(games, state, model, settings)
    probs = out[["p_home", "p_draw", "p_away"]].to_numpy()
    assert probs[0] == pytest.approx(probs[1])
    assert out["date"].iloc[0] == latest - pd.Timedelta(days=30)  # the listed date is kept
