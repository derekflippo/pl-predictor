import csv
import json

import pandas as pd
import pytest

from fixtures import (TEAM_NAMES, UnknownTeamError, add_results, latest_fixtures_file,
                      load_fixtures, upcoming)
from load_data import load_matches

NAMES = {"Arsenal FC": "Arsenal", "Leeds United FC": "Leeds", "Hull City AFC": "Hull"}
FIXTURES_FILE = latest_fixtures_file()


def game(matchweek, day, home, away, score=None):
    g = {"round": f"Matchday {matchweek}", "date": day, "time": "15:00",
         "team1": home, "team2": away}
    if score is not None:
        g["score"] = score
    return g


def write_fixtures(tmp_path, *games, season="2026-27"):
    path = tmp_path / f"fixtures_{season}.json"
    path.write_text(json.dumps({"name": "Premier League", "matches": list(games)}))
    return path


def results(*rows):
    """Minimal football-data results table: (league, season, date, home, away, hg, ag, result)."""
    df = pd.DataFrame(rows, columns=["league", "season", "date", "home_team", "away_team",
                                     "home_goals", "away_goals", "result"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_names_are_mapped_and_matchweek_parsed(tmp_path):
    path = write_fixtures(tmp_path, game(6, "2026-10-10", "Arsenal FC", "Leeds United FC"))
    row = load_fixtures(path, NAMES).iloc[0]
    assert row["season"] == "2026-27"
    assert row["matchweek"] == 6
    assert (row["home_team"], row["away_team"]) == ("Arsenal", "Leeds")


def test_both_openfootball_score_shapes_load(tmp_path):
    path = write_fixtures(tmp_path,
                          game(1, "2026-08-21", "Arsenal FC", "Hull City AFC",
                               score={"ht": [2, 0], "ft": [3, 0]}),
                          game(1, "2026-08-22", "Leeds United FC", "Arsenal FC",
                               score=[0, 0]))
    assert len(load_fixtures(path, NAMES)) == 2


def test_unknown_team_raises(tmp_path):
    path = write_fixtures(tmp_path, game(1, "2026-08-22", "Arsenal FC", "Derby County FC"))
    with pytest.raises(UnknownTeamError, match="Derby County FC"):
        load_fixtures(path, NAMES)


def test_postponed_game_still_matches_its_result(tmp_path):
    path = write_fixtures(tmp_path, game(8, "2026-10-24", "Arsenal FC", "Leeds United FC"))
    fixtures = load_fixtures(path, NAMES)
    played = results(("E0", "2026-27", "2026-12-02", "Arsenal", "Leeds", 2, 0, "H"))

    row = add_results(fixtures, played).iloc[0]
    assert row["result"] == "H"
    assert row["played_on"] == pd.Timestamp("2026-12-02")
    assert upcoming(fixtures, played).empty


def test_reverse_fixture_is_a_different_game(tmp_path):
    path = write_fixtures(tmp_path,
                          game(3, "2026-09-05", "Arsenal FC", "Leeds United FC"),
                          game(20, "2027-01-02", "Leeds United FC", "Arsenal FC"))
    fixtures = load_fixtures(path, NAMES)
    played = results(("E0", "2026-27", "2027-01-02", "Leeds", "Arsenal", 1, 1, "D"))

    todo = upcoming(fixtures, played)
    assert list(zip(todo["home_team"], todo["away_team"])) == [("Arsenal", "Leeds")]


def test_results_from_other_seasons_and_leagues_are_ignored(tmp_path):
    path = write_fixtures(tmp_path, game(3, "2026-09-05", "Arsenal FC", "Leeds United FC"))
    fixtures = load_fixtures(path, NAMES)
    other = results(("E0", "2025-26", "2025-09-06", "Arsenal", "Leeds", 3, 0, "H"),
                    ("E1", "2026-27", "2026-09-05", "Arsenal", "Leeds", 1, 0, "H"))
    assert len(upcoming(fixtures, other)) == 1


def test_team_names_file_is_clean():
    with open(TEAM_NAMES, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    keys = [r["openfootball"] for r in rows]
    assert len(keys) == len(set(keys)), "duplicate openfootball names"
    assert all(r["openfootball"].strip() and r["football_data"].strip() for r in rows)


@pytest.mark.skipif(FIXTURES_FILE is None, reason="run backend/download_data.py first")
def test_current_season_teams_all_map_to_football_data_names():
    fixtures = load_fixtures(FIXTURES_FILE)  # raises if any team is unmapped
    assert len(fixtures) == 380
    teams = set(fixtures["home_team"])
    assert len(teams) == 20 and teams == set(fixtures["away_team"])

    matches = load_matches(leagues=["E0"])
    this_season = matches[matches["season"] == fixtures["season"].iloc[0]]
    if len(this_season):  # once the season has started, spellings must agree
        assert teams == set(this_season["home_team"]) | set(this_season["away_team"])
