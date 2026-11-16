import pandas as pd
import pytest

from check_ratings import check_elo_alone, check_end_of_season, check_promoted, league_table, run
from load_data import RAW_DIR

HAVE_DATA = any(RAW_DIR.glob("E0_*.csv"))


def test_league_table_orders_by_points_then_goal_difference_then_goals():
    games = pd.DataFrame({
        "home_team": ["A", "B", "C"], "away_team": ["B", "C", "A"],
        "home_goals": [3, 1, 1], "away_goals": [0, 0, 1],
    })
    # A: W 3-0, D 1-1 -> 4 pts, GD +3. B: L, W -> 3 pts. C: L, D -> 1 pt.
    assert list(league_table(games).index) == ["A", "B", "C"]


@pytest.fixture(scope="module")
def real():
    from load_data import load_matches
    matches = load_matches()
    elo, ratings = run(matches)
    return matches, elo, ratings


@pytest.mark.skipif(not HAVE_DATA, reason="needs downloaded data")
def test_check_1_end_of_season_elo_matches_the_table(real):
    matches, elo, _ = real
    assert check_end_of_season(matches, elo)["passed"]


@pytest.mark.skipif(not HAVE_DATA, reason="needs downloaded data")
def test_check_2_elo_beats_base_rates(real):
    matches, _, ratings = real
    assert check_elo_alone(matches, ratings)["passed"]


@pytest.mark.skipif(not HAVE_DATA, reason="needs downloaded data")
def test_check_4_runs(real):
    # Information only; this just makes sure the report can be produced.
    matches, elo, _ = real
    result = check_promoted(matches, elo)
    assert result["down_count"] + result["stayed_count"] > 50
