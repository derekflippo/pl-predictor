import numpy as np
import pandas as pd
import pytest

from elo import Elo
from features import FEATURE_COLUMNS, build_features, features_for
from load_data import RAW_DIR

HAVE_DATA = any(RAW_DIR.glob("E0_*.csv"))


def games(*rows):
    """Made-up matches: (season, date, home, away, home_goals, away_goals), all E0."""
    df = pd.DataFrame(rows, columns=["season", "date", "home_team", "away_team",
                                     "home_goals", "away_goals"])
    df.insert(0, "league", "E0")
    df["date"] = pd.to_datetime(df["date"])
    df["result"] = np.select([df["home_goals"] > df["away_goals"],
                              df["home_goals"] < df["away_goals"]], ["H", "A"], "D")
    df["home_shots_on_target"] = 5.0
    df["away_shots_on_target"] = 3.0
    return df


def elo():
    return Elo(k=20, s=0.2, a=0.5)


SEASON = games(
    ("2000-01", "2000-08-19", "A", "B", 2, 0),
    ("2000-01", "2000-08-19", "C", "D", 1, 1),
    ("2000-01", "2000-08-26", "B", "C", 0, 1),
    ("2000-01", "2000-08-26", "D", "A", 3, 1),
    ("2000-01", "2000-09-02", "A", "C", 2, 2),
    ("2000-01", "2000-09-02", "B", "D", 1, 0),
    ("2000-01", "2000-09-16", "C", "A", 0, 4),
    ("2000-01", "2000-09-16", "D", "B", 2, 2),
)


def test_windows_use_the_games_available():
    table = build_features(SEASON, elo())
    row = table.iloc[6]  # C v A: A has played 3 games (W 2-0, L 1-3, D 2-2)
    assert row["away_team"] == "A"
    assert row["away_games_5"] == 3 and row["away_games_10"] == 3
    assert row["away_ppg_5"] == pytest.approx((3 + 0 + 1) / 3)
    assert row["away_gf_5"] == pytest.approx((2 + 1 + 2) / 3)
    assert row["away_ga_5"] == pytest.approx((0 + 3 + 2) / 3)
    # A's away games so far: one, the 1-3 defeat at D
    assert row["away_venue_ppg_5"] == 0
    assert row["away_rest_days"] == 14


def test_first_game_has_empty_windows():
    row = build_features(SEASON, elo()).iloc[0]
    assert row["home_games_5"] == 0
    assert np.isnan(row["home_ppg_5"]) and np.isnan(row["home_rest_days"])
    assert np.isnan(row["home_came_up"])  # first season: previous league unknown


def test_rest_days_are_capped():
    later = pd.concat([SEASON, games(("2000-01", "2000-12-01", "A", "B", 0, 0))])
    assert build_features(later, elo()).iloc[-1]["home_rest_days"] == 21


def test_no_leakage_from_the_last_result():
    changed = SEASON.copy()
    changed.loc[changed.index[-1], ["home_goals", "away_goals", "result"]] = [9, 0, "H"]
    a, b = build_features(SEASON, elo()), build_features(changed, elo())
    pd.testing.assert_frame_equal(a[FEATURE_COLUMNS], b[FEATURE_COLUMNS])


def test_same_date_games_dont_see_each_other():
    changed = SEASON.copy()
    changed.loc[changed.index[4], ["home_goals", "away_goals", "result"]] = [0, 6, "A"]
    a, b = build_features(SEASON, elo()), build_features(changed, elo())
    # Row 5 is on the same date as the changed row 4.
    pd.testing.assert_series_equal(a.loc[5, FEATURE_COLUMNS], b.loc[5, FEATURE_COLUMNS])
    assert a.loc[4, "home_advantage"] == a.loc[5, "home_advantage"]


def test_both_paths_agree_on_made_up_data():
    table = build_features(SEASON, elo())
    match = SEASON.iloc[[6]]
    before = SEASON[SEASON["date"] < match["date"].iloc[0]]
    row = features_for(match, before, elo())
    pd.testing.assert_series_equal(row.loc[0, FEATURE_COLUMNS], table.loc[6, FEATURE_COLUMNS],
                                   check_names=False)


@pytest.mark.skipif(not HAVE_DATA, reason="needs downloaded data")
def test_both_paths_agree_on_real_data():
    from load_data import load_matches

    matches = load_matches()
    table = build_features(matches, elo())
    # A mid-season Premier League match, so every team of the season has
    # already appeared in the earlier data.
    pick = table[(table["league"] == "E0") & (table["season"] == "2015-16")
                 & (table["date"] == "2016-01-02")].index[0]
    match = table.loc[[pick]]
    before = matches[matches["date"] < match["date"].iloc[0]]
    row = features_for(match, before, elo())
    pd.testing.assert_series_equal(row.loc[0, FEATURE_COLUMNS], table.loc[pick, FEATURE_COLUMNS],
                                   check_names=False)
