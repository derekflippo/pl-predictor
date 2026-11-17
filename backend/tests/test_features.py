import numpy as np
import pandas as pd
import pytest

from elo import Elo
from features import FEATURE_COLUMNS, build_features, features_and_state, features_for, rows_for
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


def two_leagues():
    """E wins the made-up Championship and goes up; D goes down."""
    pl = games(
        ("2000-01", "2000-08-19", "A", "B", 2, 0), ("2000-01", "2000-08-19", "C", "D", 1, 1),
        ("2000-01", "2000-08-26", "B", "C", 0, 1), ("2000-01", "2000-08-26", "D", "A", 3, 1),
        ("2001-02", "2001-08-18", "E", "A", 1, 1), ("2001-02", "2001-08-18", "B", "C", 0, 0),
        ("2001-02", "2001-08-25", "A", "B", 1, 0), ("2001-02", "2001-08-25", "C", "E", 0, 2),
    )
    ch = games(
        ("2000-01", "2000-08-19", "E", "F", 3, 0), ("2000-01", "2000-08-19", "G", "H", 0, 0),
        ("2000-01", "2000-08-26", "F", "G", 1, 2), ("2000-01", "2000-08-26", "H", "E", 0, 2),
        ("2001-02", "2001-08-18", "D", "F", 2, 1), ("2001-02", "2001-08-18", "G", "H", 1, 1),
    )
    ch["league"] = "E1"
    return pd.concat([pl, ch]).sort_values("date", kind="stable").reset_index(drop=True)


def test_raw_form_starts_again_in_a_new_league():
    table = build_features(two_leagues(), elo())
    first = table[(table["home_team"] == "E") & (table["league"] == "E0")].iloc[0]
    assert first["home_came_up"] == 1
    # E's two Championship wins don't count as Premier League form...
    assert first["home_games_5"] == 0
    for name in ("ppg_5", "gf_10", "ga_5", "sotf_5", "sota_10", "venue_ppg_5"):
        assert np.isnan(first[f"home_{name}"])
    # ...but its results against Elo's expectation do, because they're opponent-adjusted.
    assert not np.isnan(first["home_vs_exp_5"])
    assert not np.isnan(first["home_rest_days"])

    second = table[(table["away_team"] == "E") & (table["league"] == "E0")].iloc[0]
    assert second["away_games_5"] == 1  # only the 1-1 draw with A
    assert second["away_ppg_5"] == 1 and second["away_gf_5"] == 1
    assert np.isnan(second["away_venue_ppg_5"])  # its only away game was in the Championship

    relegated = table[(table["home_team"] == "D") & (table["league"] == "E1")].iloc[0]
    assert relegated["home_came_down"] == 1 and relegated["home_games_10"] == 0


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


def test_one_walk_gives_the_same_rows_and_state_as_two():
    table, state = features_and_state(SEASON, elo())
    assert table.equals(build_features(SEASON, elo()))
    upcoming = pd.DataFrame({"league": ["E0"], "date": [pd.Timestamp("2031-01-01")],
                             "home_team": [SEASON["away_team"].iloc[0]],
                             "away_team": [SEASON["home_team"].iloc[0]]})
    assert rows_for(upcoming, state).equals(features_for(upcoming, SEASON, elo()))
