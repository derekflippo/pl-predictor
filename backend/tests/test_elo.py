import json

import pandas as pd
import pytest

from elo import (SETTINGS_PATH, Elo, expected_score, home_advantage_points,
                 margin_multiplier, run_elo)
from load_data import RAW_DIR

HAVE_DATA = any(RAW_DIR.glob("E0_*.csv"))


def games(*rows):
    """Made-up matches: (league, season, date, home, away, home_goals, away_goals)."""
    df = pd.DataFrame(rows, columns=["league", "season", "date", "home_team",
                                     "away_team", "home_goals", "away_goals"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_equal_ratings_without_home_advantage_expect_half():
    assert expected_score(1500, 1500, 0) == 0.5


def test_home_win_moves_each_team_by_half_k():
    elo = Elo(k=20, s=0, a=0)
    elo.start_season("2000-01", {"E0": {"A", "B"}})
    _, _, h, e = elo.predict("E0", "A", "B")
    assert h == 0 and e == 0.5
    elo.update("E0", "A", "B", 1, 0, e)
    assert elo.ratings["E0"] == {"A": 1510, "B": 1490}


def test_margin_multiplier():
    assert margin_multiplier(0) == 1
    assert margin_multiplier(1) == margin_multiplier(-1) == 1
    assert margin_multiplier(2) == 1.5
    assert margin_multiplier(3) == 1.75
    assert margin_multiplier(-7) == 2.25


def test_home_advantage_points():
    assert home_advantage_points(0.5) == 0
    assert home_advantage_points(0.58) == pytest.approx(56, abs=1)


def test_ratings_balance_within_a_season():
    matches = games(
        ("E0", "2000-01", "2000-08-19", "A", "B", 3, 0),
        ("E0", "2000-01", "2000-08-19", "C", "D", 1, 1),
        ("E0", "2000-01", "2000-08-26", "B", "C", 0, 2),
        ("E0", "2000-01", "2000-08-26", "D", "A", 5, 4),
    )
    elo = Elo(k=20, s=0.3, a=0.5)
    run_elo(matches, elo)
    assert sum(elo.ratings["E0"].values()) == pytest.approx(4 * 1500)


def two_league_history():
    """Season 1: PL = A, B, C, D and Championship = E, F, G, H.

    Season 2: D goes down and E comes up. H drops to League One, and I
    arrives from League One.
    """
    return games(
        ("E0", "2000-01", "2000-08-19", "A", "D", 4, 0),
        ("E0", "2000-01", "2000-08-19", "B", "C", 1, 1),
        ("E1", "2000-01", "2000-08-19", "E", "H", 3, 0),
        ("E1", "2000-01", "2000-08-19", "F", "G", 0, 0),
        ("E0", "2001-02", "2001-08-18", "A", "E", 1, 0),
        ("E0", "2001-02", "2001-08-18", "B", "C", 1, 0),
        ("E1", "2001-02", "2001-08-18", "D", "I", 1, 0),
        ("E1", "2001-02", "2001-08-18", "F", "G", 1, 0),
    )


def test_season_boundary_converts_teams_that_change_league():
    matches = two_league_history()
    elo = Elo(k=20, s=0, a=0.5)
    run_elo(matches[matches["season"] == "2000-01"], elo)
    pl, ch = dict(elo.ratings["E0"]), dict(elo.ratings["E1"])

    elo2 = Elo(k=20, s=0, a=0.5)  # a fresh run over both seasons
    run_elo(matches, elo2)
    start_pl = elo2.season_start[("E0", "2001-02")]
    start_ch = elo2.season_start[("E1", "2001-02")]

    pl_mean, ch_mean = sum(pl.values()) / 4, sum(ch.values()) / 4
    converted_pl = {"A": pl["A"], "B": pl["B"], "C": pl["C"],
                    # E is promoted: D's old rating plus half of E's lead
                    # over the Championship average
                    "E": pl["D"] + 0.5 * (ch["E"] - ch_mean)}
    converted_ch = {"F": ch["F"], "G": ch["G"],
                    # D is relegated: E's old rating plus half of D's gap
                    # to the Premier League average
                    "D": ch["E"] + 0.5 * (pl["D"] - pl_mean),
                    # I is new from League One: it takes H's place
                    "I": ch["H"]}
    # Then each league is shifted so its average is 1500 again.
    for converted, start in ((converted_pl, start_pl), (converted_ch, start_ch)):
        shift = 1500 - sum(converted.values()) / 4
        assert start == pytest.approx({t: r + shift for t, r in converted.items()})
    assert elo2.changes[("E0", "2001-02")] == {"A": None, "B": None, "C": None, "E": "up"}
    assert elo2.changes[("E1", "2001-02")]["D"] == "down"
    assert elo2.changes[("E1", "2001-02")]["I"] == "up"


def test_summer_shrink_moves_ratings_toward_the_league_average():
    matches = two_league_history()
    flat, shrunk = Elo(k=20, s=0, a=0), Elo(k=20, s=0.5, a=0)
    run_elo(matches, flat)
    run_elo(matches, shrunk)
    before = flat.season_start[("E0", "2001-02")]
    after = shrunk.season_start[("E0", "2001-02")]
    mean = sum(before.values()) / len(before)
    for team in before:
        assert after[team] == pytest.approx(mean + 0.5 * (before[team] - mean))


def test_every_season_starts_with_league_average_1500():
    elo = Elo(k=20, s=0.1, a=1)  # a = 1 lifts promoted teams the most
    run_elo(two_league_history(), elo)
    for ratings in elo.season_start.values():
        assert sum(ratings.values()) / len(ratings) == pytest.approx(1500)


def test_same_date_games_share_home_advantage_and_dont_see_each_other():
    # 120 earlier games, two thirds home wins, give a large H.
    rows = [("E0", "2000-01", f"2000-08-{1 + i // 10:02d}", f"T{i % 10}", f"T{(i + 1) % 10}",
             1 if i % 3 else 0, 0) for i in range(120)]
    last_day = [("E0", "2000-01", "2000-09-30", "T0", "T5", 0, 5),
                ("E0", "2000-01", "2000-09-30", "T1", "T6", 0, 5)]
    matches = games(*rows, *last_day)
    out = run_elo(matches, Elo(k=20, s=0, a=0))
    h = out["home_advantage"].iloc[-2:]
    assert h.iloc[0] == h.iloc[1] > 0
    # Changing one game's result on that day doesn't change the other's values.
    changed = matches.copy()
    changed.loc[changed.index[-2], ["home_goals", "away_goals"]] = [9, 0]
    out2 = run_elo(changed, Elo(k=20, s=0, a=0))
    pd.testing.assert_series_equal(out.iloc[-1], out2.iloc[-1])


def test_settings_file_has_the_tuned_values():
    with open(SETTINGS_PATH) as f:
        settings = json.load(f)
    assert {"K", "s", "a", "n_home"} <= set(settings)
    assert settings["n_home"] == 380
    assert settings["tuning"]["mse"] < settings["tuning"]["mse_always_average"]


@pytest.mark.skipif(not HAVE_DATA, reason="needs downloaded data")
def test_tuning_never_uses_test_seasons():
    # tune() drops 2019-20 on before running anything. Check by tuning a tiny
    # grid on data where every later game is nonsense: the score is unchanged.
    import elo as elo_module
    from load_data import load_matches

    matches = load_matches()
    later = matches["season"] >= "2019-20"
    scrambled = matches.copy()
    scrambled.loc[later, "home_goals"] = 9
    grids = (elo_module.K_GRID, elo_module.S_GRID, elo_module.A_GRID)
    try:
        elo_module.K_GRID, elo_module.S_GRID, elo_module.A_GRID = [10], [0.2], [0, 0.5]
        a, _ = elo_module.tune(matches)
        b, _ = elo_module.tune(scrambled)
    finally:
        elo_module.K_GRID, elo_module.S_GRID, elo_module.A_GRID = grids
    assert a["tuning"]["mse"] == b["tuning"]["mse"]
