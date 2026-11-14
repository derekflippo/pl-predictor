import pandas as pd
import pytest

from load_data import MATCHES_PER_SEASON, RAW_DIR, load_matches, load_season, season_label

HEADER = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,WHH,WHD,WHA"
HAVE_DATA = any(RAW_DIR.glob("E0_*.csv"))


def write_csv(tmp_path, name, *rows, header=HEADER):
    path = tmp_path / name
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def test_season_label():
    assert season_label("0304") == "2003-04"
    assert season_label("2627") == "2026-27"


def test_league_and_season_come_from_file_name(tmp_path):
    path = write_csv(tmp_path, "E1_2425.csv",
                     "E1,10/08/24,Leeds,Portsmouth,3,3,D,20,8,1.4,4.5,7")
    row = load_season(path).iloc[0]
    assert row["league"] == "E1"
    assert row["season"] == "2024-25"
    assert row["home_goals"] == 3 and row["result"] == "D"


def test_rows_with_extra_empty_fields_are_kept(tmp_path):
    # Like the real 2003-04 and 2004-05 files: some rows end in extra commas.
    path = write_csv(tmp_path, "E0_0304.csv",
                     "E0,16/08/03,Arsenal,Everton,2,1,H,10,5,1.5,3.5,6",
                     "E0,16/08/03,Leeds,Newcastle,2,2,D,8,9,2.5,3.2,2.8,,,,,")
    df = load_season(path)
    assert len(df) == 2
    assert df.loc[1, "home_team"] == "Leeds"
    assert df.loc[1, "odds_away"] == 2.8


def test_blank_rows_are_dropped(tmp_path):
    path = write_csv(tmp_path, "E0_0001.csv",
                     "E0,19/08/00,Charlton,Man City,4,0,H,17,8,2.2,3.2,3",
                     ",,,,,,,,,,,")
    assert len(load_season(path)) == 1


def test_dates_are_day_first_with_two_or_four_digit_years(tmp_path):
    path = write_csv(tmp_path, "E0_1920.csv",
                     "E0,02/09/01,Arsenal,Everton,2,1,H,10,5,1.5,3.5,6",
                     "E0,03/09/2019,Leeds,Newcastle,2,2,D,8,9,2.5,3.2,2.8")
    dates = load_season(path)["date"].tolist()
    assert dates == [pd.Timestamp("2001-09-02"), pd.Timestamp("2019-09-03")]


def test_odds_use_the_best_available_source(tmp_path):
    header = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,AvgH,AvgD,AvgA"
    path = write_csv(tmp_path, "E0_2425.csv",
                     "E0,16/08/24,Man United,Fulham,1,0,H,1.6,4.2,5.5,1.62,4.1,5.3",
                     "E0,17/08/24,Ipswich,Liverpool,0,2,A,7.5,5,1.4,,,",
                     header=header)
    df = load_season(path)
    assert df["odds_home"].tolist() == [1.62, 7.5]  # market average, else Bet365
    assert df["odds_away"].tolist() == [5.3, 1.4]


@pytest.mark.skipif(not HAVE_DATA, reason="run backend/download_data.py first")
def test_downloaded_past_seasons_have_every_match():
    matches = load_matches()
    latest = matches["season"].max()
    counts = matches[matches["season"] != latest].groupby(["league", "season"]).size()
    wrong = {key: n for key, n in counts.items() if n != MATCHES_PER_SEASON[key[0]]}
    assert wrong == {}
