"""A local web page for the predictor, built with Streamlit.

It uses the same code as predict.py: the model is trained once when the
page starts (about 8 seconds), then every prediction is instant.

Usage:
    pip install -r backend/requirements-app.txt
    streamlit run backend/app.py
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import download_data  # noqa: E402
from fixtures import latest_fixtures_file, load_fixtures, upcoming  # noqa: E402
from load_data import DATA_DIR, RAW_DIR, load_matches  # noqa: E402
from predict import (PredictError, check_date, check_season, missing_results,  # noqa: E402
                     next_games, predict_games, season_teams, train_model)
from train import TRAIN_FROM, load_settings  # noqa: E402

st.set_page_config(page_title="Premier League predictor", page_icon="⚽", layout="centered")


@st.cache_resource(show_spinner="Training the model on every finished match…")
def trained(data_stamp):
    """The matches, settings, model and state. data_stamp changes when the data does."""
    matches = load_matches()
    settings = load_settings()
    model, table, state = train_model(matches, settings)
    rows = table[(table["season"] >= TRAIN_FROM) & table["result"].notna()
                 & (table["league"] == "E0")]
    return matches, settings, model, state, len(rows)


def data_stamp():
    """The newest modification time of the downloaded files, or None if there are none."""
    files = list(RAW_DIR.glob("*.csv")) + list(RAW_DIR.glob("fixtures_*.json"))
    return max(f.stat().st_mtime for f in files) if files else None


def update_data():
    """download_data.py then load_data.py: past seasons once, this season again."""
    failed = download_data.download_all(log=lambda message: None)
    load_matches().to_csv(DATA_DIR / "matches.csv", index=False)
    return failed


def probability_table(predictions):
    """One row per game, with the three probabilities as percentage bars."""
    when = predictions["date"].dt.strftime("%a %d %b")
    if "kickoff" in predictions:
        when = when + " " + predictions["kickoff"].fillna("")
    table = pd.DataFrame({
        "When": when.str.strip(),
        "Home": predictions["home_team"],
        "Away": predictions["away_team"],
        "Home win": predictions["p_home"] * 100,
        "Draw": predictions["p_draw"] * 100,
        "Away win": predictions["p_away"] * 100,
        "Favourite": predictions["favourite"],
    })
    bar = {name: st.column_config.ProgressColumn(name, min_value=0, max_value=100,
                                                 format="%.0f%%", width="small")
           for name in ("Home win", "Draw", "Away win")}
    st.dataframe(table, hide_index=True, use_container_width=True, column_config=bar)


st.title("Premier League predictor")
st.caption("Home win, draw and away win probabilities from a logistic regression on Elo "
           "ratings and recent form. Not betting advice.")

info, action = st.columns([3, 1])
if action.button("Update data", use_container_width=True,
                 help="Downloads this season's latest results and fixtures"):
    with st.spinner("Downloading…"):
        failed = update_data()
    st.cache_resource.clear()
    if failed:
        st.warning("Some downloads failed:\n\n" + "\n\n".join(failed))
    else:
        st.success("Data updated.")

stamp = data_stamp()
if stamp is None:
    st.info("No data yet. Click **Update data** to download it "
            "(about 55 files, a minute or two the first time).")
    st.stop()

matches, settings, model, state, trained_on = trained(stamp)
latest = matches["date"].max()
path = latest_fixtures_file()
fixtures = load_fixtures(path) if path else None

info.caption(f"Latest result: **{latest:%d %b %Y}**. Trained on {trained_on:,} Premier League "
             f"matches (elastic net, C {settings['c']}, {len(settings['features'])} features).")

today = date.today()
if fixtures is not None:
    missing = missing_results(fixtures, matches, today)
    if len(missing):
        st.warning(f"{len(missing)} game(s) listed since the latest result have no result yet. "
                   "Click **Update data** so the ratings and form include them.")

upcoming_tab, any_tab = st.tabs(["Upcoming matchweek", "Any match"])

with upcoming_tab:
    try:
        if fixtures is None:
            raise PredictError("No fixtures file yet. Click **Update data**.")
        check_season(fixtures, matches)
        todo = upcoming(fixtures, matches[matches["league"] == "E0"])
        weeks = sorted(todo["matchweek"].unique().tolist())
        next_week, _ = next_games(fixtures, matches, today)
        week = st.selectbox("Matchweek", weeks, index=weeks.index(next_week))
        _, games = next_games(fixtures, matches, today, week)
        probability_table(predict_games(games, state, model, settings))
        st.caption('Kickoff times are UK time. openfootball lists Saturday 15:00 until TV '
                   'picks are made. "Close call" means the top two are within 5 points.')
    except PredictError as error:
        st.error(str(error))

with any_tab:
    teams = sorted(season_teams(matches))
    left, right = st.columns(2)
    home = left.selectbox("Home team", teams, index=0)
    away = right.selectbox("Away team", teams, index=1)
    day = st.date_input("Date", value=max(today, latest.date()), min_value=latest.date())
    try:
        if fixtures is not None:
            check_season(fixtures, matches)
        if home == away:
            raise PredictError("Pick two different teams.")
        check_date(day, latest)
        game = pd.DataFrame({"date": [pd.Timestamp(day)], "home_team": [home],
                             "away_team": [away]})
        p = predict_games(game, state, model, settings).iloc[0]
        cols = st.columns(3)
        cols[0].metric(f"{home} win", f"{p['p_home']:.0%}")
        cols[1].metric("Draw", f"{p['p_draw']:.0%}")
        cols[2].metric(f"{away} win", f"{p['p_away']:.0%}")
        st.progress(float(p["p_home"]), text=f"Home win {p['p_home']:.0%}")
        st.progress(float(p["p_draw"]), text=f"Draw {p['p_draw']:.0%}")
        st.progress(float(p["p_away"]), text=f"Away win {p['p_away']:.0%}")
        st.write(f"**Favourite:** {p['favourite']}")
    except PredictError as error:
        st.error(str(error))
