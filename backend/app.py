"""A local web page for the predictor, built with Streamlit.

It uses the same code as predict.py: the model is trained once when the
page starts (about 8 seconds), then every prediction is instant. The Model
tab and "Why this prediction?" show what the model learned (explain.py).

Usage:
    pip install -r backend/requirements-app.txt
    streamlit run backend/app.py
"""

import sys
from datetime import date
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import download_data  # noqa: E402
from explain import contributions, elo_curve, weight_table  # noqa: E402
from fixtures import latest_fixtures_file, load_fixtures, upcoming  # noqa: E402
from load_data import DATA_DIR, RAW_DIR, load_matches  # noqa: E402
from predict import (PredictError, check_date, check_season, feature_rows,  # noqa: E402
                     missing_results, next_games, predict_games, season_teams, train_model)
from train import TRAIN_FROM, load_settings, model_features  # noqa: E402

# Home win, draw, away win: one color each, the same on every chart. Checked for
# color-blind separation and contrast on both light and dark backgrounds.
HOME, DRAW, AWAY = "#3987e5", "#199e70", "#d95926"

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


def push_chart(table, value, title):
    """Horizontal bars: right (blue) pushes toward a home win, left (orange) toward an away win."""
    data = table.assign(direction=np.where(table[value] >= 0, "toward home win", "toward away win"))
    return alt.Chart(data).mark_bar().encode(
        x=alt.X(f"{value}:Q", title=title),
        y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
        color=alt.Color("direction:N", title=None, legend=alt.Legend(orient="bottom"),
                        scale=alt.Scale(domain=["toward home win", "toward away win"],
                                        range=[HOME, AWAY])),
        tooltip=[alt.Tooltip("label:N", title="Input"),
                 alt.Tooltip(f"{value}:Q", title=title, format="+.3f")],
    ).properties(height=alt.Step(24))  # 24 px per bar, even in a tab that starts hidden


def curve_chart(curve):
    """Home, draw and away probability against the Elo gap."""
    long = curve.melt("gap", var_name="outcome", value_name="probability")
    long["outcome"] = long["outcome"].map({"home": "Home win", "draw": "Draw", "away": "Away win"})
    return alt.Chart(long).mark_line(strokeWidth=2.5).encode(
        x=alt.X("gap:Q", title="Elo gap: home rating minus away rating"),
        y=alt.Y("probability:Q", title="Probability", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("outcome:N", title=None, legend=alt.Legend(orient="bottom"),
                        scale=alt.Scale(domain=["Home win", "Draw", "Away win"],
                                        range=[HOME, DRAW, AWAY])),
        tooltip=[alt.Tooltip("gap:Q", title="Elo gap"), alt.Tooltip("outcome:N"),
                 alt.Tooltip("probability:Q", format=".0%")],
    ).properties(height=320)


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

features = model_features(settings)
upcoming_tab, any_tab, model_tab = st.tabs(["Upcoming matchweek", "Any match", "Model"])

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

        st.subheader("Why this prediction?")
        pushes, start = contributions(model, features, feature_rows(game, state))
        st.altair_chart(push_chart(pushes.head(10), "toward_home", "Push toward a home win"),
                        use_container_width=True)
        total = start + pushes["toward_home"].sum()
        st.caption(
            f"The 10 inputs that moved this prediction most. Each bar is the input's weight "
            f"times how far it is from average (in standard deviations). The model starts at "
            f"{start:+.2f} for every game, the home advantage it learned, and all {len(pushes)} "
            f"inputs add {total - start:+.2f}, for a total of {total:+.2f}. That total is "
            f"log(P(home) / P(away)): a home win is {np.exp(total):.1f} times as likely as an "
            f"away win. Inputs that overlap, like recent points and the Elo gap, can push in "
            f"opposite directions.")
    except PredictError as error:
        st.error(str(error))

with model_tab:
    st.subheader("How the probabilities change with the Elo gap")
    home_advantage = state.elo.home_advantage("E0")
    st.altair_chart(curve_chart(elo_curve(model, features, home_advantage)),
                    use_container_width=True)
    st.caption(
        f"Every other input is held at its average over the training matches, and Elo's "
        f"expected score follows the gap, using today's league home advantage "
        f"({home_advantage:.0f} rating points). Two equally rated teams (gap 0) still favour "
        f"the home side, and the draw peaks when the teams are close.")

    st.subheader("Weights")
    weights = weight_table(model, features)
    st.altair_chart(push_chart(weights, "toward_home", "Home-win weight minus away-win weight"),
                    use_container_width=True)
    zero = weights.index[(weights[["H", "D", "A"]] == 0).all(axis=1)]
    st.caption(
        f"How much one standard deviation of each input moves the home-win score compared "
        f"with the away-win score. Elastic net set {len(zero)} input(s) to exactly zero"
        + (f": {', '.join(weights.loc[zero, 'label'])}." if len(zero) else "."))
    with st.expander("All weights: home win, draw and away win"):
        st.dataframe(weights.rename(columns={"label": "Input", "H": "Home win", "D": "Draw",
                                             "A": "Away win", "toward_home": "Home minus away"}),
                     hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="%+.3f")
                                    for c in ("Home win", "Draw", "Away win", "Home minus away")})
