"""Predict upcoming Premier League games on your own computer.

Every run builds the features from backend/data/matches.csv, trains the
model on every finished match with the settings frozen in
backend/model_settings.json (a few seconds), and prints the home, draw and
away probabilities.

Get the latest results first:
    python backend/download_data.py
    python backend/load_data.py

Usage:
    python backend/predict.py                                  # the next matchweek
    python backend/predict.py --matchweek 14                   # that matchweek's unplayed games
    python backend/predict.py --home Arsenal --away Chelsea    # any pairing, as if played today
    python backend/predict.py --home Arsenal --away Chelsea --date 2026-12-26

Team names are football-data's, as in matches.csv: "Man United", "Nott'm Forest".
"""

import argparse
from datetime import date

import numpy as np
import pandas as pd

from features import features_and_state, rows_for
from fixtures import latest_fixtures_file, load_fixtures, upcoming
from load_data import load_matches
from train import TRAIN_FROM, add_derived, load_settings, model_features, predict, train_final

CLOSE_CALL = 0.05  # "Close call" when the top two probabilities are within 5 points


class PredictError(ValueError):
    """Something the user asked for can't be predicted; the message says why."""


def train_model(matches, settings):
    """The model trained on every finished match, the feature rows, and the state after them.

    The state (ratings and recent form after the last result) is what
    predict_games() uses for upcoming games.
    """
    table, state = features_and_state(matches)
    table = add_derived(table)
    return train_final(table, settings), table, state


def latest_result(state):
    """The date of the last match the state has seen."""
    return max(state.last_date.values())


def favourite(home, away, probs):
    """The favourite's name, "Draw", or "Close call" when the top two are within 5 points."""
    order = np.argsort(probs)[::-1]
    if probs[order[0]] - probs[order[1]] < CLOSE_CALL:
        return "Close call"
    return [home, "Draw", away][order[0]]


def feature_rows(games, state):
    """The model's input rows for unplayed games, from the ratings and form in `state`.

    A game listed before the last result (postponed, say) is treated as
    played on that date, so rest days are never negative.
    """
    games = games.reset_index(drop=True)
    dates = games["date"].clip(lower=latest_result(state))
    return add_derived(rows_for(games.assign(league="E0", date=dates), state))


def predict_games(games, state, model, settings):
    """Probabilities for games not played yet.

    `games` needs date, home_team and away_team. Each game gets the ratings
    and form in `state`: everything up to the last downloaded result (see
    feature_rows).
    """
    games = games.reset_index(drop=True)
    rows = feature_rows(games, state)
    probs = predict(model, rows, model_features(settings))
    out = games.copy()
    out[["p_home", "p_draw", "p_away"]] = probs
    out["favourite"] = [favourite(h, a, p)
                        for h, a, p in zip(out["home_team"], out["away_team"], probs)]
    return out


def check_season(fixtures, matches):
    """Stop if the fixtures' season has no results yet.

    Promoted and relegated teams' ratings are converted at the season's
    first game, so before then they'd be on the wrong scale.
    """
    season = fixtures["season"].iloc[0]
    played = matches[(matches["league"] == "E0") & (matches["season"] == season)]
    if played.empty:
        raise PredictError(
            f"No {season} results yet. Predictions start once the season's first "
            "results are downloaded, because promoted and relegated teams' ratings "
            "are converted at their first game.")


def check_date(day, latest):
    """Stop if a named game's date is before the last result.

    The ratings and form already include every downloaded result, so a
    prediction "as if played" earlier would use games from after it.
    """
    if day < latest.date():
        raise PredictError(f"--date must be on or after the latest result, "
                           f"{latest:%d %b %Y}: the ratings and form already include "
                           "every game up to then.")


def missing_results(fixtures, matches, today):
    """Fixtures listed after the latest result and before today, with no result.

    These are almost always games played since the last download. Games
    listed before the latest result and still unplayed are left out: they
    were usually postponed.
    """
    todo = upcoming(fixtures, matches[matches["league"] == "E0"])
    latest = matches["date"].max()
    listed = todo["date"]
    return todo[(listed > latest) & (listed.dt.date < today)].reset_index(drop=True)


def next_games(fixtures, matches, today, matchweek=None):
    """The unplayed games of `matchweek`, or of the next matchweek from `today` on."""
    todo = upcoming(fixtures, matches[matches["league"] == "E0"])
    if matchweek is None:
        future = todo[todo["date"].dt.date >= today]
        if future.empty:
            raise PredictError("No unplayed fixtures from today on.")
        matchweek = int(future["matchweek"].iloc[0])
    games = todo[todo["matchweek"] == matchweek]
    if games.empty:
        raise PredictError(f"Matchweek {matchweek} has no unplayed games.")
    return matchweek, games.reset_index(drop=True)


def season_teams(matches):
    """The teams in the latest Premier League season in the data."""
    pl = matches[matches["league"] == "E0"]
    latest = pl[pl["season"] == pl["season"].max()]
    return set(latest["home_team"]) | set(latest["away_team"])


def check_teams(home, away, teams):
    unknown = [t for t in (home, away) if t not in teams]
    if unknown:
        raise PredictError(f"Unknown team name: {', '.join(unknown)}. "
                           f"This season's teams: {', '.join(sorted(teams))}.")
    if home == away:
        raise PredictError("The home and away teams are the same.")


def show(predictions):
    for g in predictions.itertuples():
        kickoff = getattr(g, "kickoff", None) or ""
        print(f"  {g.date:%a %d %b} {kickoff:>5}  {g.home_team:>15} v {g.away_team:<15}"
              f"  home {g.p_home:3.0%}  draw {g.p_draw:3.0%}  away {g.p_away:3.0%}"
              f"   {g.favourite}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--matchweek", type=int, help="predict this matchweek's unplayed games")
    parser.add_argument("--home", help="home team, football-data name (e.g. Arsenal)")
    parser.add_argument("--away", help="away team, football-data name (e.g. Chelsea)")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today(),
                        help="match date for --home/--away, YYYY-MM-DD (default: today)")
    args = parser.parse_args()
    if (args.home is None) != (args.away is None):
        parser.error("give both --home and --away")

    try:
        matches = load_matches()
        path = latest_fixtures_file()
        fixtures = load_fixtures(path) if path else None
        if fixtures is not None:
            check_season(fixtures, matches)
        if args.home:
            check_teams(args.home, args.away, season_teams(matches))
            check_date(args.date, matches["date"].max())
            games = pd.DataFrame({"date": [pd.Timestamp(args.date)],
                                  "home_team": [args.home], "away_team": [args.away]})
            title = f"{args.home} v {args.away}, as if played on {args.date:%d %b %Y}"
        else:
            if fixtures is None:
                raise PredictError("No fixtures file. Run: python backend/download_data.py")
            week, games = next_games(fixtures, matches, date.today(), args.matchweek)
            title = f"Matchweek {week}, {fixtures['season'].iloc[0]}"
    except PredictError as e:
        raise SystemExit(str(e))

    if fixtures is not None:
        missing = missing_results(fixtures, matches, date.today())
        if len(missing):
            print(f"Warning: {len(missing)} game(s) listed between the latest result and today "
                  "have no result yet. Run download_data.py and load_data.py first, so the "
                  "ratings and form include them.\n")
    settings = load_settings()
    model, table, state = train_model(matches, settings)
    trained = table[(table["season"] >= TRAIN_FROM) & table["result"].notna()
                    & (table["league"] == "E0")]
    print(f"Trained on {len(trained):,} Premier League matches, {trained['season'].min()} "
          f"to {trained['season'].max()}. Latest result: {matches['date'].max():%d %b %Y}.")
    print(f"\n{title}")
    show(predict_games(games, state, model, settings))
    if not args.home:
        print("\nKickoff times are UK time. openfootball lists Saturday 15:00 until "
              "TV picks are made.")


if __name__ == "__main__":
    main()
