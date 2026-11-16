"""Turn the match history into one row of numbers per match, from before kickoff.

Every value in a row comes only from matches on earlier dates. The same
code builds rows for played matches (to train on) and for games not yet
played (to predict), so the two are always computed the same way.

The columns are described in .plans/milestone-2-features-spec.md ("Features").

Usage:
    python backend/features.py   # writes backend/data/features.csv and a summary

From other code:
    from features import build_features, features_for
    table = build_features(matches)             # one row per played match
    rows = features_for(upcoming, matches)      # rows for games not yet played
"""

from collections import deque

import numpy as np
import pandas as pd

from elo import Elo, actual_score, date_groups, season_teams

WINDOWS = (5, 10)
VENUE_WINDOW = 5
REST_CAP = 21  # days; stops the summer break from dominating rest_days

ID_COLUMNS = ["league", "season", "date", "home_team", "away_team"]
TARGET_COLUMNS = ["result", "home_goals", "away_goals"]
TEAM_FEATURES = (
    ["elo"]
    + [f"{name}_{n}" for name in ("ppg", "gf", "ga", "sotf", "sota", "vs_exp")
       for n in WINDOWS]
    + [f"venue_ppg_{VENUE_WINDOW}", "rest_days", "came_up", "came_down"]
    + [f"games_{n}" for n in WINDOWS]
)
FEATURE_COLUMNS = (["home_advantage", "elo_expected"]
                   + [f"{side}_{f}" for side in ("home", "away") for f in TEAM_FEATURES])


def _average(values):
    values = [v for v in values if not np.isnan(v)]
    return sum(values) / len(values) if values else np.nan


class MatchState:
    """Everything known about the teams at one moment: ratings and recent games.

    Walk the matches in date order: row() for every game on a date, then
    apply() for every game on that date.
    """

    def __init__(self, elo, teams):
        self.elo = elo
        self.teams = teams  # {season: {league: set of teams}}
        # team -> its last games, in either league. Each game is
        # (points, goals for, goals against, shots on target for,
        #  shots on target against, actual score minus Elo's expected score)
        self.recent = {}
        self.venue = {}      # (team, "home" or "away") -> points in recent games there
        self.last_date = {}  # team -> date of its last league game

    def start_season(self, season):
        if season != self.elo.season:
            self.elo.start_season(season, self.teams[season])

    def row(self, league, date, home, away):
        """Feature values for a game, from the current state. Doesn't change it."""
        r_home, r_away, h, expected = self.elo.predict(league, home, away)
        row = {"home_advantage": h, "elo_expected": expected}
        changes = self.elo.changes.get((league, self.elo.season))
        for side, team, rating in (("home", home, r_home), ("away", away, r_away)):
            games = list(self.recent.get(team, ()))
            values = {"elo": rating}
            for n in WINDOWS:
                window = games[-n:]
                columns = list(zip(*window)) if window else [()] * 6
                values[f"games_{n}"] = len(window)
                for name, column in zip(("ppg", "gf", "ga", "sotf", "sota", "vs_exp"), columns):
                    values[f"{name}_{n}"] = _average(column)
            values[f"venue_ppg_{VENUE_WINDOW}"] = _average(self.venue.get((team, side), ()))
            last = self.last_date.get(team)
            values["rest_days"] = (np.nan if last is None
                                   else min((date - last).days, REST_CAP))
            change = changes.get(team) if changes is not None else np.nan
            values["came_up"] = np.nan if changes is None else float(change == "up")
            values["came_down"] = np.nan if changes is None else float(change == "down")
            for name in TEAM_FEATURES:
                row[f"{side}_{name}"] = values[name]
        return row

    def apply(self, match, expected):
        """Add a played match to the state. `expected` is Elo's pre-match E."""
        hg, ag = match.home_goals, match.away_goals
        score = actual_score(hg, ag)
        points = {1.0: (3, 0), 0.5: (1, 1), 0.0: (0, 3)}[score]
        sot_h, sot_a = match.home_shots_on_target, match.away_shots_on_target
        sides = (
            (match.home_team, "home", points[0], hg, ag, sot_h, sot_a, score - expected),
            (match.away_team, "away", points[1], ag, hg, sot_a, sot_h, expected - score),
        )
        for team, side, pts, gf, ga, sotf, sota, vs_exp in sides:
            self.recent.setdefault(team, deque(maxlen=max(WINDOWS))).append(
                (pts, gf, ga, sotf, sota, vs_exp))
            self.venue.setdefault((team, side), deque(maxlen=VENUE_WINDOW)).append(pts)
            self.last_date[team] = match.date
        self.elo.update(match.league, match.home_team, match.away_team, hg, ag, expected)


def _walk(matches, elo, teams, collect):
    """Run the state over played matches. With collect=True, return their rows."""
    state = MatchState(elo or Elo.from_settings(), teams)
    rows = {}
    for season, group in date_groups(matches):
        state.start_season(season)
        found = [(m, state.row(m.league, m.date, m.home_team, m.away_team)) for m in group]
        if collect:
            rows.update((m.Index, row) for m, row in found)
        for m, row in found:
            state.apply(m, row["elo_expected"])
    return state, rows


def _season_teams(matches, extra):
    teams = season_teams(matches)
    for season, leagues in (extra or {}).items():
        for league, names in leagues.items():
            teams.setdefault(season, {}).setdefault(league, set()).update(names)
    return teams


def build_features(matches, elo=None, teams=None):
    """One row of features per played match, in date order.

    `elo` defaults to the tuned settings. `teams` ({season: {league: set}})
    adds team names to the per-season lists read from `matches`.
    """
    matches = matches.reset_index(drop=True).sort_values("date", kind="stable")
    _, rows = _walk(matches, elo, _season_teams(matches, teams), collect=True)
    features = pd.DataFrame.from_dict(rows, orient="index", columns=FEATURE_COLUMNS)
    table = pd.concat([matches[ID_COLUMNS + TARGET_COLUMNS], features], axis=1)
    return table.reset_index(drop=True)


def features_for(games, matches, elo=None, teams=None):
    """Feature rows for games not yet played, using everything in `matches`.

    `games` needs the columns league, date, home_team, away_team. The rows
    use the state after the last played match, so a game weeks away gets
    today's ratings and form. Results in between aren't simulated.
    """
    matches = matches.reset_index(drop=True).sort_values("date", kind="stable")
    state, _ = _walk(matches, elo, _season_teams(matches, teams), collect=False)
    games = games.reset_index(drop=True)
    rows = [state.row(g.league, pd.Timestamp(g.date), g.home_team, g.away_team)
            for g in games.itertuples()]
    out = games[["league", "date", "home_team", "away_team"]].copy()
    out.insert(1, "season", state.elo.season)
    return pd.concat([out, pd.DataFrame(rows, columns=FEATURE_COLUMNS)], axis=1)


def main():
    from load_data import DATA_DIR, load_matches

    table = build_features(load_matches())
    out = DATA_DIR / "features.csv"
    table.to_csv(out, index=False)
    print(f"{len(table)} rows, {len(table.columns)} columns, saved to {out}\n")
    for league in ("E0", "E1"):
        part = table[table["league"] == league]
        print(f"{league}: {len(part)} rows. Missing values per feature:")
        missing = part[FEATURE_COLUMNS].isna().sum()
        trained = part[part["season"] >= "2002-03"][FEATURE_COLUMNS].isna().sum()
        for name in FEATURE_COLUMNS:
            if missing[name]:
                print(f"  {name:<22} {missing[name]:>5}  (from 2002-03 on: {trained[name]})")
        print()


if __name__ == "__main__":
    main()
