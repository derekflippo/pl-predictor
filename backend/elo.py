"""Elo ratings for the Premier League (E0) and the Championship (E1).

Each league has its own rating scale. After every game the winner takes
rating points from the loser: more for a surprise, more for a big win.
Between seasons, teams that change league are converted to the new
league's scale, and every rating moves part of the way back to average.

Every season starts with each league's average at 1500. The formula and
its settings are described in
.plans/milestone-2-features-spec.md ("Elo").

Usage:
    python backend/elo.py --tune   # tunes K, s and a; writes backend/elo_settings.json

From other code:
    from elo import Elo, run_elo
    elo = Elo.from_settings()            # the tuned settings
    ratings = run_elo(matches, elo)      # pre-match ratings, H and E for every match
"""

import argparse
import itertools
import json
import math
from collections import deque
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import paired

SETTINGS_PATH = Path(__file__).parent / "elo_settings.json"

LEAGUE_ORDER = ["E0", "E1"]  # top league first
START_RATING = 1500.0

# Home advantage is measured from the league's last N_HOME games. Until a
# league has MIN_HOME_GAMES games, H is 0. That only affects the first weeks
# of 2000-01, a warm-up season that is never used for training.
N_HOME = 380
MIN_HOME_GAMES = 100

# Tuning: only these Premier League seasons are scored. Games from 2019-20 on
# are the final test set for milestone 3 and are never used here.
TUNE_FIRST, TUNE_LAST = "2002-03", "2018-19"
K_GRID = [5, 7.5, 10, 12.5, 15, 20, 25, 30, 40]
S_GRID = [0, 0.1, 0.2, 0.33, 0.5]
A_GRID = [0, 0.25, 0.5, 0.75, 1]


def expected_score(r_home, r_away, home_advantage):
    """Elo's expected score for the home team, between 0 and 1."""
    return 1 / (1 + 10 ** (-(r_home + home_advantage - r_away) / 400))


def actual_score(home_goals, away_goals):
    """1 for a home win, 0.5 for a draw, 0 for an away win."""
    if home_goals > away_goals:
        return 1.0
    if home_goals < away_goals:
        return 0.0
    return 0.5


def margin_multiplier(goal_difference):
    """Bigger wins count more, with diminishing returns (World Football Elo)."""
    n = abs(goal_difference)
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    return (11 + n) / 8


def home_advantage_points(p):
    """Turn an average home score p (0 to 1) into rating points.

    This is the rating gap that would make Elo expect p, so p = 0.58 gives
    about 56 points and p = 0.5 gives 0.
    """
    return 400 * math.log10(p / (1 - p))


def season_teams(matches):
    """{season: {league: set of teams}}, read from who played in each season."""
    teams = {}
    for (season, league), games in matches.groupby(["season", "league"]):
        teams.setdefault(season, {})[league] = (
            set(games["home_team"]) | set(games["away_team"]))
    return teams


def _mean(values, fallback):
    values = list(values)
    return sum(values) / len(values) if values else fallback


class Elo:
    """Ratings for every team, updated one game at a time.

    Use it in date order:
      1. start_season() when the first game of a new season comes up
      2. predict() for every game on a date
      3. update() for every game on that date, with the E from step 2

    Keeping steps 2 and 3 apart means no game sees a result from its own
    date, including through the home advantage.
    """

    def __init__(self, k, s, a, n_home=N_HOME):
        self.k, self.s, self.a, self.n_home = k, s, a, n_home
        self.season = None
        self.ratings = {}      # league -> {team: rating}
        self.home_scores = {}  # league -> recent actual home scores
        # Snapshots for the checks, keyed by (league, season):
        self.season_start = {}  # ratings just after the season boundary
        self.season_end = {}    # ratings just before the next boundary
        # (league, season) -> {team: "up", "down" or None}. Missing for the
        # first season, where nobody's previous league is known.
        self.changes = {}

    @classmethod
    def from_settings(cls, path=SETTINGS_PATH):
        """An Elo with the tuned settings from elo_settings.json."""
        with open(path) as f:
            settings = json.load(f)
        return cls(settings["K"], settings["s"], settings["a"], settings["n_home"])

    def home_advantage(self, league):
        """H in rating points, from the league's last n_home games."""
        scores = self.home_scores.get(league)
        if scores is None or len(scores) < MIN_HOME_GAMES:
            return 0.0
        return home_advantage_points(sum(scores) / len(scores))

    def rating(self, league, team):
        try:
            return self.ratings[league][team]
        except KeyError:
            raise KeyError(f"{team} has no {league} rating in season {self.season}. "
                           "Is it in that league's team list for the season?") from None

    def predict(self, league, home, away):
        """(home rating, away rating, H, E) before the game."""
        r_home, r_away = self.rating(league, home), self.rating(league, away)
        h = self.home_advantage(league)
        return r_home, r_away, h, expected_score(r_home, r_away, h)

    def update(self, league, home, away, home_goals, away_goals, expected):
        """Apply a result. `expected` is the E that predict() gave before the date."""
        score = actual_score(home_goals, away_goals)
        change = self.k * margin_multiplier(home_goals - away_goals) * (score - expected)
        ratings = self.ratings[league]
        ratings[home] += change
        ratings[away] -= change
        if league not in self.home_scores:
            self.home_scores[league] = deque(maxlen=self.n_home)
        self.home_scores[league].append(score)

    def start_season(self, season, teams):
        """Season boundary. `teams` is {league: set of teams in the new season}.

        1. Teams that changed league are converted to the new league's scale.
        2. Every rating moves part of the way (s) to its league's average.
        3. Every league is shifted so its average is 1500 again.

        Step 3 matters because promoted teams usually arrive above the
        teams they replace (when a > 0), which would make the top league's
        average creep up every year. Shifting a whole league doesn't change
        any difference between its teams, so predictions are unaffected.
        """
        if self.season is not None and season <= self.season:
            raise ValueError(f"Season {season} comes after {self.season} in the data")
        old = self.ratings
        for league, ratings in old.items():
            self.season_end[(league, self.season)] = dict(ratings)

        new = {}
        for league, now in teams.items():
            if league not in old:  # first season: everyone starts level
                new[league] = {team: START_RATING for team in now}
                continue
            new[league], self.changes[(league, season)] = self._convert(league, now, old, teams)

        for ratings in new.values():
            mean = _mean(ratings.values(), START_RATING)
            for team, r in ratings.items():
                ratings[team] = START_RATING + (1 - self.s) * (r - mean)

        self.ratings = new
        self.season = season
        for league, ratings in new.items():
            self.season_start[(league, season)] = dict(ratings)

    def _convert(self, league, now, old, teams):
        """New-season ratings for one league, before the summer shrink."""
        level = LEAGUE_ORDER.index(league)
        upper = LEAGUE_ORDER[level - 1] if level > 0 else None
        lower = LEAGUE_ORDER[level + 1] if level + 1 < len(LEAGUE_ORDER) else None
        prev = old[league]
        prev_mean = _mean(prev.values(), START_RATING)

        went_up = [t for t in prev if t in teams.get(upper, ())]
        went_down = [t for t in prev if t not in now and t not in teams.get(upper, ())]
        # A team arriving takes the place of the teams that left in the
        # other direction, plus a share (a) of how far it was above or below
        # its old league's average.
        base_from_above = _mean((prev[t] for t in went_up), prev_mean)
        base_from_below = _mean((prev[t] for t in went_down), prev_mean)

        ratings, changes = {}, {}
        for team in now:
            if team in prev:
                ratings[team], changes[team] = prev[team], None
            elif upper in old and team in old[upper]:
                above = old[upper]
                ratings[team] = base_from_above + self.a * (
                    above[team] - _mean(above.values(), START_RATING))
                changes[team] = "down"
            elif lower in old and team in old[lower]:
                below = old[lower]
                ratings[team] = base_from_below + self.a * (
                    below[team] - _mean(below.values(), START_RATING))
                changes[team] = "up"
            else:
                # New to our data, so from a lower league (League One for the
                # Championship), or back after years away. Old ratings are
                # stale, so it's treated as new.
                ratings[team], changes[team] = base_from_below, "up"
        return ratings, changes


def date_groups(matches):
    """Yield (season, rows) for each match date, oldest first.

    `rows` are the date's matches as named tuples with an `Index` field
    (the row's label in `matches`).
    """
    ordered = matches.sort_values("date", kind="stable")
    rows = list(ordered.itertuples())
    i = 0
    while i < len(rows):
        j = i
        while j < len(rows) and rows[j].date == rows[i].date:
            j += 1
        group = rows[i:j]
        seasons = {r.season for r in group}
        if len(seasons) > 1:
            raise ValueError(f"Games from seasons {seasons} share the date {rows[i].date}")
        yield group[0].season, group
        i = j


def run_elo(matches, elo, teams=None):
    """Run Elo over played matches in date order.

    Returns a DataFrame with the same index as `matches` and the pre-match
    values home_elo, away_elo, home_advantage and elo_expected. `elo` ends
    holding the ratings after the last match. `teams` overrides the
    per-season team lists (default: read from `matches`).
    """
    teams = teams or season_teams(matches)
    out = {}
    for season, group in date_groups(matches):
        if season != elo.season:
            elo.start_season(season, teams[season])
        preds = [elo.predict(m.league, m.home_team, m.away_team) for m in group]
        for m, pred in zip(group, preds):
            out[m.Index] = pred
        for m, pred in zip(group, preds):
            elo.update(m.league, m.home_team, m.away_team,
                       m.home_goals, m.away_goals, pred[3])
    result = pd.DataFrame.from_dict(
        out, orient="index",
        columns=["home_elo", "away_elo", "home_advantage", "elo_expected"])
    return result.reindex(matches.index)


# ----------------------------------------------------------------------------
# Tuning


def tune(matches):
    """Score every (K, s, a) on Premier League games from 2002-03 to 2018-19.

    Returns the settings dict that is saved to elo_settings.json, and a
    printable report.
    """
    data = matches[matches["season"] <= TUNE_LAST]  # nothing from the test seasons
    scored = ((data["league"] == "E0") & (data["season"] >= TUNE_FIRST)).to_numpy()
    actual = np.array([actual_score(h, a) for h, a in
                       zip(data["home_goals"], data["away_goals"])])[scored]
    seasons = data["season"].to_numpy()[scored]
    teams = season_teams(data)

    errors = {}
    for k, s, a in itertools.product(K_GRID, S_GRID, A_GRID):
        elo = Elo(k, s, a)
        expected = run_elo(data, elo, teams)["elo_expected"].to_numpy()[scored]
        errors[(k, s, a)] = (actual - expected) ** 2

    # Premier League games involving a promoted team, in its first season up.
    # Who was promoted doesn't depend on the settings.
    home, away = (data[col].to_numpy()[scored] for col in ("home_team", "away_team"))
    up = {(season, team) for (league, season), changes in elo.changes.items()
          if league == "E0" for team, change in changes.items() if change == "up"}
    promoted = np.array([(se, h) in up or (se, a) in up
                         for se, h, a in zip(seasons, home, away)])

    mse = {combo: float(err.mean()) for combo, err in errors.items()}
    ranked = sorted(mse, key=mse.get)
    best = ranked[0]
    lines = [f"Scored {len(actual)} Premier League games, {TUNE_FIRST} to {TUNE_LAST}.",
             f"Always predicting the average home score: MSE {actual.var():.5f}",
             f"{len(mse)} combinations of K, s and a (N_home fixed at {N_HOME}).", "",
             "Best 10 (difference from the best, paired standard error, seasons "
             "where the best scored lower, of 17):",
             f"  {'K':>5} {'s':>5} {'a':>5}   {'MSE':>8}   {'diff':>8} {'± SE':>8}   seasons"]
    season_list = sorted(set(seasons))

    def season_wins(combo):
        return sum(errors[best][seasons == se].mean() < errors[combo][seasons == se].mean()
                   for se in season_list)

    for combo in ranked[:10]:
        diff, se = paired(errors[combo], errors[best])
        wins = "" if combo == best else f"{season_wins(combo):>2}/{len(season_list)}"
        lines.append(f"  {combo[0]:>5} {combo[1]:>5} {combo[2]:>5}   {mse[combo]:.5f}   "
                     f"{diff:+.5f} {se:8.5f}   {wins}")
    worst = ranked[-1]
    diff, se = paired(errors[worst], errors[best])
    lines.append(f"  Worst: K={worst[0]}, s={worst[1]}, a={worst[2]}: {mse[worst]:.5f} "
                 f"({diff:+.5f} ± {se:.5f}, best lower in {season_wins(worst)}/{len(season_list)} seasons)")
    within = [c for c in ranked[1:] if paired(errors[c], errors[best])[0]
              < paired(errors[c], errors[best])[1]]
    lines.append(f"  {len(within)} combinations are within 1 standard error of the best.")

    # Does Championship data help? Best a > 0 against best a = 0, judged on
    # the games a can affect.
    best_flat = min((c for c in mse if c[2] == 0), key=mse.get)
    best_carry = min((c for c in mse if c[2] > 0), key=mse.get)
    diff_p, se_p = paired(errors[best_flat][promoted], errors[best_carry][promoted])
    diff_all, se_all = paired(errors[best_flat], errors[best_carry])
    lines += ["", f"Does Championship strength help (a > 0)? "
              f"{promoted.sum()} of the games involve a promoted team.",
              f"  best a = 0: K={best_flat[0]}, s={best_flat[1]}  "
              f"MSE {mse[best_flat]:.5f} (promoted-team games "
              f"{errors[best_flat][promoted].mean():.5f})",
              f"  best a > 0: K={best_carry[0]}, s={best_carry[1]}, a={best_carry[2]}  "
              f"MSE {mse[best_carry]:.5f} (promoted-team games "
              f"{errors[best_carry][promoted].mean():.5f})",
              f"  a = 0 minus a > 0, promoted-team games: {diff_p:+.5f} ± {se_p:.5f}",
              f"  a = 0 minus a > 0, all games:           {diff_all:+.5f} ± {se_all:.5f}",
              f"  With K={best[0]}, s={best[1]}, promoted-team games by a: " + ", ".join(
                  f"{a}: {errors[(best[0], best[1], a)][promoted].mean():.5f}" for a in A_GRID)]
    helps = diff_p > 2 * se_p
    lines.append("  Verdict: " + (
        "a > 0 is clearly better (more than 2 standard errors)." if helps else
        "no clear gain from a > 0 (within 2 standard errors)."))

    settings = {
        "K": best[0], "s": best[1], "a": best[2], "n_home": N_HOME,
        "tuning": {
            "seasons": f"{TUNE_FIRST} to {TUNE_LAST}",
            "games": int(len(actual)),
            "score": "mean squared error between E and the actual home score",
            "mse": round(mse[best], 6),
            "mse_always_average": round(float(actual.var()), 6),
            "combinations_within_1_se": len(within),
            "grid": {"K": K_GRID, "s": S_GRID, "a": A_GRID, "n_home": "fixed"},
            "championship_carry_over": {
                "promoted_team_games": int(promoted.sum()),
                "best_a0": {"K": best_flat[0], "s": best_flat[1],
                            "mse_promoted": round(float(errors[best_flat][promoted].mean()), 6)},
                "best_a_above_0": {"K": best_carry[0], "s": best_carry[1], "a": best_carry[2],
                                   "mse_promoted": round(float(errors[best_carry][promoted].mean()), 6)},
                "difference": round(diff_p, 6),
                "standard_error": round(se_p, 6),
                "clear_gain": bool(helps),
            },
            "tuned_on": date.today().isoformat(),
        },
    }
    return settings, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tune", action="store_true",
                        help="tune K, s and a, and write elo_settings.json")
    args = parser.parse_args()
    if not args.tune:
        parser.print_help()
        return

    from load_data import load_matches
    settings, report = tune(load_matches())
    print(report)
    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)
        f.write("\n")
    print(f"\nChose K={settings['K']}, s={settings['s']}, a={settings['a']}, "
          f"N_home={settings['n_home']}. Saved to {SETTINGS_PATH}")


if __name__ == "__main__":
    main()
