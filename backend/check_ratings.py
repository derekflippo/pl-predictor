"""Checks that the Elo ratings work. They catch bugs; they don't measure
how good the predictions are (that's milestone 3).

1. End-of-season Elo ranks teams much like the final table.
2. Elo alone predicts better than the average home/draw/away shares.
4. For information only: the starting ratings of promoted teams that went
   straight back down, against the ones that stayed up. Championship
   strength barely predicts survival in our data, so this has no pass
   mark. The unit tests check the conversion arithmetic.

Check 3 (the ratings chart) is in notebooks/evaluation.ipynb.

Usage:
    python backend/check_ratings.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from elo import Elo, run_elo, season_teams

CHECK1_FIRST = "2002-03"
CHECK1_MIN_AVERAGE = 0.8
CHECK2_TRAIN = ("2002-03", "2014-15")
CHECK2_SCORE = ("2015-16", "2018-19")
OUTCOMES = ["H", "D", "A"]


def run(matches, elo=None):
    """Run Elo over every match. Returns the Elo (with its snapshots) and its outputs."""
    elo = elo or Elo.from_settings()
    return elo, run_elo(matches, elo)


def league_table(games):
    """Final table from results: points, then goal difference, then goals scored."""
    home = pd.DataFrame({"team": games["home_team"], "gf": games["home_goals"],
                         "ga": games["away_goals"]})
    away = pd.DataFrame({"team": games["away_team"], "gf": games["away_goals"],
                         "ga": games["home_goals"]})
    rows = pd.concat([home, away])
    rows["points"] = np.select([rows["gf"] > rows["ga"], rows["gf"] == rows["ga"]], [3, 1], 0)
    table = rows.groupby("team")[["points", "gf", "ga"]].sum()
    table["gd"] = table["gf"] - table["ga"]
    table = table.sort_values(["points", "gd", "gf"], ascending=False)
    table["position"] = range(1, len(table) + 1)
    return table


def check_end_of_season(matches, elo):
    """Check 1: Spearman correlation between Elo rank and table position."""
    pl = matches[matches["league"] == "E0"]
    finished = sorted(s for s in pl["season"].unique()
                      if s >= CHECK1_FIRST and s < elo.season)
    seasons = []
    for season in finished:
        table = league_table(pl[pl["season"] == season])
        end = pd.Series(elo.season_end[("E0", season)]).reindex(table.index)
        start = pd.Series(elo.season_start[("E0", season)]).reindex(table.index)
        elo_rank = end.rank(ascending=False)
        gaps = (elo_rank - table["position"]).abs().sort_values(ascending=False)
        seasons.append({
            "season": season,
            "end": table["position"].corr(-end, method="spearman"),
            "start": table["position"].corr(-start, method="spearman"),
            "biggest_gaps": [(team, int(table.loc[team, "position"]), int(elo_rank[team]))
                             for team in gaps.index[:3]],
        })
    result = pd.DataFrame(seasons).set_index("season")
    average = result["end"].mean()
    return {"passed": bool(average >= CHECK1_MIN_AVERAGE), "average": float(average),
            "average_start": float(result["start"].mean()), "seasons": result}


def check_elo_alone(matches, ratings):
    """Check 2: one-input logistic regression on elo_expected vs base rates."""
    pl = matches["league"] == "E0"
    train = pl & matches["season"].between(*CHECK2_TRAIN)
    score = pl & matches["season"].between(*CHECK2_SCORE)
    x = ratings[["elo_expected"]]
    model = LogisticRegression().fit(x[train], matches.loc[train, "result"])
    probs = model.predict_proba(x[score])
    elo_loss = log_loss(matches.loc[score, "result"], probs, labels=model.classes_)

    shares = matches.loc[train, "result"].value_counts(normalize=True).reindex(model.classes_)
    base = np.tile(shares.to_numpy(), (int(score.sum()), 1))
    base_loss = log_loss(matches.loc[score, "result"], base, labels=model.classes_)
    return {"passed": bool(elo_loss < base_loss), "elo_log_loss": float(elo_loss),
            "base_log_loss": float(base_loss), "games": int(score.sum()),
            "accuracy": float((model.predict(x[score]) == matches.loc[score, "result"]).mean())}


def check_promoted(matches, elo):
    """Check 4: starting rating of promoted teams that went down vs stayed up."""
    teams = season_teams(matches)
    seasons = sorted(teams)
    rows = []
    for season, next_season in zip(seasons, seasons[1:]):
        changes = elo.changes.get(("E0", season))
        if changes is None:
            continue
        for team, change in changes.items():
            if change == "up":
                rows.append({"season": season, "team": team,
                             "start": elo.season_start[("E0", season)][team],
                             "went_down": team not in teams[next_season]["E0"]})
    rows = pd.DataFrame(rows)
    down = rows.loc[rows["went_down"], "start"]
    stayed = rows.loc[~rows["went_down"], "start"]
    gap = stayed.mean() - down.mean()
    se = np.sqrt(down.var() / len(down) + stayed.var() / len(stayed))
    return {"down_mean": float(down.mean()), "down_count": len(down),
            "stayed_mean": float(stayed.mean()), "stayed_count": len(stayed),
            "gap": float(gap), "standard_error": float(se), "teams": rows}


def main():
    from load_data import load_matches

    matches = load_matches()
    elo, ratings = run(matches)
    print(f"Elo settings: K={elo.k}, s={elo.s}, a={elo.a}, N_home={elo.n_home}\n")

    c1 = check_end_of_season(matches, elo)
    seasons = c1["seasons"]
    print(f"1. End-of-season Elo vs the final table, {seasons.index[0]} to {seasons.index[-1]}: "
          f"{'PASS' if c1['passed'] else 'FAIL'}")
    print(f"   Average Spearman {c1['average']:.3f} (pass at {CHECK1_MIN_AVERAGE} or more)")
    print("   Lowest 5 seasons (team: table position / Elo rank, biggest gaps):")
    for season, row in seasons.nsmallest(5, "end").iterrows():
        gaps = ", ".join(f"{t} {p}/{r}" for t, p, r in row["biggest_gaps"])
        print(f"     {season}  {row['end']:.3f}   {gaps}")
    print(f"   For information, start-of-season ratings vs the final table: "
          f"average {c1['average_start']:.3f}")
    print("   Lowest 5:", ", ".join(f"{s} {v:.3f}"
                                    for s, v in seasons["start"].nsmallest(5).items()))

    c2 = check_elo_alone(matches, ratings)
    print(f"\n2. Elo alone vs base rates, trained {CHECK2_TRAIN[0]} to {CHECK2_TRAIN[1]}, "
          f"scored {CHECK2_SCORE[0]} to {CHECK2_SCORE[1]} ({c2['games']} games): "
          f"{'PASS' if c2['passed'] else 'FAIL'}")
    print(f"   Log loss: Elo {c2['elo_log_loss']:.4f}, base rates {c2['base_log_loss']:.4f}. "
          f"Elo accuracy {c2['accuracy']:.1%}")

    c4 = check_promoted(matches, elo)
    print("\n4. Promoted teams' starting ratings (for information, no pass or fail):")
    print(f"   Went straight back down: {c4['down_mean']:.1f} on average "
          f"({c4['down_count']} teams)")
    print(f"   Stayed up:               {c4['stayed_mean']:.1f} on average "
          f"({c4['stayed_count']} teams)")
    print(f"   Stayed up minus went down: {c4['gap']:+.1f} ± {c4['standard_error']:.1f}")

    if not (c1["passed"] and c2["passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
