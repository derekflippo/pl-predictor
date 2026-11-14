"""This season's Premier League fixtures, with team names matched to football-data.

openfootball supplies the schedule: dates, kickoff times and matchweeks.
Results come from football-data.co.uk, which is more up to date.

Fixtures are matched to results on (season, home_team, away_team), never on
date. Each pair of teams meets once per season at a given home ground, so a
postponed game still finds its result when it is finally played.

Kickoff times are UK local time, as in the source.

Usage:
    python backend/fixtures.py   # summary, plus the next matchweek's games

From other code:
    from fixtures import load_fixtures, upcoming
"""

import csv
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd

from load_data import RAW_DIR, load_matches

TEAM_NAMES = Path(__file__).parent / "team_names.csv"
RESULT_COLUMNS = ["home_goals", "away_goals", "result", "played_on"]


class UnknownTeamError(ValueError):
    """An openfootball team name that team_names.csv doesn't cover."""


def load_team_names(path=TEAM_NAMES):
    """openfootball name -> football-data name."""
    with open(path, newline="", encoding="utf-8") as f:
        return {row["openfootball"]: row["football_data"] for row in csv.DictReader(f)}


def latest_fixtures_file(raw_dir=RAW_DIR):
    """Path of the newest downloaded fixtures file, or None."""
    paths = sorted(Path(raw_dir).glob("fixtures_*.json"))
    return paths[-1] if paths else None


def load_fixtures(path, team_names=None):
    """All fixtures in an openfootball season file, using football-data team names.

    Raises UnknownTeamError if a team isn't in team_names.csv, so a newly
    promoted club can't be silently dropped.
    """
    team_names = team_names if team_names is not None else load_team_names()
    season = Path(path).stem.removeprefix("fixtures_")  # e.g. '2026-27'
    games = json.loads(Path(path).read_text(encoding="utf-8"))["matches"]

    fixtures = pd.DataFrame({
        "season": season,
        "matchweek": [int(re.search(r"\d+", g["round"]).group()) for g in games],
        "date": pd.to_datetime([g["date"] for g in games]),
        "kickoff": [g.get("time") for g in games],
        "home_team": [g["team1"] for g in games],
        "away_team": [g["team2"] for g in games],
    })

    unknown = sorted((set(fixtures["home_team"]) | set(fixtures["away_team"]))
                     - set(team_names))
    if unknown:
        raise UnknownTeamError(
            f"Unknown openfootball team name(s): {unknown}. Add each one to "
            f"{TEAM_NAMES.name} with its football-data name."
        )
    for side in ["home_team", "away_team"]:
        fixtures[side] = fixtures[side].map(team_names)
    return fixtures


def add_results(fixtures, matches):
    """Fixtures with football-data's result attached; empty where not played yet.

    played_on is the date the game was actually played, which can differ
    from the fixture date when a game was moved.
    """
    results = (matches.loc[matches["league"] == "E0",
                           ["season", "home_team", "away_team", "date",
                            "home_goals", "away_goals", "result"]]
               .rename(columns={"date": "played_on"}))
    return fixtures.merge(results, on=["season", "home_team", "away_team"],
                          how="left", validate="one_to_one")


def upcoming(fixtures, matches):
    """Fixtures that have no result yet, soonest first."""
    with_results = add_results(fixtures, matches)
    unplayed = with_results[with_results["result"].isna()]
    return (unplayed.drop(columns=RESULT_COLUMNS)
            .sort_values(["date", "kickoff"]).reset_index(drop=True))


def main():
    path = latest_fixtures_file()
    if path is None:
        raise SystemExit("No fixtures file. Run: python backend/download_data.py")

    fixtures = load_fixtures(path)
    matches = load_matches(leagues=["E0"])
    with_results = add_results(fixtures, matches)
    played = with_results["result"].notna()
    season = fixtures["season"].iloc[0]
    print(f"{season}: {len(fixtures)} fixtures, {played.sum()} played, "
          f"{(~played).sum()} to go")

    moved = with_results[played & (with_results["date"] != with_results["played_on"])]
    print(f"Played on a different date than openfootball lists: {len(moved)}")

    overdue = with_results[~played & (with_results["date"].dt.date < date.today())]
    print(f"Listed date has passed but no result yet: {len(overdue)}")
    for _, g in overdue.iterrows():
        print(f"  {g['date']:%Y-%m-%d}  {g['home_team']} v {g['away_team']}")

    todo = upcoming(fixtures, matches)
    todo = todo[todo["date"].dt.date >= date.today()]
    if len(todo):
        week = todo["matchweek"].iloc[0]
        print(f"\nNext up, matchweek {week}:")
        for _, g in todo[todo["matchweek"] == week].iterrows():
            print(f"  {g['date']:%a %d %b} {g['kickoff'] or '':>5}  "
                  f"{g['home_team']} v {g['away_team']}")


if __name__ == "__main__":
    main()
