"""Combine the downloaded season CSVs into one clean table of matches.

football-data.co.uk uses short column names (FTHG, HST, ...) and the set of
columns changes between seasons. This turns every season of every league
into the same readable columns, one row per match, sorted by date.

Usage:
    python backend/load_data.py   # writes backend/data/matches.csv and prints a summary

From other code:
    from load_data import load_matches
    matches = load_matches()                     # Premier League and Championship
    premier = load_matches(leagues=["E0"])       # Premier League only
"""

import csv
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"

LEAGUES = {"E0": "Premier League", "E1": "Championship"}
MATCHES_PER_SEASON = {"E0": 380, "E1": 552}  # 20 and 24 teams

# football-data column -> our column name
COLUMNS = {
    "Date": "date",
    "Time": "kickoff",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",  # H = home win, D = draw, A = away win
    "HTHG": "home_goals_ht",
    "HTAG": "away_goals_ht",
    "Referee": "referee",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
    "HxG": "home_xg",  # only in recent seasons
    "AxG": "away_xg",
}

# Pre-match betting odds, best source first: the market average when the
# file has it, otherwise a single bookmaker (older seasons have fewer).
ODDS_SOURCES = ["Avg", "BbAv", "B365", "WH", "IW"]


def season_label(code):
    """'2425' -> '2024-25'."""
    return f"20{code[:2]}-{code[2:]}"


def load_season(path):
    """Load one league-season CSV (for example E0_2425.csv) with our column names."""
    league, code = Path(path).stem.split("_")

    with open(path, encoding="utf-8-sig", errors="replace") as f:
        n_cols = len(next(csv.reader(f)))
    # Some old files (2003-04, 2004-05) have rows ending in extra empty fields.
    # Cut those rows down to the header's length.
    raw = pd.read_csv(path, encoding="utf-8-sig", encoding_errors="replace",
                      engine="python", on_bad_lines=lambda row: row[:n_cols])
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "FTR"])  # drop blank rows

    df = raw[[c for c in COLUMNS if c in raw.columns]].rename(columns=COLUMNS)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed")
    df.insert(1, "league", league)
    df.insert(2, "season", season_label(code))

    for outcome, name in [("H", "odds_home"), ("D", "odds_draw"), ("A", "odds_away")]:
        cols = [src + outcome for src in ODDS_SOURCES if src + outcome in raw.columns]
        # first available bookmaker in priority order, per match
        df[name] = raw[cols].bfill(axis=1).iloc[:, 0] if cols else pd.NA

    return df


def load_matches(raw_dir=RAW_DIR, leagues=tuple(LEAGUES)):
    """All downloaded seasons of the given leagues as one DataFrame, oldest first."""
    paths = sorted(p for league in leagues for p in Path(raw_dir).glob(f"{league}_*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No season files in {raw_dir}. Run: python backend/download_data.py"
        )
    matches = pd.concat([load_season(p) for p in paths], ignore_index=True)
    return matches.sort_values(["date", "kickoff", "league"],
                               na_position="first").reset_index(drop=True)


def main():
    matches = load_matches()
    out = DATA_DIR / "matches.csv"
    matches.to_csv(out, index=False)
    print(f"{len(matches)} matches saved to {out}\n")

    latest = matches["season"].max()
    for league, name in LEAGUES.items():
        m = matches[matches["league"] == league]
        per_season = m.groupby("season").size()
        short = per_season[(per_season != MATCHES_PER_SEASON[league])
                           & (per_season.index != latest)]
        share = m["result"].value_counts(normalize=True).reindex(["H", "D", "A"]) * 100
        print(f"{name}: {len(m)} matches, {len(per_season)} seasons, "
              f"{m['date'].min():%Y-%m-%d} to {m['date'].max():%Y-%m-%d}")
        print(f"  {latest} so far: {per_season.get(latest, 0)} matches")
        print(f"  Past seasons with the wrong number of matches: "
              f"{dict(short) if len(short) else 'none'}")
        print(f"  Home wins {share['H']:.1f}%, draws {share['D']:.1f}%, "
              f"away wins {share['A']:.1f}%")
        print(f"  Missing odds: {int(m['odds_home'].isna().sum())}, "
              f"missing shot stats: {int(m['home_shots'].isna().sum())}\n")


if __name__ == "__main__":
    main()
