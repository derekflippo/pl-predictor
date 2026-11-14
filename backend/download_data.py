"""Download match data.

- Results from football-data.co.uk for the Premier League (E0) and the
  Championship (E1), one CSV per league and season, for example
  data/raw/E0_2425.csv for the 2024-25 Premier League.
- This season's Premier League fixtures from openfootball, saved as
  data/raw/fixtures_2026-27.json.

Past seasons are downloaded once and then reused. The current season's
files are always downloaded again, because results are added every week.

Usage:
    python backend/download_data.py              # 2000-01 up to the current season
    python backend/download_data.py --from 2015  # 2015-16 onwards
    python backend/download_data.py --refresh    # download every season again
"""

import argparse
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

RAW_DIR = Path(__file__).parent / "data" / "raw"
RESULTS_URL = "https://football-data.co.uk/mmz4281/{code}/{league}.csv"
FIXTURES_URL = ("https://raw.githubusercontent.com/openfootball/football.json/"
                "master/{season}/en.1.json")
LEAGUES = {"E0": "Premier League", "E1": "Championship"}
FIRST_SEASON = 2000


def current_season_start(today=None):
    """Year the current season started. A season runs from August to May."""
    today = today or date.today()
    return today.year if today.month >= 8 else today.year - 1


def season_code(start_year):
    """2024 -> '2425', the code football-data uses for the 2024-25 season."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_name(start_year):
    """2024 -> '2024-25'."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def fetch(url, path):
    """Download url to path."""
    request = urllib.request.Request(url, headers={"User-Agent": "pl-predictor"})
    with urllib.request.urlopen(request, timeout=30) as response:
        path.write_bytes(response.read())
    time.sleep(0.5)  # be polite to the server


def download_results(league, start_year, refresh=False):
    """Download one league's results for one season. Returns 'downloaded' or 'cached'."""
    code = season_code(start_year)
    path = RAW_DIR / f"{league}_{code}.csv"
    if path.exists() and not refresh:
        return "cached"
    fetch(RESULTS_URL.format(code=code, league=league), path)
    return "downloaded"


def download_fixtures(start_year):
    """Download one season's Premier League fixture list from openfootball."""
    season = season_name(start_year)
    fetch(FIXTURES_URL.format(season=season), RAW_DIR / f"fixtures_{season}.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="first", type=int, default=FIRST_SEASON,
                        help=f"first season's start year (default {FIRST_SEASON})")
    parser.add_argument("--refresh", action="store_true",
                        help="download every season again, not just the current one")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    current = current_season_start()
    failed = False

    for league, name in LEAGUES.items():
        counts = {"downloaded": 0, "cached": 0}
        for year in range(args.first, current + 1):
            try:
                status = download_results(league, year,
                                          refresh=args.refresh or year == current)
                counts[status] += 1
            except urllib.error.URLError as error:
                print(f"{name} {season_name(year)}: FAILED ({error})")
                failed = True
        print(f"{name}: {counts['downloaded']} downloaded, {counts['cached']} already here")

    try:
        download_fixtures(current)
        print(f"Fixtures {season_name(current)}: downloaded")
    except urllib.error.URLError as error:
        print(f"Fixtures {season_name(current)}: FAILED ({error})")
        failed = True

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
