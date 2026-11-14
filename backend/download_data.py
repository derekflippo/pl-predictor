"""Download Premier League results from football-data.co.uk.

Saves one CSV per season to backend/data/raw/, for example E0_2425.csv for
the 2024-25 season. Past seasons are downloaded once and then reused. The
current season is always downloaded again, because new results are added
to it every week.

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
URL = "https://football-data.co.uk/mmz4281/{code}/E0.csv"  # E0 = Premier League
FIRST_SEASON = 2000


def current_season_start(today=None):
    """Year the current season started. A season runs from August to May."""
    today = today or date.today()
    return today.year if today.month >= 8 else today.year - 1


def season_code(start_year):
    """2024 -> '2425', the code football-data uses for the 2024-25 season."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def download_season(start_year, refresh=False):
    """Download one season's CSV. Returns 'downloaded' or 'cached'."""
    code = season_code(start_year)
    path = RAW_DIR / f"E0_{code}.csv"
    if path.exists() and not refresh:
        return "cached"

    request = urllib.request.Request(
        URL.format(code=code), headers={"User-Agent": "pl-predictor"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        path.write_bytes(response.read())
    time.sleep(0.5)  # be polite to the server
    return "downloaded"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="first", type=int, default=FIRST_SEASON,
                        help=f"first season's start year (default {FIRST_SEASON})")
    parser.add_argument("--refresh", action="store_true",
                        help="download every season again, not just the current one")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    current = current_season_start()

    for year in range(args.first, current + 1):
        label = f"{year}-{(year + 1) % 100:02d}"
        try:
            status = download_season(year, refresh=args.refresh or year == current)
            print(f"{label}: {status}")
        except urllib.error.URLError as error:
            print(f"{label}: FAILED ({error})")


if __name__ == "__main__":
    main()
