# pl-predictor

Predicts Premier League match outcomes (home win, draw, away win) with a
machine learning model, and shows the predictions for upcoming games on a
web page.

The model uses football data only (results, form, team ratings). Betting
odds are not an input; they are the benchmark the model is measured
against. Predictions are locked before kickoff and scored afterwards, so
the site's track record is a real one.

## Project layout

```
backend/     Python: data download, features, model training, predictions
frontend/    React + Vite + Tailwind static site (coming in milestone 6)
.plans/      Design specs
docs/        Methodology explainer (source of a private page; reused in milestone 6)
```

The site is static: a daily GitHub Action retrains the model and writes the
predictions as JSON files, and the page reads them. There is no server.
See [.plans/pl-predictor-v1-spec.md](.plans/pl-predictor-v1-spec.md) for
the full design.

## Setup

Requires Python 3.10+.

```bash
pip install -r backend/requirements.txt
```

## Get the data

```bash
python backend/download_data.py   # Premier League + Championship since 2000-01, and this season's fixtures
python backend/load_data.py       # combines the results into backend/data/matches.csv
python backend/fixtures.py        # this season's fixtures matched to results, and the next matchweek
```

Run `download_data.py` again after each matchweek to get the latest
results. Past seasons are only downloaded once.

`backend/data/` is not committed to git. Anyone who clones the repo
rebuilds it with the commands above.

### What's in `matches.csv`

One row per match, about 24,500 rows from August 2000 to today: the
Premier League and the Championship. The Championship is there so that
promoted teams have a history.

- `league` (`E0` = Premier League, `E1` = Championship)
- `date`, `season`, `home_team`, `away_team`
- `home_goals`, `away_goals`, `result` (`H` / `D` / `A`), half-time goals
- shots, shots on target, corners, fouls, yellow and red cards
- `odds_home`, `odds_draw`, `odds_away`: pre-match betting odds (market
  average where available, otherwise a single bookmaker)
- `home_xg`, `away_xg`: expected goals, only from 2026-27

### Fixtures and team names

The schedule (dates, kickoff times, matchweeks) comes from openfootball.
Results always come from football-data. The two sources spell teams
differently ("Manchester United FC" vs "Man United"), so
[backend/team_names.csv](backend/team_names.csv) maps one to the other.
It covers every Premier League team since 2010-11. A newly promoted team
that isn't in the file stops the pipeline with an error that names it,
so it's never silently dropped.

Fixtures are matched to results by season and the two teams, not by date:
openfootball's dates can be out of date when games are moved.

## Tests

```bash
pip install -r backend/requirements-dev.txt
pytest
```

Some tests check the downloaded data and are skipped if you haven't run
`download_data.py` yet.

## Roadmap

0. [x] **Data pipeline**: download and clean every Premier League season
   since 2000-01
1. [x] **More data**: Championship results (for promoted teams) and this
   season's fixtures from openfootball
2. [ ] **Features**: Elo ratings and recent form, with a test that no
   feature uses data from after the match
3. [ ] **Model**: logistic regression, evaluated season by season against
   "always pick home" (45.6%) and the bookmakers
4. [ ] **Predictions**: JSON for the site, and an append-only file of
   predictions locked before kickoff
5. [ ] **Daily GitHub Action**: retrain and lock predictions every day;
   the track record starts here
6. [ ] **Website (v1)**: upcoming games, track record, how it works;
   hosted on GitHub Pages
7. [ ] **v1.1**: team picker, key factors, head-to-head
8. [ ] **Later**: gradient boosting, a Poisson goals model

## Data sources

- Results, match stats and odds:
  [football-data.co.uk](https://www.football-data.co.uk/)
- Upcoming fixtures (planned):
  [openfootball](https://github.com/openfootball/football.json) (public domain)

Not betting advice.
